using System;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Tensor = TorchSharp.torch.Tensor;

namespace Genesis.Sentience.Learning
{
    [Serializable]
    public class PPOConfig
    {
        public int NumSteps = 2048;
        public int NumEpochs = 10;
        public int MiniBatchSize = 256;
        public float ClipEpsilon = 0.2f;
        public float ValueLossCoef = 0.5f;
        public float EntropyCoef = 0.003f;
        public float MaxGradNorm = 0.5f;
        public float Gamma = 0.99f;
        public float GAELambda = 0.95f;
        public float TargetKL = 0.16f;
        public float LearningRate = 3e-4f;
        public bool AnnealLR = true;
        public float MinLearningRate = 1e-7f;
        public bool NormalizeAdvantages = true;
        public bool ClipValueLoss = true;
        public int Hidden1 = 256;
        public int Hidden2 = 256;

        [UnityEngine.Tooltip("Initial log_std value. 0 = std=1.0. Negative = less exploration.")]
        public float LogStdInit = 0f;
    }

    /// <summary>
    /// PPO actor-critic agent with double-buffered CPU inference.
    /// Training happens on GPU (MPS/CUDA). Inference (GetActionAndValue,
    /// GetDeterministicAction) uses CPU clones swapped atomically, so the
    /// main thread never touches MPS — avoiding concurrent access crashes.
    /// </summary>
    public class PPOAgent : IDisposable
    {
        private readonly Sequential _critic;
        private readonly Sequential _actorMean;
        private readonly Parameter _actorLogStd;
        private optim.Optimizer _optimizer;

        public readonly Device Device;
        public readonly int ObsDim;
        public readonly int ActDim;

        private readonly float _maxGradNorm;
        private int _updateCount;

        // Double-buffered CPU inference networks (main thread reads, never GPU)
        private Sequential _infActorA, _infActorB;
        private Sequential _infCriticA, _infCriticB;
        private Tensor _infLogStdA, _infLogStdB;
        private volatile int _infSlot; // 0 = A active, 1 = B active

        private Tensor _infObsTensor; // CPU tensor for inference
        private readonly int _infObsBytes;
        private readonly float[] _actionBuffer;

        public int UpdateCount => _updateCount;

        public PPOAgent(int obsDim, int actDim, PPOConfig config, Device device)
        {
            ObsDim = obsDim;
            ActDim = actDim;
            Device = device;
            _maxGradNorm = config.MaxGradNorm;

            _critic = Sequential(
                LayerInit(Linear(obsDim, config.Hidden1), Math.Sqrt(2)),
                Tanh(),
                LayerInit(Linear(config.Hidden1, config.Hidden2), Math.Sqrt(2)),
                Tanh(),
                LayerInit(Linear(config.Hidden2, 1), 1.0)
            );

            _actorMean = Sequential(
                LayerInit(Linear(obsDim, config.Hidden1), Math.Sqrt(2)),
                Tanh(),
                LayerInit(Linear(config.Hidden1, config.Hidden2), Math.Sqrt(2)),
                Tanh(),
                LayerInit(Linear(config.Hidden2, actDim), 0.01)
            );

            _actorLogStd = Parameter(torch.full(new long[] { 1, actDim }, config.LogStdInit, device: device));

            _critic.to(device);
            _actorMean.to(device);

            _optimizer = optim.Adam(AllParameters(), lr: config.LearningRate, eps: 1e-5);

            // CPU inference clones (double-buffered)
            _infActorA = BuildActorClone(obsDim, actDim, config);
            _infActorB = BuildActorClone(obsDim, actDim, config);
            _infCriticA = BuildCriticClone(obsDim, config);
            _infCriticB = BuildCriticClone(obsDim, config);
            _infLogStdA = torch.full(new long[] { 1, actDim }, config.LogStdInit);
            _infLogStdB = torch.full(new long[] { 1, actDim }, config.LogStdInit);
            _infSlot = 0;
            SyncInferenceWeights();

            _actionBuffer = new float[actDim];
            _infObsTensor = torch.zeros(1, obsDim, dtype: ScalarType.Float32);
            _infObsBytes = obsDim * sizeof(float);
        }

        private static Sequential BuildActorClone(int obsDim, int actDim, PPOConfig config)
        {
            var s = Sequential(
                LayerInit(Linear(obsDim, config.Hidden1), Math.Sqrt(2)),
                Tanh(),
                LayerInit(Linear(config.Hidden1, config.Hidden2), Math.Sqrt(2)),
                Tanh(),
                LayerInit(Linear(config.Hidden2, actDim), 0.01));
            s.to(torch.CPU);
            return s;
        }

        private static Sequential BuildCriticClone(int obsDim, PPOConfig config)
        {
            var s = Sequential(
                LayerInit(Linear(obsDim, config.Hidden1), Math.Sqrt(2)),
                Tanh(),
                LayerInit(Linear(config.Hidden1, config.Hidden2), Math.Sqrt(2)),
                Tanh(),
                LayerInit(Linear(config.Hidden2, 1), 1.0));
            s.to(torch.CPU);
            return s;
        }

        private static Module<Tensor, Tensor> LayerInit(Linear layer, double std, double biasConst = 0.0)
        {
            try
            {
                nn.init.orthogonal_(layer.weight, std);
            }
            catch (ExternalException)
            {
                // orthogonal_ requires LAPACK which isn't available on Android/ARM64.
                // Fall back to Kaiming uniform (the PyTorch default for Linear layers).
                // Weights will be overwritten when loading a saved model.
                nn.init.kaiming_uniform_(layer.weight, Math.Sqrt(5));
            }
            nn.init.constant_(layer.bias, biasConst);
            return layer;
        }

        private Parameter[] AllParameters()
        {
            var list = new System.Collections.Generic.List<Parameter>();
            foreach (var p in _critic.parameters()) list.Add(p);
            foreach (var p in _actorMean.parameters()) list.Add(p);
            list.Add(_actorLogStd);
            return list.ToArray();
        }

        /// <summary>
        /// Sample stochastic action and return (action, logProb, value).
        /// Uses CPU inference clones — safe to call from the main thread
        /// while the training thread runs GPU operations.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public (float[] action, float logProb, float value) GetActionAndValue(float[] obs)
        {
            using var scope = NewDisposeScope();
            float logProbVal, valueVal;

            // Snapshot the active slot (volatile read)
            var slot = _infSlot;
            var actor = slot == 0 ? _infActorA : _infActorB;
            var critic = slot == 0 ? _infCriticA : _infCriticB;
            var logStdTensor = slot == 0 ? _infLogStdA : _infLogStdB;

            using (no_grad())
            {
                _infObsTensor.bytes = MemoryMarshal.AsBytes<float>(obs.AsSpan());
                var obsTensor = _infObsTensor;

                var mean = actor.forward(obsTensor);
                var logStd = logStdTensor.clamp(-2.0f, 0.5f).expand_as(mean);
                var std = logStd.exp();

                var noise = torch.randn_like(mean);
                var action = mean + std * noise;

                var diff = action - mean;
                var logProb = (-0.5f * ((diff / std).pow(2) + 2f * logStd
                              + (float)Math.Log(2 * Math.PI))).sum(-1);

                var value = critic.forward(obsTensor).flatten();

                CopyTensorToBuffer(action, _actionBuffer);
                logProbVal = logProb.item<float>();
                valueVal = value.item<float>();
            }

            return (_actionBuffer, logProbVal, valueVal);
        }

        /// <summary>Get deterministic (mean) action for inference. CPU-only.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float[] GetDeterministicAction(float[] obs)
        {
            using var scope = NewDisposeScope();
            var actor = _infSlot == 0 ? _infActorA : _infActorB;
            using (no_grad())
            {
                _infObsTensor.bytes = MemoryMarshal.AsBytes<float>(obs.AsSpan());
                var mean = actor.forward(_infObsTensor);
                CopyTensorToBuffer(mean, _actionBuffer);
            }
            return _actionBuffer;
        }

        /// <summary>
        /// Evaluate batched actions for the PPO update.
        /// Returns (newLogProbs, values, entropy) as tensors for gradient computation.
        /// </summary>
        public (Tensor logProbs, Tensor values, Tensor entropy) EvaluateActions(
            Tensor obsBatch, Tensor actionBatch)
        {
            var mean = _actorMean.forward(obsBatch);
            var logStd = _actorLogStd.clamp(-2.0f, 0.5f).expand_as(mean);
            var std = logStd.exp();

            var diff = actionBatch - mean;
            var logProbs = (-0.5f * ((diff / std).pow(2) + 2f * logStd
                           + (float)Math.Log(2 * Math.PI))).sum(-1);

            var entropy = (0.5f * (1f + (float)Math.Log(2 * Math.PI)) + logStd).sum(-1);

            var values = _critic.forward(obsBatch).view(-1);

            return (logProbs, values, entropy);
        }

        /// <summary>Get value estimate for a batch of observations.</summary>
        public Tensor GetValue(Tensor obsBatch)
        {
            return _critic.forward(obsBatch).view(-1);
        }

        /// <summary>
        /// Run one PPO mini-batch update step.
        /// Returns (policyLoss, valueLoss, entropy, approxKL, clipFrac).
        /// </summary>
        public (float pgLoss, float vLoss, float ent, float kl, float clipFrac) UpdateStep(
            Tensor mbObs, Tensor mbActions, Tensor mbOldLogProbs,
            Tensor mbAdvantages, Tensor mbReturns, Tensor mbOldValues,
            PPOConfig config)
        {
            var (newLogProb, newValue, entropy) = EvaluateActions(mbObs, mbActions);
            var logRatio = newLogProb - mbOldLogProbs;
            var ratio = logRatio.exp();

            float approxKL, clipFrac;
            using (no_grad())
            {
                approxKL = ((ratio - 1) - logRatio).mean().item<float>();
                clipFrac = ((ratio - 1.0f).abs() > config.ClipEpsilon)
                    .to(ScalarType.Float32).mean().item<float>();
            }

            if (config.NormalizeAdvantages)
                mbAdvantages = (mbAdvantages - mbAdvantages.mean()) / (mbAdvantages.std() + 1e-8f);

            var pgLoss1 = -mbAdvantages * ratio;
            var pgLoss2 = -mbAdvantages * ratio.clamp(
                1 - config.ClipEpsilon, 1 + config.ClipEpsilon);
            var pgLoss = torch.max(pgLoss1, pgLoss2).mean();

            Tensor vLoss;
            if (config.ClipValueLoss)
            {
                var vUnclipped = (newValue - mbReturns).pow(2);
                var vClipped = mbOldValues + (newValue - mbOldValues)
                    .clamp(-config.ClipEpsilon, config.ClipEpsilon);
                var vLossClipped = (vClipped - mbReturns).pow(2);
                vLoss = 0.5f * torch.max(vUnclipped, vLossClipped).mean();
            }
            else
            {
                vLoss = 0.5f * (newValue - mbReturns).pow(2).mean();
            }

            var entropyLoss = entropy.mean();
            var loss = pgLoss - config.EntropyCoef * entropyLoss + vLoss * config.ValueLossCoef;

            _optimizer.zero_grad();
            loss.backward();
            if (_maxGradNorm > 0f)
                nn.utils.clip_grad_norm_(AllParameters(), _maxGradNorm);
            _optimizer.step();

            _updateCount++;

            return (pgLoss.item<float>(), vLoss.item<float>(),
                    entropyLoss.item<float>(), approxKL, clipFrac);
        }

        /// <summary>
        /// Copy GPU training weights to the staging CPU inference clone,
        /// then atomically swap the active slot. Lock-free for the reader.
        /// Call from the training thread after each PPO update round.
        /// </summary>
        public void SyncInferenceWeights()
        {
            using var scope = NewDisposeScope();

            int active = _infSlot;
            var stagingActor = active == 0 ? _infActorB : _infActorA;
            var stagingCritic = active == 0 ? _infCriticB : _infCriticA;
            var stagingLogStd = active == 0 ? _infLogStdB : _infLogStdA;

            using (no_grad())
            {
                CopyWeights(_actorMean, stagingActor);
                CopyWeights(_critic, stagingCritic);
                stagingLogStd.copy_(_actorLogStd.cpu());
            }

            // Atomic swap
            _infSlot = active == 0 ? 1 : 0;
        }

        private static void CopyWeights(Sequential src, Sequential dst)
        {
            using var scope = NewDisposeScope();
            var srcParams = src.named_parameters();
            var dstDict = new System.Collections.Generic.Dictionary<string, Parameter>();
            foreach (var (name, param) in dst.named_parameters())
                dstDict[name] = param;

            foreach (var (name, srcP) in srcParams)
            {
                if (dstDict.TryGetValue(name, out var dstP))
                    dstP.copy_(srcP.cpu());
            }
        }

        public void SetLearningRate(float lr)
        {
            foreach (var pg in _optimizer.ParamGroups)
                pg.LearningRate = lr;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CopyTensorToBuffer(Tensor t, float[] dst)
        {
            var accessor = t.data<float>();
            for (int i = 0; i < dst.Length; i++)
                dst[i] = accessor[i];
        }

        public void Save(string directory)
        {
            Directory.CreateDirectory(directory);
            _critic.save(Path.Combine(directory, "ppo_critic.pt"));
            _actorMean.save(Path.Combine(directory, "ppo_actor.pt"));

            using var bw = new BinaryWriter(File.Create(
                Path.Combine(directory, "ppo_state.bin")));
            bw.Write(_updateCount);
            var logStdData = _actorLogStd.data<float>();
            bw.Write(ActDim);
            for (int i = 0; i < ActDim; i++)
                bw.Write(logStdData[i]);
        }

        public void Load(string directory)
        {
            string criticPath = Path.Combine(directory, "ppo_critic.pt");
            string actorPath = Path.Combine(directory, "ppo_actor.pt");
            if (File.Exists(criticPath)) _critic.load(criticPath);
            if (File.Exists(actorPath)) _actorMean.load(actorPath);

            string statePath = Path.Combine(directory, "ppo_state.bin");
            if (File.Exists(statePath))
            {
                using var br = new BinaryReader(File.OpenRead(statePath));
                _updateCount = br.ReadInt32();
                int dim = br.ReadInt32();
                if (dim == ActDim)
                {
                    using (no_grad())
                    {
                        for (int i = 0; i < dim; i++)
                            _actorLogStd[0, i] = br.ReadSingle();
                    }
                }
            }

            _optimizer?.Dispose();
            _optimizer = optim.Adam(AllParameters(), lr: 3e-4f, eps: 1e-5);
            SyncInferenceWeights();
        }

        public void Dispose()
        {
            _critic?.Dispose();
            _actorMean?.Dispose();
            _actorLogStd?.Dispose();
            _optimizer?.Dispose();
            _infObsTensor?.Dispose();
            _infActorA?.Dispose();
            _infActorB?.Dispose();
            _infCriticA?.Dispose();
            _infCriticB?.Dispose();
            _infLogStdA?.Dispose();
            _infLogStdB?.Dispose();
        }
    }
}
