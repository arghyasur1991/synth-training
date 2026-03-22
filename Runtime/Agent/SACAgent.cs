using System;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Tensor = TorchSharp.torch.Tensor;

namespace Genesis.Sentience.Learning
{
    public class SACActorNetwork : Module<Tensor, (Tensor action, Tensor logProb, Tensor mean)>
    {
        private const float LOG_STD_MIN = -5f;
        private const float LOG_STD_MAX = 2f;

        // V1 flat-MLP layers (used when _latentDim == 0)
        private readonly Linear fc1, fc2;

        // Encoder bottleneck layers (used when _latentDim > 0)
        private readonly Linear encFc1, encFc2, policyFc1, policyFc2;

        private readonly Linear fcMean, fcLogStd;
        private readonly float _actionScale;
        private readonly float _actionBias;
        private readonly int _latentDim;

        public int LatentDim => _latentDim;

        public SACActorNetwork(int obsDim, int actDim, int hidden1 = 256, int hidden2 = 256,
            float actionScale = 1f, float actionBias = 0f, int latentDim = 0, int encoderHidden = 128)
            : base("SACActorNetwork")
        {
            _actionScale = actionScale;
            _actionBias = actionBias;
            _latentDim = latentDim;

            if (latentDim > 0)
            {
                encFc1 = Linear(obsDim, encoderHidden);
                encFc2 = Linear(encoderHidden, latentDim);
                policyFc1 = Linear(latentDim, hidden1);
                policyFc2 = Linear(hidden1, hidden2);
            }
            else
            {
                fc1 = Linear(obsDim, hidden1);
                fc2 = Linear(hidden1, hidden2);
            }

            fcMean = Linear(hidden2, actDim);
            fcLogStd = Linear(hidden2, actDim);

            RegisterComponents();
        }

        public override (Tensor action, Tensor logProb, Tensor mean) forward(Tensor obs)
        {
            Tensor x;
            if (_latentDim > 0)
            {
                var h = functional.relu(encFc1.forward(obs));
                var z = encFc2.forward(h);
                x = functional.relu(policyFc1.forward(z));
                x = functional.relu(policyFc2.forward(x));
            }
            else
            {
                x = functional.relu(fc1.forward(obs));
                x = functional.relu(fc2.forward(x));
            }

            var mean = fcMean.forward(x);
            var logStd = fcLogStd.forward(x).clamp(LOG_STD_MIN, LOG_STD_MAX);
            var std = logStd.exp();

            var noise = torch.randn_like(mean);
            var xT = mean + std * noise;
            var action = torch.tanh(xT) * _actionScale + _actionBias;

            var logProbRaw = -0.5f * (((xT - mean) / std).pow(2) + 2f * logStd
                            + (float)Math.Log(2.0 * Math.PI));
            var logProb = logProbRaw.sum(-1);
            logProb = logProb - (2f * ((float)Math.Log(2.0) - xT
                     - functional.softplus(-2f * xT))).sum(1);

            var meanAction = torch.tanh(mean) * _actionScale + _actionBias;

            return (action, logProb, meanAction);
        }
    }

    public class SoftQNetwork : Module<Tensor, Tensor, Tensor>
    {
        private readonly Linear fc1, fc2, fc3;
        private readonly Tensor[] _catBuf = new Tensor[2];

        public SoftQNetwork(int obsDim, int actDim, int hidden1 = 256, int hidden2 = 256)
            : base("SoftQNetwork")
        {
            fc1 = Linear(obsDim + actDim, hidden1);
            fc2 = Linear(hidden1, hidden2);
            fc3 = Linear(hidden2, 1);

            using (no_grad())
            {
                fc3.weight.mul_(0.01f);
                fc3.bias?.fill_(0f);
            }

            RegisterComponents();
        }

        public override Tensor forward(Tensor obs, Tensor action)
        {
            _catBuf[0] = obs;
            _catBuf[1] = action;
            var x = torch.cat(_catBuf, dim: 1);
            x = functional.relu(fc1.forward(x));
            x = functional.relu(fc2.forward(x));
            return fc3.forward(x);
        }
    }

    /// <summary>
    /// SAC Agent with GC-optimized paths:
    ///   - Pre-allocated CPU tensor for inference obs (bytes setter, no per-call wrapper)
    ///   - Cached Tensor[] and Dictionary to avoid hot-path allocations
    ///   - DisposeScope for intermediate tensor cleanup
    ///   - Optional TemporalContextEncoder for history-conditioned policy
    /// </summary>
    public class SACAgent : IDisposable
    {
        public readonly SACActorNetwork Actor;
        public readonly SoftQNetwork QF1, QF2;
        public readonly SoftQNetwork QF1Target, QF2Target;

        private optim.Optimizer _qOptimizer;
        private optim.Optimizer _actorOptimizer;
        private optim.Optimizer _alphaOptimizer;

        private readonly float _qLr;
        private readonly float _policyLr;

        private Parameter _logAlpha;

        public readonly Device Device;
        public readonly int ObsDim;
        public readonly int ActDim;
        private readonly float _actionScale;

        private readonly float _gamma;
        private readonly float _tau;
        private readonly int _policyFrequency;
        private float _targetEntropy;
        private readonly float _qGradClipNorm;
        private readonly float _actorGradClipNorm;
        private readonly float _logAlphaMin;
        private readonly float _maxQValue;
        private readonly float _targetSmoothNoise;
        private readonly float _targetNoiseClip;

        private int _trainStep;
        private float _lastQLoss;
        private float _lastActorLoss;
        private float _lastAlphaLoss;

        // Temporal context encoder (null when ContextDim == 0)
        private readonly TemporalContextEncoder _contextEncoder;
        private TemporalContextEncoder _infContextA;
        private TemporalContextEncoder _infContextB;
        private TemporalContextEncoder _activeInfContext;
        private readonly int _contextDim;
        private readonly int _contextWarmupSteps;
        private readonly int _contextSeqLen;
        private readonly int _contextEntryDim; // obsDim + actDim + 1

        public float Alpha => (float)Math.Exp(_logAlpha.item<float>());
        public float LastQLoss => _lastQLoss;
        public float LastActorLoss => _lastActorLoss;
        public float LastAlphaLoss => _lastAlphaLoss;
        public int TrainSteps => _trainStep;
        public float TargetEntropy => _targetEntropy;
        public bool HasEncoder => Actor.LatentDim > 0;
        public bool HasContext => _contextDim > 0;
        public int ContextDim => _contextDim;

        public void SetTargetEntropy(int activeDims, float entropyScale)
        {
            _targetEntropy = -activeDims * entropyScale;
        }

        private SACActorNetwork _inferenceActorA;
        private SACActorNetwork _inferenceActorB;
        private SACActorNetwork _activeInferenceActor;

        private readonly float[] _actionBuffer;

        // Ornstein-Uhlenbeck noise state for correlated exploration
        private readonly float[] _ouState;
        private readonly float _ouTheta;
        private readonly float _ouSigma;
        private readonly bool _useCorrelatedNoise;
        private float[] _perJointOUSigma;

        // Pre-allocated inference tensor: avoids creating a new C# Tensor wrapper per call.
        // Lives outside any DisposeScope so it persists across calls.
        private Tensor _infObsTensor;
        private readonly int _infObsBytes;

        // Cached dictionary for SyncInferenceWeights — avoids re-allocating per call.
        private readonly System.Collections.Generic.Dictionary<string, Parameter> _syncDict
            = new System.Collections.Generic.Dictionary<string, Parameter>();

        public SACAgent(int obsDim, int actDim, SACConfig config, Device device)
        {
            ObsDim = obsDim;
            ActDim = actDim;
            Device = device;

            _actionScale = config.ActionScale;
            _gamma = config.Gamma;
            _tau = config.Tau;
            _policyFrequency = config.PolicyFrequency;
            _targetEntropy = -actDim * config.TargetEntropyScale;
            _qGradClipNorm = config.QGradClipNorm;
            _actorGradClipNorm = config.ActorGradClipNorm;
            _logAlphaMin = (float)Math.Log(Math.Max(config.AlphaMin, 1e-6));
            _maxQValue = config.MaxQValue;
            _targetSmoothNoise = config.TargetSmoothNoise;
            _targetNoiseClip = config.TargetNoiseClip;
            _qLr = config.QLr;
            _policyLr = config.PolicyLr;

            _contextDim = config.ContextDim;
            _contextWarmupSteps = config.ContextWarmupSteps;
            _contextSeqLen = config.ContextSeqLen;
            _contextEntryDim = obsDim + actDim + 1;

            // When context is enabled, all networks see augmented observations
            int netObsDim = _contextDim > 0 ? obsDim + _contextDim : obsDim;

            Actor = new SACActorNetwork(netObsDim, actDim, config.Hidden1, config.Hidden2,
                actionScale: config.ActionScale, latentDim: config.LatentDim,
                encoderHidden: config.EncoderHidden);
            QF1 = new SoftQNetwork(netObsDim, actDim, config.Hidden1, config.Hidden2);
            QF2 = new SoftQNetwork(netObsDim, actDim, config.Hidden1, config.Hidden2);
            QF1Target = new SoftQNetwork(netObsDim, actDim, config.Hidden1, config.Hidden2);
            QF2Target = new SoftQNetwork(netObsDim, actDim, config.Hidden1, config.Hidden2);

            Actor.to(device);
            QF1.to(device);
            QF2.to(device);
            QF1Target.to(device);
            QF2Target.to(device);

            CopyWeights(QF1, QF1Target);
            CopyWeights(QF2, QF2Target);

            // Context encoder (when enabled)
            if (_contextDim > 0)
            {
                _contextEncoder = new TemporalContextEncoder(
                    _contextEntryDim, _contextDim, config.ContextSeqLen,
                    config.ContextChannelSize);
                _contextEncoder.to(device);
            }

            float initAlpha = Math.Max(config.AlphaInit, config.AlphaMin);
            _logAlpha = new Parameter(torch.tensor((float)Math.Log(initAlpha), device: device).unsqueeze(0));

            _qOptimizer = optim.Adam(ConcatParams(QF1, QF2), lr: config.QLr);
            _actorOptimizer = _contextEncoder != null
                ? optim.Adam(ConcatParams(Actor, _contextEncoder), lr: config.PolicyLr)
                : optim.Adam(Actor.parameters(), lr: config.PolicyLr);
            _alphaOptimizer = optim.Adam(new[] { _logAlpha }, lr: config.PolicyLr);

            _inferenceActorA = new SACActorNetwork(netObsDim, actDim, config.Hidden1, config.Hidden2,
                actionScale: config.ActionScale, latentDim: config.LatentDim,
                encoderHidden: config.EncoderHidden);
            _inferenceActorB = new SACActorNetwork(netObsDim, actDim, config.Hidden1, config.Hidden2,
                actionScale: config.ActionScale, latentDim: config.LatentDim,
                encoderHidden: config.EncoderHidden);
            _inferenceActorA.to(torch.CPU);
            _inferenceActorB.to(torch.CPU);
            _activeInferenceActor = _inferenceActorA;

            if (_contextDim > 0)
            {
                _infContextA = new TemporalContextEncoder(
                    _contextEntryDim, _contextDim, config.ContextSeqLen,
                    config.ContextChannelSize);
                _infContextB = new TemporalContextEncoder(
                    _contextEntryDim, _contextDim, config.ContextSeqLen,
                    config.ContextChannelSize);
                _infContextA.to(torch.CPU);
                _infContextB.to(torch.CPU);
                _activeInfContext = _infContextA;
            }

            SyncInferenceWeights();

            _actionBuffer = new float[actDim];
            // Inference tensor size: netObsDim (obs + context when enabled)
            _infObsTensor = torch.zeros(1, netObsDim, dtype: ScalarType.Float32);
            _infObsBytes = obsDim * sizeof(float);

            _useCorrelatedNoise = config.CorrelatedNoise;
            _ouTheta = config.OUTheta;
            _ouSigma = config.OUSigma;
            _ouState = new float[actDim];
        }

        /// <summary>
        /// Set per-joint OU sigma array for differentiated exploration.
        /// Array length must match ActDim. When set, GetRandomAction uses
        /// per-element sigma instead of the uniform _ouSigma.
        /// </summary>
        public void SetPerJointOUSigma(float[] sigmaArray)
        {
            if (sigmaArray != null && sigmaArray.Length == ActDim)
                _perJointOUSigma = sigmaArray;
        }

        private static System.Collections.Generic.IEnumerable<Parameter> ConcatParams(
            Module m1, Module m2)
        {
            foreach (var p in m1.parameters()) yield return p;
            foreach (var p in m2.parameters()) yield return p;
        }

        /// <summary>
        /// Get action from inference actor. Lock-free actor read via volatile reference.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float[] GetAction(float[] obs)
        {
            using var scope = NewDisposeScope();
            using (no_grad())
            {
                _infObsTensor.bytes = MemoryMarshal.AsBytes<float>(obs.AsSpan());
                var actor = Volatile.Read(ref _activeInferenceActor);
                var (action, _, _) = actor.forward(_infObsTensor);
                CopyTensorToBuffer(action, _actionBuffer);
            }

            return _actionBuffer;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float[] GetDeterministicAction(float[] obs)
        {
            using var scope = NewDisposeScope();
            using (no_grad())
            {
                _infObsTensor.bytes = MemoryMarshal.AsBytes<float>(obs.AsSpan());
                var actor = Volatile.Read(ref _activeInferenceActor);
                var (_, _, meanAction) = actor.forward(_infObsTensor);
                CopyTensorToBuffer(meanAction, _actionBuffer);
            }

            return _actionBuffer;
        }

        /// <summary>
        /// Get action with temporal context. historySeq: (seqLen, 1, entryDim),
        /// historyMask: (1, seqLen) float tensor (1=padded).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float[] GetActionWithContext(float[] obs, Tensor historySeq, Tensor historyMask)
        {
            if (_contextDim <= 0) return GetAction(obs);

            using var scope = NewDisposeScope();
            using (no_grad())
            {
                var ctxEnc = Volatile.Read(ref _activeInfContext);
                var boolMask = historyMask.to_type(ScalarType.Bool);
                var context = ctxEnc.forward(historySeq, boolMask);

                float gate = _contextWarmupSteps > 0
                    ? Math.Min(1f, (float)_trainStep / _contextWarmupSteps)
                    : 1f;
                if (gate < 1f)
                    context = context * gate;

                // Build augmented obs: [obs | context]
                var obsTensor = torch.zeros(1, ObsDim, dtype: ScalarType.Float32);
                obsTensor.bytes = MemoryMarshal.AsBytes<float>(obs.AsSpan());
                var augObs = torch.cat(new[] { obsTensor, context }, dim: 1);

                var actor = Volatile.Read(ref _activeInferenceActor);
                var (action, _, _) = actor.forward(augObs);
                CopyTensorToBuffer(action, _actionBuffer);
            }

            return _actionBuffer;
        }

        /// <summary>
        /// Get deterministic action with temporal context.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float[] GetDeterministicActionWithContext(float[] obs, Tensor historySeq, Tensor historyMask)
        {
            if (_contextDim <= 0) return GetDeterministicAction(obs);

            using var scope = NewDisposeScope();
            using (no_grad())
            {
                var ctxEnc = Volatile.Read(ref _activeInfContext);
                var boolMask = historyMask.to_type(ScalarType.Bool);
                var context = ctxEnc.forward(historySeq, boolMask);

                float gate = _contextWarmupSteps > 0
                    ? Math.Min(1f, (float)_trainStep / _contextWarmupSteps)
                    : 1f;
                if (gate < 1f)
                    context = context * gate;

                var obsTensor = torch.zeros(1, ObsDim, dtype: ScalarType.Float32);
                obsTensor.bytes = MemoryMarshal.AsBytes<float>(obs.AsSpan());
                var augObs = torch.cat(new[] { obsTensor, context }, dim: 1);

                var actor = Volatile.Read(ref _activeInferenceActor);
                var (_, _, meanAction) = actor.forward(augObs);
                CopyTensorToBuffer(meanAction, _actionBuffer);
            }

            return _actionBuffer;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float[] GetRandomAction(Random rng)
        {
            if (_useCorrelatedNoise)
            {
                bool perJoint = _perJointOUSigma != null;
                for (int i = 0; i < ActDim; i++)
                {
                    float noise = (float)(rng.NextDouble() * 2.0 - 1.0) * 1.732f;
                    float sigma = perJoint ? _perJointOUSigma[i] : _ouSigma;
                    _ouState[i] += _ouTheta * (0f - _ouState[i]) + sigma * noise;
                    float clamped = _ouState[i] > 1f ? 1f : (_ouState[i] < -1f ? -1f : _ouState[i]);
                    _actionBuffer[i] = clamped * _actionScale;
                }
            }
            else
            {
                for (int i = 0; i < ActDim; i++)
                    _actionBuffer[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * _actionScale;
            }
            return _actionBuffer;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CopyTensorToBuffer(Tensor t, float[] dst)
        {
            var accessor = t.data<float>();
            for (int i = 0; i < dst.Length; i++)
                dst[i] = accessor[i];
        }

        /// <summary>
        /// One SAC training step with PER support.
        /// Uses importance-sampling weights for unbiased Q-loss,
        /// and writes per-sample |TD-error| into batch.TDErrors for priority update.
        /// </summary>
        public void TrainStep(Batch batch)
        {
            using var scope = NewDisposeScope();

            var obs = torch.tensor(batch.Obs).reshape(batch.Size, batch.ObsDim).to(Device);
            var actions = torch.tensor(batch.Actions).reshape(batch.Size, batch.ActDim).to(Device);
            var rewards = torch.tensor(batch.Rewards).reshape(batch.Size, 1).to(Device);
            var nextObs = torch.tensor(batch.NextObs).reshape(batch.Size, batch.ObsDim).to(Device);
            var dones = torch.tensor(batch.Dones).reshape(batch.Size, 1).to(Device);
            var isWeights = torch.tensor(batch.ISWeights).reshape(batch.Size, 1).to(Device);

            float alpha = Alpha;

            // --- Q-network update with IS-weighted loss ---
            // Reward-only Bellman target: entropy regularization is applied only in the
            // actor loss, NOT in the Q-target. With 225-dim actions, including entropy in
            // the Q-target inflates Q-values to ~40K+ (entropy bonus ~400/step vs reward
            // ~0.3/step), making the reward signal invisible to the Q-networks.
            {
                Tensor nextQValue;
                using (no_grad())
                {
                    var (nextAction, _, _) = Actor.forward(nextObs);

                    // TD3-style target policy smoothing: add clipped noise to prevent
                    // the actor from exploiting narrow Q-network peaks in 225-dim space.
                    if (_targetSmoothNoise > 0f)
                    {
                        var noise = (torch.randn_like(nextAction) * _targetSmoothNoise)
                            .clamp(-_targetNoiseClip, _targetNoiseClip);
                        nextAction = (nextAction + noise).clamp(-_actionScale, _actionScale);
                    }

                    var qf1NextTarget = QF1Target.forward(nextObs, nextAction);
                    var qf2NextTarget = QF2Target.forward(nextObs, nextAction);
                    var minQNext = torch.min(qf1NextTarget, qf2NextTarget);
                    nextQValue = (rewards + _gamma * (1f - dones) * minQNext).clamp(-_maxQValue, _maxQValue);
                }

                var qf1Val = QF1.forward(obs, actions);
                var qf2Val = QF2.forward(obs, actions);

                var td1 = qf1Val - nextQValue;
                var td2 = qf2Val - nextQValue;

                using (no_grad())
                {
                    var absTD = torch.max(td1.detach().abs(), td2.detach().abs()).view(-1).cpu();
                    var accessor = absTD.data<float>();
                    for (int i = 0; i < batch.Size; i++)
                        batch.TDErrors[i] = accessor[i];
                }

                var loss = (isWeights * (td1.pow(2) + td2.pow(2))).mean();

                _qOptimizer.zero_grad();
                loss.backward();
                if (_qGradClipNorm > 0f)
                {
                    torch.nn.utils.clip_grad_norm_(ConcatParams(QF1, QF2), _qGradClipNorm);
                }
                _qOptimizer.step();
                _lastQLoss = loss.item<float>();
            }

            // --- Actor and alpha update (every policyFrequency steps) ---
            _trainStep++;
            if (_trainStep % _policyFrequency == 0)
            {
                var (piAction, logPi, _) = Actor.forward(obs);
                var qf1Pi = QF1.forward(obs, piAction);
                var qf2Pi = QF2.forward(obs, piAction);
                var minQPi = torch.min(qf1Pi, qf2Pi);

                var actorLoss = (alpha * logPi.unsqueeze(1) - minQPi).mean();
                _actorOptimizer.zero_grad();
                actorLoss.backward();
                if (_actorGradClipNorm > 0f)
                {
                    torch.nn.utils.clip_grad_norm_(Actor.parameters(), _actorGradClipNorm);
                }
                _actorOptimizer.step();
                _lastActorLoss = actorLoss.item<float>();

                var alphaLoss = (-_logAlpha.exp() * (logPi.detach() + _targetEntropy)).mean();
                _alphaOptimizer.zero_grad();
                alphaLoss.backward();
                _alphaOptimizer.step();
                _lastAlphaLoss = alphaLoss.item<float>();

                using (no_grad())
                {
                    if (_logAlpha.item<float>() < _logAlphaMin)
                        _logAlpha.fill_(_logAlphaMin);
                }
            }

            // --- Target network Polyak update ---
            PolyakUpdate(QF1, QF1Target, _tau);
            PolyakUpdate(QF2, QF2Target, _tau);
        }

        /// <summary>
        /// Train only the Q-networks on an imagined (dream) batch.
        /// Skips actor and alpha updates — the policy should only be shaped
        /// by real-world experience to avoid compounding model errors.
        /// Dream batches should have ISWeights set to 1.0 (uniform).
        /// </summary>
        public void TrainCriticOnly(Batch batch)
        {
            using var scope = NewDisposeScope();

            var obs = torch.tensor(batch.Obs).reshape(batch.Size, batch.ObsDim).to(Device);
            var actions = torch.tensor(batch.Actions).reshape(batch.Size, batch.ActDim).to(Device);
            var rewards = torch.tensor(batch.Rewards).reshape(batch.Size, 1).to(Device);
            var nextObs = torch.tensor(batch.NextObs).reshape(batch.Size, batch.ObsDim).to(Device);
            var dones = torch.tensor(batch.Dones).reshape(batch.Size, 1).to(Device);
            var isWeights = torch.tensor(batch.ISWeights).reshape(batch.Size, 1).to(Device);

            Tensor nextQValue;
            using (no_grad())
            {
                var (nextAction, _, _) = Actor.forward(nextObs);

                if (_targetSmoothNoise > 0f)
                {
                    var noise = (torch.randn_like(nextAction) * _targetSmoothNoise)
                        .clamp(-_targetNoiseClip, _targetNoiseClip);
                    nextAction = (nextAction + noise).clamp(-_actionScale, _actionScale);
                }

                var qf1NextTarget = QF1Target.forward(nextObs, nextAction);
                var qf2NextTarget = QF2Target.forward(nextObs, nextAction);
                var minQNext = torch.min(qf1NextTarget, qf2NextTarget);
                nextQValue = (rewards + _gamma * (1f - dones) * minQNext)
                    .clamp(-_maxQValue, _maxQValue);
            }

            var qf1Val = QF1.forward(obs, actions);
            var qf2Val = QF2.forward(obs, actions);

            var td1 = qf1Val - nextQValue;
            var td2 = qf2Val - nextQValue;
            var loss = (isWeights * (td1.pow(2) + td2.pow(2))).mean();

            _qOptimizer.zero_grad();
            loss.backward();
            if (_qGradClipNorm > 0f)
                torch.nn.utils.clip_grad_norm_(ConcatParams(QF1, QF2), _qGradClipNorm);
            _qOptimizer.step();

            PolyakUpdate(QF1, QF1Target, _tau);
            PolyakUpdate(QF2, QF2Target, _tau);
        }

        /// <summary>
        /// SAC training step with temporal context from history sequences.
        /// The context encoder is trained end-to-end via the actor loss.
        /// </summary>
        public void TrainStep(SequenceBatch batch)
        {
            if (_contextEncoder == null)
            {
                // Fallback: create a regular Batch view and use the non-context path.
                // This shouldn't normally happen (trainer selects the right path).
                var simpleBatch = new Batch(batch.Size, batch.ObsDim, batch.ActDim);
                Buffer.BlockCopy(batch.Obs, 0, simpleBatch.Obs, 0, batch.Obs.Length * sizeof(float));
                Buffer.BlockCopy(batch.Actions, 0, simpleBatch.Actions, 0, batch.Actions.Length * sizeof(float));
                Buffer.BlockCopy(batch.Rewards, 0, simpleBatch.Rewards, 0, batch.Rewards.Length * sizeof(float));
                Buffer.BlockCopy(batch.NextObs, 0, simpleBatch.NextObs, 0, batch.NextObs.Length * sizeof(float));
                Buffer.BlockCopy(batch.Dones, 0, simpleBatch.Dones, 0, batch.Dones.Length * sizeof(float));
                Array.Copy(batch.Indices, simpleBatch.Indices, batch.Size);
                Buffer.BlockCopy(batch.ISWeights, 0, simpleBatch.ISWeights, 0, batch.Size * sizeof(float));
                TrainStep(simpleBatch);
                Array.Copy(simpleBatch.TDErrors, batch.TDErrors, batch.Size);
                return;
            }

            using var scope = NewDisposeScope();

            var rawObs = torch.tensor(batch.Obs).reshape(batch.Size, batch.ObsDim).to(Device);
            var actions = torch.tensor(batch.Actions).reshape(batch.Size, batch.ActDim).to(Device);
            var rewards = torch.tensor(batch.Rewards).reshape(batch.Size, 1).to(Device);
            var rawNextObs = torch.tensor(batch.NextObs).reshape(batch.Size, batch.ObsDim).to(Device);
            var dones = torch.tensor(batch.Dones).reshape(batch.Size, 1).to(Device);
            var isWeights = torch.tensor(batch.ISWeights).reshape(batch.Size, 1).to(Device);

            // Build history tensor: reshape from flat (B*S*E) to (S, B, E) for transformer
            var histFlat = torch.tensor(batch.HistoryData)
                .reshape(batch.Size, batch.SeqLen, batch.EntryDim).to(Device);
            var histSeq = histFlat.permute(1, 0, 2).contiguous(); // (seqLen, batch, entryDim)
            var histMask = torch.tensor(batch.HistoryMask)
                .reshape(batch.Size, batch.SeqLen).to_type(ScalarType.Bool).to(Device);

            float gate = _contextWarmupSteps > 0
                ? Math.Min(1f, (float)_trainStep / _contextWarmupSteps)
                : 1f;

            // Compute context WITHOUT gradient for critic update
            Tensor contextDetached;
            using (no_grad())
            {
                var ctx = _contextEncoder.forward(histSeq, histMask);
                if (gate < 1f) ctx = ctx * gate;
                contextDetached = ctx.detach();
            }

            var obs = torch.cat(new[] { rawObs, contextDetached }, dim: 1);
            var nextObs = torch.cat(new[] { rawNextObs, contextDetached }, dim: 1);

            float alpha = Alpha;

            // --- Q-network update (context detached, no gradient through transformer) ---
            {
                Tensor nextQValue;
                using (no_grad())
                {
                    var (nextAction, _, _) = Actor.forward(nextObs);
                    if (_targetSmoothNoise > 0f)
                    {
                        var noise = (torch.randn_like(nextAction) * _targetSmoothNoise)
                            .clamp(-_targetNoiseClip, _targetNoiseClip);
                        nextAction = (nextAction + noise).clamp(-_actionScale, _actionScale);
                    }
                    var qf1NextTarget = QF1Target.forward(nextObs, nextAction);
                    var qf2NextTarget = QF2Target.forward(nextObs, nextAction);
                    var minQNext = torch.min(qf1NextTarget, qf2NextTarget);
                    nextQValue = (rewards + _gamma * (1f - dones) * minQNext).clamp(-_maxQValue, _maxQValue);
                }

                var qf1Val = QF1.forward(obs, actions);
                var qf2Val = QF2.forward(obs, actions);
                var td1 = qf1Val - nextQValue;
                var td2 = qf2Val - nextQValue;

                using (no_grad())
                {
                    var absTD = torch.max(td1.detach().abs(), td2.detach().abs()).view(-1).cpu();
                    var accessor = absTD.data<float>();
                    for (int i = 0; i < batch.Size; i++)
                        batch.TDErrors[i] = accessor[i];
                }

                var loss = (isWeights * (td1.pow(2) + td2.pow(2))).mean();
                _qOptimizer.zero_grad();
                loss.backward();
                if (_qGradClipNorm > 0f)
                    torch.nn.utils.clip_grad_norm_(ConcatParams(QF1, QF2), _qGradClipNorm);
                _qOptimizer.step();
                _lastQLoss = loss.item<float>();
            }

            // --- Actor and alpha update (context WITH gradient for transformer training) ---
            _trainStep++;
            if (_trainStep % _policyFrequency == 0)
            {
                var contextWithGrad = _contextEncoder.forward(histSeq, histMask);
                if (gate < 1f) contextWithGrad = contextWithGrad * gate;
                var augObsActor = torch.cat(new[] { rawObs, contextWithGrad }, dim: 1);

                var (piAction, logPi, _) = Actor.forward(augObsActor);
                var qf1Pi = QF1.forward(augObsActor, piAction);
                var qf2Pi = QF2.forward(augObsActor, piAction);
                var minQPi = torch.min(qf1Pi, qf2Pi);

                var actorLoss = (alpha * logPi.unsqueeze(1) - minQPi).mean();
                _actorOptimizer.zero_grad();
                actorLoss.backward();
                if (_actorGradClipNorm > 0f)
                    torch.nn.utils.clip_grad_norm_(ConcatParams(Actor, _contextEncoder), _actorGradClipNorm);
                _actorOptimizer.step();
                _lastActorLoss = actorLoss.item<float>();

                var alphaLoss = (-_logAlpha.exp() * (logPi.detach() + _targetEntropy)).mean();
                _alphaOptimizer.zero_grad();
                alphaLoss.backward();
                _alphaOptimizer.step();
                _lastAlphaLoss = alphaLoss.item<float>();

                using (no_grad())
                {
                    if (_logAlpha.item<float>() < _logAlphaMin)
                        _logAlpha.fill_(_logAlphaMin);
                }
            }

            PolyakUpdate(QF1, QF1Target, _tau);
            PolyakUpdate(QF2, QF2Target, _tau);
        }

        /// <summary>
        /// Copy training actor weights into the inactive inference actor,
        /// then atomically swap. GetAction() reads the active reference
        /// without any lock, so inference is never blocked.
        /// </summary>
        public void SyncInferenceWeights()
        {
            var active = _activeInferenceActor;
            var staging = (active == _inferenceActorA) ? _inferenceActorB : _inferenceActorA;

            using var scope = NewDisposeScope();

            _syncDict.Clear();
            foreach (var (name, param) in staging.named_parameters())
                _syncDict[name] = param;

            using (no_grad())
            {
                foreach (var (name, param) in Actor.named_parameters())
                {
                    if (_syncDict.TryGetValue(name, out var dst))
                        dst.copy_(param.cpu());
                }
            }

            Interlocked.Exchange(ref _activeInferenceActor, staging);

            // Sync context encoder inference copies
            if (_contextEncoder != null && _infContextA != null)
            {
                var activeCtx = _activeInfContext;
                var stagingCtx = (activeCtx == _infContextA) ? _infContextB : _infContextA;

                _syncDict.Clear();
                foreach (var (name, param) in stagingCtx.named_parameters())
                    _syncDict[name] = param;

                using (no_grad())
                {
                    foreach (var (name, param) in _contextEncoder.named_parameters())
                    {
                        if (_syncDict.TryGetValue(name, out var dst))
                            dst.copy_(param.cpu());
                    }
                }

                Interlocked.Exchange(ref _activeInfContext, stagingCtx);
            }
        }

        private static void CopyWeights(Module src, Module dst)
        {
            using var scope = NewDisposeScope();
            using (no_grad())
            {
                var srcParams = src.named_parameters();
                var dstDict = new System.Collections.Generic.Dictionary<string, Parameter>();
                foreach (var (name, p) in dst.named_parameters())
                    dstDict[name] = p;

                foreach (var (name, p) in srcParams)
                {
                    if (dstDict.TryGetValue(name, out var d))
                        d.copy_(p);
                }
            }
        }

        private static void PolyakUpdate(Module source, Module target, float tau)
        {
            using var scope = NewDisposeScope();
            using (no_grad())
            {
                var srcEnum = source.parameters().GetEnumerator();
                var tgtEnum = target.parameters().GetEnumerator();
                while (srcEnum.MoveNext() && tgtEnum.MoveNext())
                {
                    tgtEnum.Current.mul_(1f - tau).add_(srcEnum.Current, alpha: tau);
                }
            }
        }

        public void Save(string directory)
        {
            Directory.CreateDirectory(directory);
            Actor.save(Path.Combine(directory, "actor.pt"));
            QF1.save(Path.Combine(directory, "qf1.pt"));
            QF2.save(Path.Combine(directory, "qf2.pt"));
            QF1Target.save(Path.Combine(directory, "qf1_target.pt"));
            QF2Target.save(Path.Combine(directory, "qf2_target.pt"));

            _contextEncoder?.save(Path.Combine(directory, "context_encoder.pt"));

            using var bw = new BinaryWriter(File.Create(Path.Combine(directory, "sac_state.bin")));
            bw.Write(_logAlpha.item<float>());
            bw.Write(_trainStep);
        }

        public void Load(string directory)
        {
            Actor.load(Path.Combine(directory, "actor.pt"));
            QF1.load(Path.Combine(directory, "qf1.pt"));
            QF2.load(Path.Combine(directory, "qf2.pt"));
            QF1Target.load(Path.Combine(directory, "qf1_target.pt"));
            QF2Target.load(Path.Combine(directory, "qf2_target.pt"));

            if (_contextEncoder != null)
            {
                var ctxPath = Path.Combine(directory, "context_encoder.pt");
                if (File.Exists(ctxPath))
                    _contextEncoder.load(ctxPath);
            }

            var path = Path.Combine(directory, "sac_state.bin");
            if (File.Exists(path))
            {
                using var br = new BinaryReader(File.OpenRead(path));
                float logAlpha = br.ReadSingle();
                _trainStep = br.ReadInt32();
                using (no_grad())
                    _logAlpha.fill_(logAlpha);
            }

            RebuildOptimizers();
            SyncInferenceWeights();
        }

        private void RebuildOptimizers()
        {
            _qOptimizer?.Dispose();
            _actorOptimizer?.Dispose();
            _alphaOptimizer?.Dispose();

            _qOptimizer = optim.Adam(ConcatParams(QF1, QF2), lr: _qLr);
            _actorOptimizer = _contextEncoder != null
                ? optim.Adam(ConcatParams(Actor, _contextEncoder), lr: _policyLr)
                : optim.Adam(Actor.parameters(), lr: _policyLr);
            _alphaOptimizer = optim.Adam(new[] { _logAlpha }, lr: _policyLr);
        }

        public void Dispose()
        {
            Actor?.Dispose();
            QF1?.Dispose();
            QF2?.Dispose();
            QF1Target?.Dispose();
            QF2Target?.Dispose();
            _inferenceActorA?.Dispose();
            _inferenceActorB?.Dispose();
            _contextEncoder?.Dispose();
            _infContextA?.Dispose();
            _infContextB?.Dispose();
            _logAlpha?.Dispose();
            _qOptimizer?.Dispose();
            _actorOptimizer?.Dispose();
            _alphaOptimizer?.Dispose();
            _infObsTensor?.Dispose();
        }
    }

    [Serializable]
    public class SACConfig
    {
        public int BufferSize = 200_000;
        public float Gamma = 0.99f;
        public float Tau = 0.005f;
        public int BatchSize = 512;
        public float PolicyLr = 3e-4f;
        public float QLr = 3e-4f;
        public int PolicyFrequency = 2;
        public int Hidden1 = 1024;
        public int Hidden2 = 512;
        public int WeightSyncFrequency = 100;

        [UnityEngine.Tooltip("Target entropy = -actDim * scale. Higher values encourage more exploration, " +
            "critical for high-dimensional action spaces. 0.3 recommended for ~90 DOF bodies.")]
        [UnityEngine.Range(0.01f, 1f)]
        public float TargetEntropyScale = 0.3f;

        [UnityEngine.Tooltip("Hard ceiling on Q-target magnitude. Safety net only — " +
            "target smoothing is the primary overestimation defense. " +
            "Set to ~5x expected max Q (maxReward * rewardScale / (1 - Gamma)).")]
        public float MaxQValue = 200f;

        [UnityEngine.Header("Target Policy Smoothing (TD3)")]

        [UnityEngine.Tooltip("Std of noise added to target actions. Smooths Q-targets over a " +
            "neighborhood in action space, preventing the actor from exploiting narrow Q-peaks. " +
            "0 = disabled. Typical: 0.2 * ActionScale.")]
        public float TargetSmoothNoise = 0.1f;

        [UnityEngine.Tooltip("Max magnitude of target smoothing noise per action dim.")]
        public float TargetNoiseClip = 0.2f;

        [UnityEngine.Tooltip("Max gradient norm for Q-networks. Prevents Q-loss explosion. 0 = disabled.")]
        public float QGradClipNorm = 1.0f;

        [UnityEngine.Tooltip("Max gradient norm for actor network. 0 = disabled.")]
        public float ActorGradClipNorm = 1.0f;

        [UnityEngine.Tooltip("Actor output scale. tanh output is multiplied by this. " +
            "Match to the actuator ctrl range so the full tanh range is useful.")]
        public float ActionScale = 0.4f;

        [UnityEngine.Tooltip("Initial alpha (entropy temperature). Higher values encourage more " +
            "exploration early on. Auto-tuning adjusts it toward target entropy.")]
        [UnityEngine.Range(0.01f, 2.0f)]
        public float AlphaInit = 1.0f;

        [UnityEngine.Tooltip("Minimum alpha floor. With 225 dims, even alpha=0.01 " +
            "gives substantial exploration (225 independent noise sources).")]
        [UnityEngine.Range(0.001f, 1.0f)]
        public float AlphaMin = 0.10f;

        [UnityEngine.Header("Exploration")]

        [UnityEngine.Tooltip("Use temporally-correlated (Ornstein-Uhlenbeck) noise instead of " +
            "independent Gaussian. Produces smoother, more physically meaningful torque patterns " +
            "in high-dimensional action spaces.")]
        public bool CorrelatedNoise = true;

        [UnityEngine.Tooltip("OU noise mean reversion rate (theta). Higher = faster reversion to zero.")]
        [UnityEngine.Range(0.01f, 1f)]
        public float OUTheta = 0.15f;

        [UnityEngine.Tooltip("OU noise diffusion scale (sigma). Higher = more exploration variance.")]
        [UnityEngine.Range(0.01f, 1f)]
        public float OUSigma = 0.3f;

        [UnityEngine.Tooltip("Enable per-joint OU sigma based on actuator body-part names. " +
            "When enabled, different joints use different exploration scales.")]
        public bool PerJointOUSigmaEnabled = false;

        [UnityEngine.Tooltip("OU sigma for hip actuators.")]
        public float OUSigmaHip = 0.3f;
        [UnityEngine.Tooltip("OU sigma for knee actuators.")]
        public float OUSigmaKnee = 0.4f;
        [UnityEngine.Tooltip("OU sigma for ankle actuators.")]
        public float OUSigmaAnkle = 0.15f;
        [UnityEngine.Tooltip("OU sigma for shoulder actuators.")]
        public float OUSigmaShoulder = 0.2f;
        [UnityEngine.Tooltip("OU sigma for elbow actuators.")]
        public float OUSigmaElbow = 0.2f;
        [UnityEngine.Tooltip("OU sigma for waist/torso actuators.")]
        public float OUSigmaWaist = 0.1f;
        [UnityEngine.Tooltip("OU sigma for actuators that don't match any keyword.")]
        public float OUSigmaDefault = 0.3f;

        [UnityEngine.Header("Prioritized Experience Replay")]

        [UnityEngine.Tooltip("PER priority exponent. 0 = uniform sampling, 1 = full prioritization. " +
            "Higher values make high-TD-error transitions much more likely to be sampled.")]
        [UnityEngine.Range(0f, 1f)]
        public float PERAlpha = 0.6f;

        [UnityEngine.Tooltip("PER importance-sampling correction (initial). Anneals to 1.0 over training. " +
            "Lower values speed up early learning at the cost of some bias.")]
        [UnityEngine.Range(0f, 1f)]
        public float PERBetaStart = 0.4f;

        [UnityEngine.Tooltip("Training steps over which PER beta anneals from PERBetaStart to 1.0.")]
        public int PERBetaAnnealSteps = 100_000;

        [UnityEngine.Header("Encoder-in-Actor")]

        [UnityEngine.Tooltip("Latent dimension for encoder bottleneck inside the actor. " +
            "0 = flat MLP (V1 mode). When > 0, the actor compresses obs through an " +
            "encoder before the policy head, and outputs a learned progress signal.")]
        public int LatentDim = 0;

        [UnityEngine.Tooltip("Encoder hidden layer size (only used when LatentDim > 0).")]
        public int EncoderHidden = 128;

        [UnityEngine.Header("Temporal Context")]

        [UnityEngine.Tooltip("Context vector dimension. 0 = disabled (reactive mode). " +
            "When > 0, a Conv1D encoder compresses recent history into a context " +
            "vector prepended to observations for actor and critic.")]
        public int ContextDim = 0;

        [UnityEngine.Tooltip("History window length (number of past transitions). " +
            "16 recommended (0.64s at 25Hz). Conv1D scales linearly with length.")]
        public int ContextSeqLen = 16;

        [UnityEngine.Tooltip("Base channel width for Conv1D encoder. " +
            "Actual channels are 3x, 2x, 1x this value.")]
        public int ContextChannelSize = 10;

        [UnityEngine.Tooltip("Steps before context reaches full strength. " +
            "During warmup, context is scaled by min(1, step/warmup).")]
        public int ContextWarmupSteps = 5000;

        [UnityEngine.Header("Drag Force (Assisted Getting-Up)")]

        [UnityEngine.Tooltip("Enable upward drag force on root body to help the agent " +
            "experience standing states early in training.")]
        public bool DragForceEnabled = false;

        [UnityEngine.Tooltip("Base upward force in Newtons during assist phase. " +
            "Actual magnitude drifts via OU process between DragForceMin and DragForceMax.")]
        public float DragForceNewtons = 2000f;

        [UnityEngine.Tooltip("Minimum drag force (Newtons) for OU drift range.")]
        public float DragForceMin = 50f;

        [UnityEngine.Tooltip("Maximum drag force (Newtons) for OU drift range.")]
        public float DragForceMax = 3000f;

        [UnityEngine.Tooltip("OU mean-reversion rate (higher = faster return to DragForceNewtons).")]
        public float DragForceOUTheta = 0.01f;

        [UnityEngine.Tooltip("OU noise scale (higher = more variation).")]
        public float DragForceOUSigma = 30f;

        [UnityEngine.Tooltip("Steps over which drag force ramps from 0 to full strength.")]
        public int DragForceWarmupSteps = 5000;

        [UnityEngine.Tooltip("Force fraction applied to each upper body bone " +
            "(chest, spine, shoulders, head). 1.0 = full base magnitude per bone.")]
        [UnityEngine.Range(0f, 1f)]
        public float DragUpperBodyFraction = 0.5f;

        [UnityEngine.Tooltip("Decision steps of assisted (high force) phase per cycle.")]
        public int DragAssistOnSteps = 2000;

        [UnityEngine.Tooltip("Decision steps of free (minimal force) phase per cycle. " +
            "Agent must learn to maintain posture without assist.")]
        public int DragAssistOffSteps = 5000;

        [UnityEngine.Tooltip("Force used during the free (off) phase. " +
            "Set to 0 for fully unassisted, or a small value for gentle nudge.")]
        public float DragOffForceNewtons = 50f;

        [UnityEngine.Header("Dyna-Style Dreaming")]

        [UnityEngine.Tooltip("Enable world-model dreaming: train a forward model on real transitions, " +
            "periodically generate imagined transitions, and train the critic on them.")]
        public bool DreamEnabled = true;

        [UnityEngine.Tooltip("Real training steps between dream phases.")]
        public int DreamInterval = 1000;

        [UnityEngine.Tooltip("Number of imagined batches generated per dream phase.")]
        public int DreamBatchCount = 4;

        [UnityEngine.Tooltip("Transitions per imagined batch.")]
        public int DreamBatchSize = 256;

        [UnityEngine.Tooltip("World model training steps before dreaming begins. " +
            "Ensures model predictions are reasonable before affecting the critic.")]
        public int DreamWarmupSteps = 5000;

        [UnityEngine.Tooltip("World model learning rate.")]
        public float WorldModelLr = 3e-4f;

        [UnityEngine.Tooltip("World model first hidden layer size.")]
        public int WorldModelHidden1 = 256;

        [UnityEngine.Tooltip("World model second hidden layer size.")]
        public int WorldModelHidden2 = 128;
    }
}
