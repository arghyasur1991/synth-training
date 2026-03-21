using System;
using System.IO;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Tensor = TorchSharp.torch.Tensor;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Learned state encoder: maps raw proprioceptive observations to a compact
    /// latent representation z plus a scalar progress signal.
    ///
    /// Replaces the hand-crafted AgentPhase state machine. The progress signal
    /// is bootstrapped with an auxiliary MSE loss against the heuristic standBlend,
    /// but receives RL gradients through the actor/critic, allowing it to learn
    /// richer representations over time.
    ///
    /// Architecture:
    ///   obs(obsDim) → Linear(64) → ReLU → Linear(zDim) = z
    ///   z → Linear(1) → Sigmoid = progress
    /// </summary>
    public class StateEncoderNetwork : Module<Tensor, (Tensor z, Tensor progress)>
    {
        private readonly Linear fc1, fc2, progressHead;

        public StateEncoderNetwork(int obsDim, int zDim = 32)
            : base("StateEncoderNetwork")
        {
            fc1 = Linear(obsDim, 64);
            fc2 = Linear(64, zDim);
            progressHead = Linear(zDim, 1);

            RegisterComponents();
        }

        public override (Tensor z, Tensor progress) forward(Tensor obs)
        {
            var h = functional.relu(fc1.forward(obs));
            var z = fc2.forward(h);
            var progress = torch.sigmoid(progressHead.forward(z));
            return (z, progress);
        }
    }

    /// <summary>
    /// Manages the state encoder lifecycle: training, inference copies, save/load.
    /// The encoder is trained via two gradient sources:
    ///   1. Auxiliary loss: MSE(progress, standBlend_target)
    ///   2. End-to-end RL: gradients from actor/critic flow through z
    /// </summary>
    public class StateEncoder : IDisposable
    {
        private readonly StateEncoderNetwork _net;
        private readonly optim.Optimizer _optimizer;
        private readonly Device _device;
        private readonly int _zDim;

        private StateEncoderNetwork _inferenceNetA;
        private StateEncoderNetwork _inferenceNetB;
        private volatile StateEncoderNetwork _activeInferenceNet;

        private float _lastAuxLoss;
        private int _trainSteps;

        public int ZDim => _zDim;
        public float LastAuxLoss => _lastAuxLoss;
        public int TrainSteps => _trainSteps;
        public StateEncoderNetwork TrainingNet => _net;

        public StateEncoder(int obsDim, float lr, Device device, int zDim = 32)
        {
            _zDim = zDim;
            _device = device;

            _net = new StateEncoderNetwork(obsDim, zDim);
            _net.to(device);
            _optimizer = optim.Adam(_net.parameters(), lr: lr);

            _inferenceNetA = new StateEncoderNetwork(obsDim, zDim);
            _inferenceNetB = new StateEncoderNetwork(obsDim, zDim);
            _inferenceNetA.to(torch.CPU);
            _inferenceNetB.to(torch.CPU);
            _activeInferenceNet = _inferenceNetA;

            SyncInferenceWeights();
        }

        /// <summary>
        /// Run inference on the main thread (CPU). Returns (z, progress) as float arrays.
        /// </summary>
        public (float[] z, float progress) Infer(Tensor obsTensor)
        {
            using var scope = NewDisposeScope();
            using (no_grad())
            {
                var net = System.Threading.Volatile.Read(ref _activeInferenceNet);
                var (zT, pT) = net.forward(obsTensor);
                var zData = zT.data<float>();
                var zArr = new float[_zDim];
                for (int i = 0; i < _zDim; i++) zArr[i] = zData[i];
                float progress = pT.data<float>()[0];
                return (zArr, progress);
            }
        }

        /// <summary>
        /// Train the auxiliary loss: MSE(progress, standBlend_target).
        /// Called from the training thread alongside SAC updates.
        /// standBlendTargets shape: [batchSize, 1].
        /// Returns the auxiliary loss value.
        /// </summary>
        public float TrainAuxiliary(Tensor obs, Tensor standBlendTargets)
        {
            using var scope = NewDisposeScope();

            var (_, progress) = _net.forward(obs);
            var loss = functional.mse_loss(progress, standBlendTargets);

            _optimizer.zero_grad();
            loss.backward();
            torch.nn.utils.clip_grad_norm_(_net.parameters(), 1.0);
            _optimizer.step();

            _lastAuxLoss = loss.item<float>();
            _trainSteps++;
            return _lastAuxLoss;
        }

        /// <summary>
        /// Forward pass on training device (for computing z that flows into actor/critic).
        /// Gradients are retained for end-to-end RL training.
        /// </summary>
        public (Tensor z, Tensor progress) ForwardTraining(Tensor obs)
        {
            return _net.forward(obs);
        }

        public void SyncInferenceWeights()
        {
            var active = _activeInferenceNet;
            var staging = (active == _inferenceNetA) ? _inferenceNetB : _inferenceNetA;

            using var scope = NewDisposeScope();
            using (no_grad())
            {
                var dstDict = new System.Collections.Generic.Dictionary<string, Parameter>();
                foreach (var (name, param) in staging.named_parameters())
                    dstDict[name] = param;

                foreach (var (name, param) in _net.named_parameters())
                {
                    if (dstDict.TryGetValue(name, out var dst))
                        dst.copy_(param.cpu());
                }
            }
            System.Threading.Interlocked.Exchange(ref _activeInferenceNet, staging);
        }

        public void Save(string directory)
        {
            _net.save(Path.Combine(directory, "state_encoder.pt"));
            var statePath = Path.Combine(directory, "state_encoder_state.bin");
            using var bw = new BinaryWriter(File.Create(statePath));
            bw.Write(_trainSteps);
            bw.Write(_lastAuxLoss);
        }

        public bool Load(string directory)
        {
            var modelPath = Path.Combine(directory, "state_encoder.pt");
            if (!File.Exists(modelPath)) return false;
            try
            {
                _net.load(modelPath);
                var statePath = Path.Combine(directory, "state_encoder_state.bin");
                if (File.Exists(statePath))
                {
                    using var br = new BinaryReader(File.OpenRead(statePath));
                    _trainSteps = br.ReadInt32();
                    _lastAuxLoss = br.ReadSingle();
                }
                SyncInferenceWeights();
                return true;
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogWarning($"StateEncoder: Load failed — {e.Message}");
                return false;
            }
        }

        public void Dispose()
        {
            _net?.Dispose();
            _inferenceNetA?.Dispose();
            _inferenceNetB?.Dispose();
            _optimizer?.Dispose();
        }
    }
}
