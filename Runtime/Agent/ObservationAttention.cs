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
    /// Learned observation attention: produces a per-dimension mask from the
    /// encoder's latent state z. The actor sees obs * attention, focusing on
    /// dimensions relevant to the current body state.
    ///
    /// Expected behavior: when fallen, ankle/knee/hip channels dominate;
    /// when standing/walking, foot contact and arm proprioception increase.
    ///
    /// Architecture: Linear(zDim, 64) → ReLU → Linear(64, obsDim) → Softmax * obsDim
    /// The softmax * obsDim scaling preserves input magnitude (average weight ≈ 1.0).
    /// Trained end-to-end through the SAC actor loss.
    /// </summary>
    public class ObservationAttentionNetwork : Module<Tensor, Tensor>
    {
        private readonly Linear fc1, fc2;
        private readonly int _obsDim;

        public ObservationAttentionNetwork(int zDim, int obsDim)
            : base("ObservationAttentionNetwork")
        {
            _obsDim = obsDim;
            fc1 = Linear(zDim, 64);
            fc2 = Linear(64, obsDim);

            using (no_grad())
            {
                fc2.weight.mul_(0.01f);
                fc2.bias?.fill_(0f);
            }

            RegisterComponents();
        }

        public override Tensor forward(Tensor z)
        {
            var h = functional.relu(fc1.forward(z));
            var logits = fc2.forward(h);
            var weights = functional.softmax(logits, dim: -1) * _obsDim;
            return weights;
        }
    }

    /// <summary>
    /// Manages the attention network: training-device network + CPU inference copy.
    /// </summary>
    public class ObservationAttention : IDisposable
    {
        private readonly ObservationAttentionNetwork _net;
        private readonly int _obsDim;

        private ObservationAttentionNetwork _inferenceNetA;
        private ObservationAttentionNetwork _inferenceNetB;
        private volatile ObservationAttentionNetwork _activeInferenceNet;

        public ObservationAttentionNetwork TrainingNet => _net;

        public ObservationAttention(int zDim, int obsDim, Device device)
        {
            _obsDim = obsDim;
            _net = new ObservationAttentionNetwork(zDim, obsDim);
            _net.to(device);

            _inferenceNetA = new ObservationAttentionNetwork(zDim, obsDim);
            _inferenceNetB = new ObservationAttentionNetwork(zDim, obsDim);
            _inferenceNetA.to(torch.CPU);
            _inferenceNetB.to(torch.CPU);
            _activeInferenceNet = _inferenceNetA;

            SyncInferenceWeights();
        }

        /// <summary>
        /// Apply attention on the main thread (CPU inference).
        /// Takes z tensor and raw obs tensor, returns attended obs tensor.
        /// </summary>
        public Tensor ApplyInference(Tensor z, Tensor obs)
        {
            using var scope = NewDisposeScope();
            using (no_grad())
            {
                var net = System.Threading.Volatile.Read(ref _activeInferenceNet);
                var attnWeights = net.forward(z);
                return (obs * attnWeights).MoveToOuterDisposeScope();
            }
        }

        /// <summary>
        /// Get the current attention weights for diagnostics (CPU inference).
        /// </summary>
        public float[] GetWeights(Tensor z)
        {
            using var scope = NewDisposeScope();
            using (no_grad())
            {
                var net = System.Threading.Volatile.Read(ref _activeInferenceNet);
                var attnWeights = net.forward(z);
                var data = attnWeights.data<float>();
                var result = new float[_obsDim];
                for (int i = 0; i < _obsDim; i++) result[i] = data[i];
                return result;
            }
        }

        /// <summary>
        /// Forward on training device (gradients flow through for actor loss).
        /// </summary>
        public Tensor ForwardTraining(Tensor z)
        {
            return _net.forward(z);
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
            _net.save(Path.Combine(directory, "obs_attention.pt"));
        }

        public bool Load(string directory)
        {
            var path = Path.Combine(directory, "obs_attention.pt");
            if (!File.Exists(path)) return false;
            try
            {
                _net.load(path);
                SyncInferenceWeights();
                return true;
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogWarning($"ObservationAttention: Load failed — {e.Message}");
                return false;
            }
        }

        public void Dispose()
        {
            _net?.Dispose();
            _inferenceNetA?.Dispose();
            _inferenceNetB?.Dispose();
        }
    }
}
