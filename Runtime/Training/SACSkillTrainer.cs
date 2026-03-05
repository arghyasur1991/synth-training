using System;
using System.Threading;
using Debug = UnityEngine.Debug;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// ISkillTrainer wrapping SACAgent + ReplayBuffer with PER.
    /// Background training loop is provided by BaseSkillTrainer.
    /// </summary>
    public class SACSkillTrainer : BaseSkillTrainer
    {
        private SACAgent _agent;
        private ReplayBuffer _buffer;
        private Batch _batch;
        private readonly SACConfig _config;
        private readonly int _learningStarts;

        private float _perBeta;
        private float _perBetaIncrement;

        public SACAgent Agent => _agent;
        public ReplayBuffer Buffer => _buffer;

        public override int ExperienceCount => _buffer?.Count ?? 0;

        public SACSkillTrainer(SACConfig config, int learningStarts, bool isMobile = false)
            : base(isMobile)
        {
            _config = config;
            _learningStarts = learningStarts;
        }

        public override void Initialize(int obsDim, int actDim, TorchSharp.torch.Device device)
        {
            _agent = new SACAgent(obsDim, actDim, _config, device);
            _buffer = new ReplayBuffer(_config.BufferSize, obsDim, actDim,
                perAlpha: _config.PERAlpha);
            _batch = new Batch(_config.BatchSize, obsDim, actDim);

            _perBeta = _config.PERBetaStart;
            _perBetaIncrement = _config.PERBetaAnnealSteps > 0
                ? (1f - _config.PERBetaStart) / _config.PERBetaAnnealSteps
                : 0f;

            Debug.Log($"SACSkillTrainer: Initialized (obs={obsDim}, act={actDim}, " +
                $"device={device}, buffer={_config.BufferSize})");
        }

        public override float[] GetAction(float[] obs) => _agent.GetAction(obs);
        public override float[] GetRandomAction(Random rng) => _agent.GetRandomAction(rng);

        public override void StoreTransition(float[] obs, float[] action, float reward,
                                             float[] nextObs, bool done)
        {
            _buffer.Add(obs, action, reward, nextObs, done ? 1f : 0f);
        }

        protected override bool ReadyToTrain() => _buffer.Count >= _learningStarts;

        protected override void DoTrainStep()
        {
            _buffer.SampleInto(_batch, _perBeta);
            _agent.TrainStep(_batch);
            _buffer.UpdatePriorities(_batch.Indices, _batch.TDErrors, _batch.Size);

            _perBeta = Math.Min(1f, _perBeta + _perBetaIncrement);

            long steps = Interlocked.Read(ref _totalSteps);
            if ((steps + 1) % _config.WeightSyncFrequency == 0)
                _agent.SyncInferenceWeights();
        }

        public override void Save(string directory)
        {
            _agent.Save(directory);
            var path = System.IO.Path.Combine(directory, "replay_buffer.bin");
            using var bw = new System.IO.BinaryWriter(System.IO.File.Create(path));
            _buffer.Save(bw);
        }

        public override bool Load(string directory)
        {
            try
            {
                _agent.Load(directory);
                var path = System.IO.Path.Combine(directory, "replay_buffer.bin");
                if (System.IO.File.Exists(path))
                {
                    using var br = new System.IO.BinaryReader(System.IO.File.OpenRead(path));
                    _buffer.Load(br);
                }
                return true;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"SACSkillTrainer: Load failed — {e.Message}");
                return false;
            }
        }

        public void SetTargetEntropy(int activeDims, float entropyScale)
        {
            _agent?.SetTargetEntropy(activeDims, entropyScale);
        }

        public override void Dispose()
        {
            base.Dispose();
            _agent?.Dispose();
        }
    }
}
