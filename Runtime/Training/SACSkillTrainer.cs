using System;
using System.Threading;
using Debug = UnityEngine.Debug;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// ISkillTrainer wrapping SACAgent + ReplayBuffer with PER.
    /// Background training loop is provided by BaseSkillTrainer.
    /// Optionally includes a WorldModel for Dyna-style dreaming.
    /// </summary>
    public class SACSkillTrainer : BaseSkillTrainer
    {
        private SACAgent _agent;
        private ReplayBuffer _buffer;
        private Batch _batch;
        private SequenceBatch _seqBatch;
        private readonly SACConfig _config;
        private readonly int _learningStarts;
        private bool _useSequences;

        private float _perBeta;
        private float _perBetaIncrement;

        // Dyna dreaming
        private WorldModel _worldModel;
        private Batch _dreamBatch;
        private int _dreamPhaseCount;
        private float _lastWorldModelLoss;

        public SACAgent Agent => _agent;
        public ReplayBuffer Buffer => _buffer;

        public override int ExperienceCount => _buffer?.Count ?? 0;
        public float LastWorldModelLoss => _lastWorldModelLoss;
        public int DreamPhaseCount => _dreamPhaseCount;
        public bool DreamEnabled => _config.DreamEnabled && _worldModel != null;

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

            _useSequences = _config.ContextDim > 0;
            if (_useSequences)
                _seqBatch = new SequenceBatch(_config.BatchSize, obsDim, actDim, _config.ContextSeqLen);
            else
                _batch = new Batch(_config.BatchSize, obsDim, actDim);

            _perBeta = _config.PERBetaStart;
            _perBetaIncrement = _config.PERBetaAnnealSteps > 0
                ? (1f - _config.PERBetaStart) / _config.PERBetaAnnealSteps
                : 0f;

            if (_config.DreamEnabled)
            {
                _worldModel = new WorldModel(obsDim, actDim, _config.WorldModelLr, device,
                    _config.WorldModelHidden1, _config.WorldModelHidden2);
                _dreamBatch = new Batch(_config.DreamBatchSize, obsDim, actDim);
                Debug.Log($"SACSkillTrainer: Dreaming enabled — interval={_config.DreamInterval}, " +
                    $"warmup={_config.DreamWarmupSteps}, batches={_config.DreamBatchCount}x{_config.DreamBatchSize}");
            }

            Debug.Log($"SACSkillTrainer: Initialized (obs={obsDim}, act={actDim}, " +
                $"device={device}, buffer={_config.BufferSize}" +
                (_useSequences ? $", context={_config.ContextDim}, seqLen={_config.ContextSeqLen}" : "") +
                ")");
        }

        public override float[] GetAction(float[] obs) => _agent.GetAction(obs);
        public override float[] GetDeterministicAction(float[] obs) => _agent.GetDeterministicAction(obs);
        public override float[] GetRandomAction(Random rng) => _agent.GetRandomAction(rng);

        public float[] GetActionWithContext(float[] obs, TorchSharp.torch.Tensor histSeq, TorchSharp.torch.Tensor histMask)
            => _agent.GetActionWithContext(obs, histSeq, histMask);

        public float[] GetDeterministicActionWithContext(float[] obs, TorchSharp.torch.Tensor histSeq, TorchSharp.torch.Tensor histMask)
            => _agent.GetDeterministicActionWithContext(obs, histSeq, histMask);

        public override void StoreTransition(float[] obs, float[] action, float reward,
                                             float[] nextObs, bool done)
        {
            _buffer.Add(obs, action, reward, nextObs, done ? 1f : 0f);
        }

        protected override bool ReadyToTrain() => _buffer.Count >= _learningStarts;

        protected override void DoTrainStep()
        {
            int[] indices;
            float[] tdErrors;

            if (_useSequences)
            {
                _buffer.SampleSequencesInto(_seqBatch, _perBeta);
                _agent.TrainStep(_seqBatch);
                indices = _seqBatch.Indices;
                tdErrors = _seqBatch.TDErrors;
            }
            else
            {
                _buffer.SampleInto(_batch, _perBeta);
                _agent.TrainStep(_batch);
                indices = _batch.Indices;
                tdErrors = _batch.TDErrors;
            }

            _buffer.UpdatePriorities(indices, tdErrors, _config.BatchSize);

            _perBeta = Math.Min(1f, _perBeta + _perBetaIncrement);

            long steps = Interlocked.Read(ref _totalSteps);
            if ((steps + 1) % _config.WeightSyncFrequency == 0)
                _agent.SyncInferenceWeights();

            // --- Dyna dreaming (single-step, no temporal context) ---
            if (_worldModel != null)
            {
                // World model always trains on single-step transitions.
                // When using sequences, provide a flat batch view for the world model.
                Batch wmBatch;
                if (_useSequences)
                {
                    if (_batch.Obs == null)
                        _batch = new Batch(_config.BatchSize, _buffer.ObsDim, _buffer.ActDim);
                    _buffer.SampleInto(_batch, _perBeta);
                    wmBatch = _batch;
                }
                else
                {
                    wmBatch = _batch;
                }

                _lastWorldModelLoss = _worldModel.TrainStep(wmBatch);

                int wmSteps = _worldModel.TrainSteps;
                if (wmSteps <= 100 && wmSteps % 100 == 0 ||
                    wmSteps <= 1000 && wmSteps % 500 == 0 ||
                    wmSteps % 5000 == 0)
                {
                    Debug.Log($"SACSkillTrainer: World model step {wmSteps} — " +
                        $"loss={_lastWorldModelLoss:F4}" +
                        (wmSteps < _config.DreamWarmupSteps
                            ? $", warmup {wmSteps}/{_config.DreamWarmupSteps}"
                            : ", dreaming active"));
                }

                if (wmSteps >= _config.DreamWarmupSteps &&
                    wmSteps % _config.DreamInterval == 0)
                {
                    for (int i = 0; i < _config.DreamBatchCount; i++)
                    {
                        _worldModel.GenerateDreamBatch(wmBatch, _agent.Actor, _dreamBatch);
                        _agent.TrainCriticOnly(_dreamBatch);
                    }
                    _dreamPhaseCount++;

                    if (_dreamPhaseCount <= 5 || _dreamPhaseCount % 50 == 0)
                    {
                        Debug.Log($"SACSkillTrainer: Dream phase #{_dreamPhaseCount} — " +
                            $"wmLoss={_lastWorldModelLoss:F4}, wmSteps={wmSteps}, " +
                            $"{_config.DreamBatchCount}x{_config.DreamBatchSize} imagined transitions");
                    }
                }
            }
        }

        public override void Save(string directory)
        {
            _agent.Save(directory);
            var path = System.IO.Path.Combine(directory, "replay_buffer.bin");
            using var bw = new System.IO.BinaryWriter(System.IO.File.Create(path));
            _buffer.Save(bw);

            _worldModel?.Save(directory);
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

                if (_worldModel != null)
                {
                    if (_worldModel.Load(directory))
                        Debug.Log($"SACSkillTrainer: World model restored — " +
                            $"{_worldModel.TrainSteps} steps, loss={_worldModel.LastLoss:F4}");
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
            _worldModel?.Dispose();
        }
    }
}
