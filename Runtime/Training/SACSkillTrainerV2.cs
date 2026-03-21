using System;
using System.Threading;
using TorchSharp;
using static TorchSharp.torch;
using Debug = UnityEngine.Debug;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// V2 SAC trainer: uses StructuredWorldModel (dynamics-only prediction)
    /// and StateEncoder (auxiliary progress loss). Replaces the V1 trainer's
    /// raw-obs WorldModel with structured prediction that zeroes contacts/strain.
    ///
    /// Training loop additions:
    ///   1. StructuredWorldModel trains on smooth dims only
    ///   2. StateEncoder auxiliary loss: MSE(progress, standBlend_target)
    ///   3. Dream batches have zeroed contact/strain dims
    ///   4. Encoder + attention inference weights synced alongside actor
    /// </summary>
    public class SACSkillTrainerV2 : BaseSkillTrainer
    {
        private SACAgent _agent;
        private ReplayBuffer _buffer;
        private Batch _batch;
        private readonly SACConfig _config;
        private readonly int _learningStarts;

        private float _perBeta;
        private float _perBetaIncrement;

        // V2 components
        private StructuredWorldModel _worldModel;
        private StateEncoder _encoder;
        private ObservationAttention _attention;
        private Batch _dreamBatch;
        private int _dreamPhaseCount;
        private float _lastWorldModelLoss;
        private float _lastEncoderAuxLoss;

        // Observation layout info
        private int _smoothDim;
        private int _contactDim;
        private int _strainDim;

        public SACAgent Agent => _agent;
        public ReplayBuffer Buffer => _buffer;
        public StateEncoder Encoder => _encoder;
        public ObservationAttention Attention => _attention;

        public override int ExperienceCount => _buffer?.Count ?? 0;
        public float LastWorldModelLoss => _lastWorldModelLoss;
        public float LastEncoderAuxLoss => _lastEncoderAuxLoss;
        public int DreamPhaseCount => _dreamPhaseCount;
        public bool DreamEnabled => _config.DreamEnabled && _worldModel != null;

        public SACSkillTrainerV2(SACConfig config, int learningStarts,
            int smoothDim, int contactDim, int strainDim,
            bool isMobile = false)
            : base(isMobile)
        {
            _config = config;
            _learningStarts = learningStarts;
            _smoothDim = smoothDim;
            _contactDim = contactDim;
            _strainDim = strainDim;
        }

        public override void Initialize(int obsDim, int actDim, Device device)
        {
            _agent = new SACAgent(obsDim, actDim, _config, device);
            _buffer = new ReplayBuffer(_config.BufferSize, obsDim, actDim,
                perAlpha: _config.PERAlpha);
            _batch = new Batch(_config.BatchSize, obsDim, actDim);

            _perBeta = _config.PERBetaStart;
            _perBetaIncrement = _config.PERBetaAnnealSteps > 0
                ? (1f - _config.PERBetaStart) / _config.PERBetaAnnealSteps
                : 0f;

            // V2: StateEncoder
            _encoder = new StateEncoder(obsDim, _config.PolicyLr * 0.5f, device);

            // V2: ObservationAttention
            _attention = new ObservationAttention(_encoder.ZDim, obsDim, device);

            // V2: StructuredWorldModel
            if (_config.DreamEnabled)
            {
                _worldModel = new StructuredWorldModel(
                    obsDim, actDim, _smoothDim, _contactDim, _strainDim,
                    _config.WorldModelLr, device,
                    zDim: 32, hidden: _config.WorldModelHidden1);
                _dreamBatch = new Batch(_config.DreamBatchSize, obsDim, actDim);
                Debug.Log($"SACSkillTrainerV2: StructuredWorldModel enabled — " +
                    $"smoothDim={_smoothDim}, contactDim={_contactDim}, strainDim={_strainDim}");
            }

            Debug.Log($"SACSkillTrainerV2: Initialized (obs={obsDim}, act={actDim}, " +
                $"device={device}, encoder_z={_encoder.ZDim})");
        }

        public override float[] GetAction(float[] obs) => _agent.GetAction(obs);
        public override float[] GetDeterministicAction(float[] obs) => _agent.GetDeterministicAction(obs);
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
            {
                _agent.SyncInferenceWeights();
                _encoder.SyncInferenceWeights();
                _attention.SyncInferenceWeights();
            }

            // --- Encoder auxiliary loss ---
            TrainEncoderAuxiliary(_batch);

            // --- Structured world model dreaming ---
            if (_worldModel != null)
            {
                _lastWorldModelLoss = _worldModel.TrainStep(_batch);

                int wmSteps = _worldModel.TrainSteps;
                if (wmSteps <= 100 && wmSteps % 100 == 0 ||
                    wmSteps <= 1000 && wmSteps % 500 == 0 ||
                    wmSteps % 5000 == 0)
                {
                    Debug.Log($"SACSkillTrainerV2: World model step {wmSteps} — " +
                        $"loss={_lastWorldModelLoss:F4} (dyn={_worldModel.LastDynLoss:F4}, " +
                        $"rew={_worldModel.LastRewardLoss:F4})" +
                        (wmSteps < _config.DreamWarmupSteps
                            ? $", warmup {wmSteps}/{_config.DreamWarmupSteps}"
                            : ", dreaming active"));
                }

                if (wmSteps >= _config.DreamWarmupSteps &&
                    wmSteps % _config.DreamInterval == 0)
                {
                    for (int i = 0; i < _config.DreamBatchCount; i++)
                    {
                        _worldModel.GenerateDreamBatch(_batch, _agent.Actor, _dreamBatch);
                        _agent.TrainCriticOnly(_dreamBatch);
                    }
                    _dreamPhaseCount++;

                    if (_dreamPhaseCount <= 5 || _dreamPhaseCount % 50 == 0)
                    {
                        Debug.Log($"SACSkillTrainerV2: Dream phase #{_dreamPhaseCount} — " +
                            $"wmLoss={_lastWorldModelLoss:F4}, wmSteps={wmSteps}");
                    }
                }
            }
        }

        /// <summary>
        /// Train encoder's auxiliary loss: MSE(progress, standBlend_target).
        /// standBlend is computed from rootZ in the batch observations.
        /// rootZ is at obs index 0 (first element of physics obs = qpos[2]).
        /// </summary>
        private void TrainEncoderAuxiliary(Batch batch)
        {
            using var scope = NewDisposeScope();

            var enumerator = _agent.Actor.parameters().GetEnumerator();
            if (!enumerator.MoveNext() || enumerator.Current == null) return;
            var device = enumerator.Current.device;
            var obs = torch.tensor(batch.Obs).reshape(batch.Size, batch.ObsDim).to(device);

            var rootZ = obs.select(1, 0);
            const float FALLEN_Z = 0.25f;
            const float STANDING_Z = 0.7f;
            var standBlend = ((rootZ - FALLEN_Z) / (STANDING_Z - FALLEN_Z)).clamp(0f, 1f);
            var targets = standBlend.unsqueeze(1);

            _lastEncoderAuxLoss = _encoder.TrainAuxiliary(obs, targets);
        }

        public void SetTargetEntropy(int activeDims, float entropyScale)
        {
            _agent?.SetTargetEntropy(activeDims, entropyScale);
        }

        public override void Save(string directory)
        {
            _agent.Save(directory);
            var path = System.IO.Path.Combine(directory, "replay_buffer.bin");
            using var bw = new System.IO.BinaryWriter(System.IO.File.Create(path));
            _buffer.Save(bw);

            _encoder?.Save(directory);
            _attention?.Save(directory);
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

                if (_encoder?.Load(directory) == true)
                    Debug.Log($"SACSkillTrainerV2: Encoder restored — {_encoder.TrainSteps} steps");
                _attention?.Load(directory);

                if (_worldModel?.Load(directory) == true)
                    Debug.Log($"SACSkillTrainerV2: StructuredWorldModel restored — " +
                        $"{_worldModel.TrainSteps} steps");

                return true;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"SACSkillTrainerV2: Load failed — {e.Message}");
                return false;
            }
        }

        public override void Dispose()
        {
            base.Dispose();
            _agent?.Dispose();
            _encoder?.Dispose();
            _attention?.Dispose();
            _worldModel?.Dispose();
        }
    }
}
