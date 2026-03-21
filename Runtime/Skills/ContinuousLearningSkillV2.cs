using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using UnityEngine;
using TorchSharp;
using Mujoco;
using Genesis.Sentience.Synth;
using Random = System.Random;
using Debug = UnityEngine.Debug;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// V2 continuous learning skill: replaces hand-crafted heuristics with
    /// learned components while sharing SAC + persistence infrastructure
    /// from BaseTrainingSkill.
    ///
    /// Compared to ContinuousLearningSkill (V1):
    ///   - StateEncoder replaces AgentPhase (learned progress signal)
    ///   - ContinuingRewardV2 with 5 terms, no discrete phases
    ///   - ObservationAttention focuses on relevant proprioceptive channels
    ///   - SmoothActuatorCurriculum replaces binary ActionCurriculum
    ///   - StructuredWorldModel replaces raw-obs WorldModel
    ///   - No assisted pose teleports (dreams replace them)
    ///   - PBT-configurable reward weights
    /// </summary>
    public class ContinuousLearningSkillV2 : BaseTrainingSkill
    {
        [Header("SAC")]
        [Tooltip("SAC hyperparameters")]
        public SACConfig sacConfig = new SACConfig();

        [Header("Reward Weights (PBT-configurable)")]
        public RewardWeightsV2 rewardWeights = RewardWeightsV2.Default;

        [Header("Reference Motion")]
        [Tooltip("Animation clips used as reward attractors")]
        public AnimationClip[] referenceClips;

        [Tooltip("Sampling rate for motion extraction")]
        public float extractionFps = 30f;

        [Tooltip("FPS for nearest-frame matching")]
        public float matchingFps = 5f;

        [Tooltip("Recompute nearest-frame every N decision steps")]
        [Range(1, 16)]
        public int nearestFrameInterval = 4;

        [Tooltip("Whether clips loop")]
        public bool clipsAreLooping = true;

        [Header("Quest SAC Overrides")]
        [Tooltip("Training batch size on Quest")]
        public int questBatchSize = 128;

        [Header("Smooth Actuator Curriculum")]
        [Tooltip("Enable smooth per-joint gain curriculum")]
        public bool enableCurriculum = true;

        [Tooltip("Minimum joint gain (passive spring-damped)")]
        [Range(0.01f, 0.5f)]
        public float gainMin = 0.1f;

        [Tooltip("Base ramp rate per competency improvement")]
        public float gainRampRate = 1e-5f;

        // ── ISynthSkill ─────────────────────────────────────────────────

        public override string Name => "ContinuousLearningV2";

        // ── Internal state ──────────────────────────────────────────────

        private ContinuingRewardV2 _reward;
        private SmoothActuatorCurriculum _curriculum;
        private float _bodyWeight;
        private float _lastProgress;

        // ── Diagnostics ─────────────────────────────────────────────────

        public float Progress => _lastProgress;
        public float RawReward => _reward?.LastRawReward ?? 0f;
        public float CenteredReward => _reward?.LastCenteredReward ?? 0f;
        public float RewardBar => _reward?.RewardBar ?? 0f;
        public float NearestFrameDistance => _reward?.LastNearestFrameDistance ?? 0f;
        public float Alpha => V2Trainer?.Agent?.Alpha ?? 0f;
        public float LastQLoss => V2Trainer?.Agent?.LastQLoss ?? 0f;
        public float LastActorLoss => V2Trainer?.Agent?.LastActorLoss ?? 0f;
        public int TrainSteps => V2Trainer?.Agent?.TrainSteps ?? 0;
        public float TrainingSPS => _trainer?.StepsPerSecond ?? 0;
        public int ReplayBufferCount => _trainer?.ExperienceCount ?? 0;
        public float WorldModelLoss => V2Trainer?.LastWorldModelLoss ?? 0f;
        public float EncoderAuxLoss => V2Trainer?.LastEncoderAuxLoss ?? 0f;
        public int DreamPhaseCount => V2Trainer?.DreamPhaseCount ?? 0;
        public float AverageGain => _curriculum?.AverageGain ?? 1f;
        public float LocomotionGain => _curriculum?.LocomotionGain ?? 1f;
        public float FineMotorGain => _curriculum?.FineMotorGain ?? 1f;

        private SACSkillTrainerV2 V2Trainer => _trainer as SACSkillTrainerV2;

        // ── BaseTrainingSkill hooks ─────────────────────────────────────

        protected override ISkillTrainer CreateTrainer()
        {
            int contactDim = _filter.contactObsDim;
            int strainDim = _filter.strainObsDim;
            int smoothDim = _filter.physicsObsDim - contactDim - strainDim;

            return new SACSkillTrainerV2(sacConfig, learningStarts,
                smoothDim, contactDim, strainDim, _isMobile);
        }

        protected override (int obsDim, int actDim) GetDimensions()
        {
            int physObs = _filter.physicsObsDim;
            int actDim = _filter.actDim;
            return (physObs + actDim, actDim);
        }

        protected override void ApplyMobileOverrides()
        {
            sacConfig.BatchSize = questBatchSize;
            Debug.Log($"ContinuousLearningV2: Mobile — batch={sacConfig.BatchSize}");
        }

        protected override unsafe void OnSkillInitialize()
        {
            float standingZ = (float)MjScene.Instance.Data->qpos[2];

            _reward = new ContinuingRewardV2(standingZ, _filter.includedQposIdx);
            _reward.SetNearestFrameInterval(nearestFrameInterval);

            // Always compute body weight from the MuJoCo model — needed for
            // contact reward normalization even if _contact is discovered later.
            var mjModel = MjScene.Instance.Model;
            int nb = (int)mjModel->nbody;
            double totalMass = 0;
            for (int i = 0; i < nb; i++)
                totalMass += mjModel->body_mass[i];
            _bodyWeight = (float)(totalMass * 9.81);

            if (_contact != null)
            {
                Debug.Log($"ContinuousLearningV2: Contact rewards enabled — " +
                    $"bodyWeight={_bodyWeight:F1}N ({totalMass:F2}kg)");
            }
            else
            {
                Debug.LogWarning($"ContinuousLearningV2: _contact is NULL — " +
                    $"contact rewards will be zero! bodyWeight={_bodyWeight:F1}N");
            }

            if (referenceClips != null && referenceClips.Length > 0)
                IndexReferenceClips();

            if (enableCurriculum)
            {
                _curriculum = new SmoothActuatorCurriculum();
                _curriculum.Initialize(MjScene.Instance.Model, _filter, _entity?.BoneMapper,
                    gainMin, 1.0f, gainRampRate);
            }

            // Set target entropy proportional to effective action dimensions.
            // With smooth curriculum at gainMin, the effective DOF is actDim * gainMin.
            // This prevents alpha from diverging when most joints are near-passive.
            float effectiveDims = _filter.actDim * gainMin;
            V2Trainer?.SetTargetEntropy((int)Mathf.Max(4f, effectiveDims), sacConfig.TargetEntropyScale);
            Debug.Log($"ContinuousLearningV2: Target entropy set for " +
                $"{effectiveDims:F0} effective dims (actDim={_filter.actDim}, gainMin={gainMin:F2})");

            Debug.Log($"ContinuousLearningV2: Initialized — " +
                $"reward weights: H={rewardWeights.Height:F2} O={rewardWeights.Orientation:F2} " +
                $"C={rewardWeights.Contact:F2} E={rewardWeights.Energy:F2} I={rewardWeights.Imitation:F2}");
        }

        protected override float[] BuildFullObs(float[] rawPhysicsObs)
        {
            Buffer.BlockCopy(rawPhysicsObs, 0, _normalizedObs, 0,
                _physicsObsDim * sizeof(float));
            Buffer.BlockCopy(_smoothedAction, 0, _normalizedObs,
                _physicsObsDim * sizeof(float), _smoothedAction.Length * sizeof(float));
            return _normalizedObs;
        }

        protected override float[] TransformObservation(float[] normalizedObs)
        {
            if (V2Trainer?.Encoder == null) return normalizedObs;

            using var scope = TorchSharp.torch.NewDisposeScope();
            var obsTensor = TorchSharp.torch.tensor(normalizedObs, dtype: TorchSharp.torch.ScalarType.Float32)
                .unsqueeze(0);
            var (z, progress) = V2Trainer.Encoder.Infer(obsTensor);
            _lastProgress = progress;

            return normalizedObs;
        }

        protected override unsafe float ComputeReward()
        {
            return _reward.Compute(MjScene.Instance.Data, MjScene.Instance.Model,
                _lastProgress, in rewardWeights, _contact, _bodyWeight);
        }

        protected override void OnTransitionStored(float reward, bool done)
        {
            if (_metrics != null)
            {
                var agent = V2Trainer?.Agent;
                _metrics.Sample(
                    in _reward.LastSnapshot,
                    agent?.Alpha ?? 0f,
                    agent?.LastQLoss ?? 0f,
                    agent?.LastActorLoss ?? 0f,
                    agent?.LastAlphaLoss ?? 0f,
                    _trainer?.StepsPerSecond ?? 0f,
                    _trainer?.ExperienceCount ?? 0,
                    V2Trainer?.LastWorldModelLoss ?? 0f,
                    V2Trainer?.LastEncoderAuxLoss ?? 0f,
                    _curriculum?.AverageGain ?? 1f,
                    _curriculum?.LocomotionGain ?? 1f,
                    _curriculum?.FineMotorGain ?? 1f);
            }

            _curriculum?.Step(_reward.LastCenteredReward);

            // Periodically update target entropy as gains ramp up
            if (_curriculum != null && _totalDecisions % 500 == 0)
            {
                float effectiveDims = _filter.actDim * _curriculum.AverageGain;
                V2Trainer?.SetTargetEntropy((int)Mathf.Max(4f, effectiveDims), sacConfig.TargetEntropyScale);
            }
        }

        protected override void PostProcessAction(float[] rawAction)
        {
            _curriculum?.ApplyGains(rawAction);
        }

        protected override void SaveExtraState(string directory)
        {
            if (_reward != null)
            {
                BaseTrainingSkill.WriteBinaryTmpStatic(
                    Path.Combine(directory, "reward_v2_state.bin"),
                    bw => _reward.Save(bw));
            }
            if (_curriculum != null)
            {
                BaseTrainingSkill.WriteBinaryTmpStatic(
                    Path.Combine(directory, "smooth_curriculum_state.bin"),
                    bw => _curriculum.Save(bw));
            }
        }

        protected override void LoadExtraState(string directory)
        {
            string rewardPath = Path.Combine(directory, "reward_v2_state.bin");
            if (_reward != null && File.Exists(rewardPath))
            {
                using var br = new BinaryReader(File.OpenRead(rewardPath));
                _reward.Load(br);
            }

            string currPath = Path.Combine(directory, "smooth_curriculum_state.bin");
            if (_curriculum != null && File.Exists(currPath))
            {
                try
                {
                    using var br = new BinaryReader(File.OpenRead(currPath));
                    _curriculum.Load(br);
                    Debug.Log($"ContinuousLearningV2: Loaded curriculum — " +
                        $"avgGain={_curriculum.AverageGain:F3}");
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"ContinuousLearningV2: Curriculum load failed " +
                        $"({e.Message}), starting fresh");
                }
            }
        }

        public override Dictionary<string, float> GetDiagnostics()
        {
            var d = new Dictionary<string, float>();
            var agent = V2Trainer?.Agent;
            if (agent != null)
            {
                d["alpha"] = agent.Alpha;
                d["qLoss"] = agent.LastQLoss;
                d["actorLoss"] = agent.LastActorLoss;
                d["alphaLoss"] = agent.LastAlphaLoss;
            }
            if (_reward != null)
            {
                d["rawReward"] = _reward.LastRawReward;
                d["rewardBar"] = _reward.RewardBar;
                d["progress"] = _lastProgress;
            }
            if (_curriculum != null)
            {
                d["avgGain"] = _curriculum.AverageGain;
                d["locoGain"] = _curriculum.LocomotionGain;
                d["fineGain"] = _curriculum.FineMotorGain;
            }
            if (V2Trainer != null)
            {
                d["wmLoss"] = V2Trainer.LastWorldModelLoss;
                d["encAuxLoss"] = V2Trainer.LastEncoderAuxLoss;
                d["dreamCount"] = V2Trainer.DreamPhaseCount;
            }
            return d;
        }

        protected override void OnSkillValidate()
        {
            if (sacConfig == null) sacConfig = new SACConfig();
        }

        // ── Reference Motion Indexing (reused from V1) ──────────────────

        private const int MOTION_CACHE_VERSION = 4;

        private unsafe void IndexReferenceClips()
        {
            string cachePath = GetMotionCachePath();
            if (cachePath != null && TryLoadMotionCache(cachePath))
                return;

            var humanoidRoot = _entity != null ? _entity.gameObject : gameObject;
            var extractor = new MotionClipExtractor();
            var motionData = new MotionReferenceData[referenceClips.Length];
            var sw = Stopwatch.StartNew();

            Debug.Log($"ContinuousLearningV2: Extracting {referenceClips.Length} clips...");
            for (int i = 0; i < referenceClips.Length; i++)
            {
                if (referenceClips[i] == null) continue;
                motionData[i] = extractor.Extract(
                    referenceClips[i], humanoidRoot, MjScene.Instance.Model,
                    extractionFps, clipsAreLooping, Array.Empty<string>());
            }

            int validCount = 0;
            for (int i = 0; i < motionData.Length; i++)
                if (motionData[i] != null) validCount++;

            var validData = new MotionReferenceData[validCount];
            int idx = 0;
            for (int i = 0; i < motionData.Length; i++)
                if (motionData[i] != null) validData[idx++] = motionData[i];

            _reward.IndexMotionClips(validData, matchingFps);

            if (cachePath != null && _reward.HasReferenceFrames && _reward.NumReferenceFrames >= 2)
                SaveMotionCache(cachePath);
        }

        private string GetMotionCachePath()
        {
            if (referenceClips == null || referenceClips.Length == 0)
                return null;

            uint hash = 2166136261u;
            void feed(string s)
            {
                foreach (char c in s)
                {
                    hash ^= c;
                    hash *= 16777619u;
                }
            }

            feed($"v2-{MOTION_CACHE_VERSION}|mfps{matchingFps:F1}|efps{extractionFps:F1}|n{referenceClips.Length}|");
            var names = new string[referenceClips.Length];
            for (int i = 0; i < referenceClips.Length; i++)
                names[i] = referenceClips[i] != null ? referenceClips[i].name : "null";
            Array.Sort(names, StringComparer.Ordinal);
            foreach (var n in names) feed(n + "|");

            string dir = Path.Combine(Application.persistentDataPath, saveSubdirectory,
                gameObject.name, "motion_cache");
            return Path.Combine(dir, $"refs_v2_{hash:X8}.bin");
        }

        private bool TryLoadMotionCache(string path)
        {
            if (!File.Exists(path)) return false;
            try
            {
                using var fs = File.OpenRead(path);
                using var br = new BinaryReader(fs);
                int ver = br.ReadInt32();
                if (ver != MOTION_CACHE_VERSION) return false;
                _reward.LoadReferenceIndex(br);
                if (_reward.NumReferenceFrames < 2) return false;
                Debug.Log($"ContinuousLearningV2: Loaded motion cache — " +
                    $"{_reward.NumReferenceFrames} frames");
                return true;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"ContinuousLearningV2: Motion cache load failed — {e.Message}");
                return false;
            }
        }

        private void SaveMotionCache(string path)
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(path));
                using var fs = File.Create(path);
                using var bw = new BinaryWriter(fs);
                bw.Write(MOTION_CACHE_VERSION);
                _reward.SaveReferenceIndex(bw);
                Debug.Log($"ContinuousLearningV2: Saved motion cache — " +
                    $"{_reward.NumReferenceFrames} frames");
            }
            catch (Exception e)
            {
                Debug.LogWarning($"ContinuousLearningV2: Motion cache save failed — {e.Message}");
            }
        }
    }
}
