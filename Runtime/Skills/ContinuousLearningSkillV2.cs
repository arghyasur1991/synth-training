using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using UnityEngine;
using Mujoco;
using Genesis.Sentience.Synth;
using Random = System.Random;
using Debug = UnityEngine.Debug;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// V2 continuous learning skill — simplified reward, no progress signal.
    ///
    /// When LatentDim > 0, the actor compresses observations through an
    /// encoder bottleneck for better representation — but no separate
    /// progress head. The physics provides the curriculum via heightFraction.
    ///
    /// Compared to V1 (ContinuousLearningSkill):
    ///   - 5-term reward (height, orientation, contact, energy, imitation)
    ///   - No discrete AgentPhase — physics-based gating via heightFraction
    ///   - No ActionCurriculum — all joints active from step 1
    ///   - PBT-configurable reward weights
    ///   - Same V1 SACSkillTrainer + WorldModel for dreaming
    /// </summary>
    public class ContinuousLearningSkillV2 : BaseTrainingSkill
    {
        [Header("SAC")]
        [Tooltip("SAC hyperparameters. Set LatentDim > 0 to enable encoder-in-actor.")]
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

        // ── ISynthSkill ─────────────────────────────────────────────────

        public override string Name => "ContinuousLearningV2";

        // ── Internal state ──────────────────────────────────────────────

        private ContinuingRewardV2 _reward;
        private float _bodyWeight;

        // ── Diagnostics ─────────────────────────────────────────────────
        public float RawReward => _reward?.LastRawReward ?? 0f;
        public float CenteredReward => _reward?.LastCenteredReward ?? 0f;
        public float RewardBar => _reward?.RewardBar ?? 0f;
        public float NearestFrameDistance => _reward?.LastNearestFrameDistance ?? 0f;
        public float Alpha => SACTrainer?.Agent?.Alpha ?? 0f;
        public float LastQLoss => SACTrainer?.Agent?.LastQLoss ?? 0f;
        public float LastActorLoss => SACTrainer?.Agent?.LastActorLoss ?? 0f;
        public int TrainSteps => SACTrainer?.Agent?.TrainSteps ?? 0;
        public float TrainingSPS => _trainer?.StepsPerSecond ?? 0;
        public int ReplayBufferCount => _trainer?.ExperienceCount ?? 0;
        public float WorldModelLoss => SACTrainer?.LastWorldModelLoss ?? 0f;
        public int DreamPhaseCount => SACTrainer?.DreamPhaseCount ?? 0;

        private SACSkillTrainer SACTrainer => _trainer as SACSkillTrainer;

        // ── BaseTrainingSkill hooks ─────────────────────────────────────

        protected override ISkillTrainer CreateTrainer()
        {
            return new SACSkillTrainer(sacConfig, learningStarts, _isMobile);
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

            Debug.Log($"ContinuousLearningV2: Initialized — " +
                $"encoder={sacConfig.LatentDim > 0} (latent={sacConfig.LatentDim}), " +
                $"weights: H={rewardWeights.Height:F2} O={rewardWeights.Orientation:F2} " +
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

        protected override unsafe float ComputeReward()
        {
            return _reward.Compute(MjScene.Instance.Data, MjScene.Instance.Model,
                in rewardWeights, _contact, _bodyWeight);
        }

        protected override void OnTransitionStored(float reward, bool done)
        {
            if (_metrics != null)
            {
                var agent = SACTrainer?.Agent;
                _metrics.Sample(
                    in _reward.LastSnapshot,
                    agent?.Alpha ?? 0f,
                    agent?.LastQLoss ?? 0f,
                    agent?.LastActorLoss ?? 0f,
                    agent?.LastAlphaLoss ?? 0f,
                    _trainer?.StepsPerSecond ?? 0f,
                    _trainer?.ExperienceCount ?? 0,
                    SACTrainer?.LastWorldModelLoss ?? 0f);
            }
        }

        protected override void SaveExtraState(string directory)
        {
            if (_reward != null)
            {
                BaseTrainingSkill.WriteBinaryTmpStatic(
                    Path.Combine(directory, "reward_v2_state.bin"),
                    bw => _reward.Save(bw));
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
        }

        public override Dictionary<string, float> GetDiagnostics()
        {
            var d = new Dictionary<string, float>();
            var agent = SACTrainer?.Agent;
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
            }
            if (SACTrainer != null)
            {
                d["wmLoss"] = SACTrainer.LastWorldModelLoss;
                d["dreamCount"] = SACTrainer.DreamPhaseCount;
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
