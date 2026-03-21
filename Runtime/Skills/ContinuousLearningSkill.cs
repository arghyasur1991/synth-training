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
    /// L0 skill for continuous, non-episodic learning via SAC.
    /// Inherits common infrastructure from BaseTrainingSkill; keeps only
    /// ContinuingReward, ActionCurriculum, assisted poses, and motion indexing.
    /// </summary>
    public class ContinuousLearningSkill : BaseTrainingSkill
    {
        [Header("SAC")]
        [Tooltip("SAC hyperparameters")]
        public SACConfig sacConfig = new SACConfig();

        [Header("Reference Motion")]
        [Tooltip("Animation clips used as reward attractors (nearest-frame matching)")]
        public AnimationClip[] referenceClips;

        [Tooltip("Sampling rate for motion extraction")]
        public float extractionFps = 30f;

        [Tooltip("FPS for nearest-frame matching. Lower = fewer frames to search.")]
        public float matchingFps = 5f;

        [Tooltip("Recompute nearest-frame every N decision steps.")]
        [Range(1, 16)]
        public int nearestFrameInterval = 4;

        [Tooltip("Whether clips loop")]
        public bool clipsAreLooping = true;

        [Header("Quest SAC Overrides")]
        [Tooltip("Training batch size on Quest")]
        public int questBatchSize = 128;

        [Header("Action Curriculum")]
        [Tooltip("Enable progressive action curriculum — unlock joints in stages")]
        public bool enableCurriculum = true;

        [Header("Assisted Poses")]
        [Tooltip("Periodically teleport to reference poses when stuck fallen.")]
        public bool enableAssistedPoses = true;

        [Tooltip("Base seconds in Fallen phase before teleporting.")]
        [Range(5f, 300f)]
        public float assistIntervalSeconds = 30f;

        [Tooltip("Extra seconds added per assist (annealing).")]
        [Range(0f, 30f)]
        public float assistAnnealRate = 5f;

        [Tooltip("Maximum assist interval after annealing (seconds).")]
        [Range(30f, 600f)]
        public float assistMaxInterval = 300f;

        [Tooltip("Random noise added to joint angles after teleport (radians).")]
        [Range(0f, 0.1f)]
        public float assistPoseNoise = 0.02f;

        [Tooltip("Decision steps to hold pose with zero torques after teleport.")]
        [Range(0, 600)]
        public int assistHoldFrames = 150;

        // ── ISynthSkill ─────────────────────────────────────────────────

        public override string Name => "ContinuousLearning";

        // ── Internal state ──────────────────────────────────────────────

        private ContinuingReward _reward;
        private ActionCurriculum _curriculum;
        private float _bodyWeight;

        // Assisted pose state
        private double[] _standingQpos;
        private double[] _assistQposBuf;
        private float _fallenTimer;
        private float _lastFallenStartTime;
        private bool _wasFallen;
        private int _assistCount;
        private int _assistHoldRemaining;
        private float[] _zeroAction;

        // ── Diagnostics ─────────────────────────────────────────────────

        public AgentPhase CurrentPhase => _reward?.LastPhase ?? AgentPhase.Fallen;
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
        public int AssistCount => _assistCount;
        public float FallenTimer => _fallenTimer;
        public int AssistHoldRemaining => _assistHoldRemaining;
        public float AssistEffectiveInterval => Mathf.Min(
            assistIntervalSeconds + assistAnnealRate * _assistCount, assistMaxInterval);
        public int CurriculumStage => _curriculum?.CurrentStage ?? -1;
        public int CurriculumActiveJoints => _curriculum?.ActiveActionDim ?? 0;
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
            Debug.Log($"ContinuousLearning: Mobile — batch={sacConfig.BatchSize}");
        }

        protected override unsafe void OnSkillInitialize()
        {
            int actDim = _filter.actDim;
            float standingZ = (float)MjScene.Instance.Data->qpos[2];

            _reward = new ContinuingReward(
                standingZ,
                _filter.includedQposIdx,
                _filter.includedQvelIdx,
                _filter.nbody);
            _reward.SetNearestFrameInterval(nearestFrameInterval);

            if (_contact != null)
            {
                var mjModel = MjScene.Instance.Model;
                int nb = (int)mjModel->nbody;
                double totalMass = 0;
                for (int i = 0; i < nb; i++)
                    totalMass += mjModel->body_mass[i];
                _bodyWeight = (float)(totalMass * 9.81);
                Debug.Log($"ContinuousLearning: Contact rewards enabled — " +
                    $"bodyWeight={_bodyWeight:F1}N ({totalMass:F2}kg)");
            }

            if (referenceClips != null && referenceClips.Length > 0)
                IndexReferenceClips();

            _zeroAction = new float[actDim];

            if (enableCurriculum)
            {
                _curriculum = new ActionCurriculum();
                _curriculum.Initialize(MjScene.Instance.Model, _filter, _entity?.BoneMapper);
                SACTrainer?.SetTargetEntropy(_curriculum.ActiveActionDim, sacConfig.TargetEntropyScale);
            }

            int nq = (int)MjScene.Instance.Model->nq;
            _standingQpos = new double[nq];
            for (int i = 0; i < nq; i++) _standingQpos[i] = MjScene.Instance.Data->qpos[i];
            _assistQposBuf = new double[nq];
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
            float meanStrain = _proprioSense.Strain?.MeanStrain() ?? 0f;
            return _reward.Compute(MjScene.Instance.Data, MjScene.Instance.Model,
                meanStrain, _contact, _bodyWeight);
        }

        protected override void OnTransitionStored(float reward, bool done)
        {
            float now = Time.realtimeSinceStartup;
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
                    _curriculum?.CurrentStage ?? -1,
                    _curriculum?.ActiveActionDim ?? 0,
                    SACTrainer?.LastWorldModelLoss ?? 0f);
            }
        }

        protected override void PostProcessAction(float[] rawAction)
        {
            _curriculum?.MaskActions(rawAction);

            if (_curriculum != null && _reward != null)
            {
                if (_curriculum.Step(_reward.LastPhase))
                    SACTrainer?.SetTargetEntropy(_curriculum.ActiveActionDim, sacConfig.TargetEntropyScale);
            }
        }

        protected override bool ShouldSkipDecision()
        {
            return _assistHoldRemaining > 0 || ShouldTeleportAssist();
        }

        protected override float[] OnSkipDecision()
        {
            if (_assistHoldRemaining > 0)
            {
                _assistHoldRemaining--;
                ResetToHeldPose();

                Buffer.BlockCopy(_normalizedObs, 0, _prevObs, 0,
                    _normalizedObs.Length * sizeof(float));
                Buffer.BlockCopy(_zeroAction, 0, _prevAction, 0,
                    _zeroAction.Length * sizeof(float));
                _hasPrevTransition = true;
                _totalDecisions++;

                return _zeroAction;
            }

            // Teleport
            TeleportToAssistedPose();
            _lastFallenStartTime = Time.realtimeSinceStartup;
            return null;
        }

        private bool ShouldTeleportAssist()
        {
            if (!enableAssistedPoses) return false;

            bool isFallen = _reward.LastPhase == AgentPhase.Fallen;
            if (isFallen && !_wasFallen)
                _lastFallenStartTime = Time.realtimeSinceStartup;

            _wasFallen = isFallen;
            _fallenTimer = isFallen
                ? Time.realtimeSinceStartup - _lastFallenStartTime
                : 0f;

            float effectiveInterval = Mathf.Min(
                assistIntervalSeconds + assistAnnealRate * _assistCount,
                assistMaxInterval);

            return _fallenTimer >= effectiveInterval && _assistQposBuf != null;
        }

        protected override void SaveExtraState(string directory)
        {
            if (_reward != null)
            {
                BaseTrainingSkill.WriteBinaryTmpStatic(
                    Path.Combine(directory, "reward_state.bin"),
                    bw => _reward.Save(bw));
            }
            if (_curriculum != null)
            {
                BaseTrainingSkill.WriteBinaryTmpStatic(
                    Path.Combine(directory, "curriculum_state.bin"),
                    bw => _curriculum.Save(bw));
            }
        }

        protected override void LoadExtraState(string directory)
        {
            string rewardPath = Path.Combine(directory, "reward_state.bin");
            if (_reward != null && File.Exists(rewardPath))
            {
                using var br = new BinaryReader(File.OpenRead(rewardPath));
                _reward.Load(br);
            }

            string currPath = Path.Combine(directory, "curriculum_state.bin");
            if (_curriculum != null && File.Exists(currPath))
            {
                try
                {
                    using var br = new BinaryReader(File.OpenRead(currPath));
                    _curriculum.Load(br);
                    SACTrainer?.SetTargetEntropy(_curriculum.ActiveActionDim, sacConfig.TargetEntropyScale);
                    Debug.Log($"ContinuousLearning: Loaded curriculum — stage {_curriculum.CurrentStage}");
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"ContinuousLearning: Curriculum load failed ({e.Message}), starting from stage 0");
                }
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
                d["phase"] = (float)_reward.LastPhase;
            }
            if (_curriculum != null)
            {
                d["currStage"] = _curriculum.CurrentStage;
                d["activeJoints"] = _curriculum.ActiveActionDim;
            }
            d["assistCount"] = _assistCount;
            d["fallenTimer"] = _fallenTimer;
            if (SACTrainer != null && SACTrainer.DreamEnabled)
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

        // ── Assisted Poses ──────────────────────────────────────────────

        private unsafe void TeleportToAssistedPose()
        {
            bool useStanding = !_reward.HasReferenceFrames || (_assistCount % 3 == 0);
            string poseDesc;

            if (useStanding)
            {
                Buffer.BlockCopy(_standingQpos, 0, _assistQposBuf, 0,
                    _standingQpos.Length * sizeof(double));
                poseDesc = "standing pose";
            }
            else
            {
                _reward.GetRandomReferenceQpos(_rng, _assistQposBuf);
                poseDesc = "reference frame";
            }

            var data = MjScene.Instance.Data;
            var model = MjScene.Instance.Model;
            _assistQposBuf[0] = data->qpos[0];
            _assistQposBuf[1] = data->qpos[1];

            if (assistPoseNoise > 0f)
            {
                for (int i = 7; i < _assistQposBuf.Length; i++)
                    _assistQposBuf[i] += (_rng.NextDouble() * 2.0 - 1.0) * assistPoseNoise;
            }

            int nq = Math.Min(_assistQposBuf.Length, (int)model->nq);
            for (int i = 0; i < nq; i++)
                data->qpos[i] = _assistQposBuf[i];

            int nv = (int)model->nv;
            for (int i = 0; i < nv; i++)
                data->qvel[i] = 0.0;

            MujocoLib.mj_forward(model, data);

            _hasPrevTransition = false;
            _fallenTimer = 0f;
            _assistHoldRemaining = assistHoldFrames;
            _assistCount++;

            float nextInterval = Mathf.Min(
                assistIntervalSeconds + assistAnnealRate * _assistCount,
                assistMaxInterval);
            Debug.Log($"[ContinuousLearning] Assisted pose #{_assistCount} — {poseDesc}, " +
                $"holding {assistHoldFrames} steps, rootZ={_assistQposBuf[2]:F3}, " +
                $"nextInterval={nextInterval:F0}s");
        }

        private unsafe void ResetToHeldPose()
        {
            var data = MjScene.Instance.Data;
            var model = MjScene.Instance.Model;

            int nq = Math.Min(_assistQposBuf.Length, (int)model->nq);
            for (int i = 0; i < nq; i++)
                data->qpos[i] = _assistQposBuf[i];

            int nv = (int)model->nv;
            for (int i = 0; i < nv; i++)
                data->qvel[i] = 0.0;

            int nu = (int)model->nu;
            for (int i = 0; i < nu; i++)
                data->ctrl[i] = 0.0;

            MujocoLib.mj_forward(model, data);
        }

        // ── Reference Motion Indexing ───────────────────────────────────

        private const int MOTION_CACHE_VERSION = 3;

        private unsafe void IndexReferenceClips()
        {
            string cachePath = GetMotionCachePath();
            if (cachePath != null && TryLoadMotionCache(cachePath))
                return;

            var humanoidRoot = _entity != null ? _entity.gameObject : gameObject;
            var extractor = new MotionClipExtractor();
            var motionData = new MotionReferenceData[referenceClips.Length];
            var sw = Stopwatch.StartNew();

            Debug.Log($"ContinuousLearning: Extracting {referenceClips.Length} clips...");
            for (int i = 0; i < referenceClips.Length; i++)
            {
                if (referenceClips[i] == null) continue;
                motionData[i] = extractor.Extract(
                    referenceClips[i], humanoidRoot, MjScene.Instance.Model,
                    extractionFps, clipsAreLooping, Array.Empty<string>());
                if ((i + 1) % 10 == 0 || i == referenceClips.Length - 1)
                    Debug.Log($"ContinuousLearning: Extracted {i + 1}/{referenceClips.Length} " +
                        $"({sw.ElapsedMilliseconds}ms)");
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
            {
                int uniquePoses = _reward.ValidateFrameVariance();
                if (uniquePoses >= 2)
                    SaveMotionCache(cachePath);
                else
                    Debug.LogWarning($"ContinuousLearning: Not caching — only {uniquePoses} unique poses");
            }
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

            feed($"v{MOTION_CACHE_VERSION}|mfps{matchingFps:F1}|efps{extractionFps:F1}|n{referenceClips.Length}|");
            var names = new string[referenceClips.Length];
            for (int i = 0; i < referenceClips.Length; i++)
                names[i] = referenceClips[i] != null ? referenceClips[i].name : "null";
            Array.Sort(names, StringComparer.Ordinal);
            foreach (var n in names) feed(n + "|");

            string dir = Path.Combine(Application.persistentDataPath, saveSubdirectory,
                gameObject.name, "motion_cache");
            return Path.Combine(dir, $"refs_{hash:X8}.bin");
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

                if (_reward.NumReferenceFrames < 2)
                {
                    Debug.LogWarning("ContinuousLearning: Cached data has < 2 frames, discarding");
                    return false;
                }

                Debug.Log($"ContinuousLearning: Loaded motion cache — " +
                    $"{_reward.NumReferenceFrames} frames");
                return true;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"ContinuousLearning: Motion cache load failed — {e.Message}");
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
                Debug.Log($"ContinuousLearning: Saved motion cache — {_reward.NumReferenceFrames} frames");
            }
            catch (Exception e)
            {
                Debug.LogWarning($"ContinuousLearning: Motion cache save failed — {e.Message}");
            }
        }
    }
}
