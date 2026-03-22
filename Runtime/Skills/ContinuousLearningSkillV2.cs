using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using UnityEngine;
using Mujoco;
using Genesis.Sentience.Synth;
using Random = System.Random;
using Debug = UnityEngine.Debug;
using TorchSharp;
using Tensor = TorchSharp.torch.Tensor;

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

        // Temporal context history (null when ContextDim == 0)
        private HistoryRingBuffer _historyBuffer;
        private Tensor _historyTensor;  // preallocated (seqLen, 1, entryDim)
        private Tensor _historyMask;    // preallocated (1, seqLen) float

        // Drag force state
        private float _standingZ;
        private float _standingHeadZ;
        private int _headBodyIdx = -1;
        private long _dragDecisionCount;
        private float _dragOU; // current OU-drifting drag magnitude
        private Random _dragRng;

        // Upper body indices for distributed drag (chest, spine, shoulders)
        private int[] _upperBodyIdx;
        private float[] _upperBodyStandingZ;

        // Action guide state
        private float[] _guideAction;   // standing-pose ctrl for each included actuator
        private long _guideDecisionCount;
        private Random _guideRng;
        private bool _lastStepWasGuided;

        // ── Diagnostics ─────────────────────────────────────────────────
        public float CurrentDragForce => _dragOU;
        public float DragTaperProgress => sacConfig.DragTaperSteps > 0
            ? Mathf.Clamp01((float)_dragDecisionCount / sacConfig.DragTaperSteps) : 1f;
        public float GuideProb => sacConfig.ActionGuideTaperSteps > 0
            ? Mathf.Lerp(sacConfig.ActionGuideProb, sacConfig.ActionGuideProbFloor,
                Mathf.Clamp01((float)_guideDecisionCount / sacConfig.ActionGuideTaperSteps))
            : sacConfig.ActionGuideProbFloor;
        public float GuideTaperProgress => sacConfig.ActionGuideTaperSteps > 0
            ? Mathf.Clamp01((float)_guideDecisionCount / sacConfig.ActionGuideTaperSteps) : 1f;
        public bool LastStepWasGuided => _lastStepWasGuided;
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
            _standingZ = standingZ;

            var mjModel = MjScene.Instance.Model;
            var mjData = MjScene.Instance.Data;
            int nb = (int)mjModel->nbody;

            _headBodyIdx = FindBodyByName(mjModel, nb, "head");
            _standingHeadZ = _headBodyIdx > 0
                ? (float)mjData->xpos[_headBodyIdx * 3 + 2]
                : standingZ;

            _reward = new ContinuingRewardV2(standingZ, _filter.includedQposIdx,
                _headBodyIdx, _standingHeadZ);
            _reward.SetNearestFrameInterval(nearestFrameInterval);

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

            if (sacConfig.ContextDim > 0)
            {
                var (obsDim, actDim) = GetDimensions();
                int entryDim = obsDim + actDim + 1;
                _historyBuffer = new HistoryRingBuffer(sacConfig.ContextSeqLen, obsDim, actDim);
                _historyTensor = torch.zeros(sacConfig.ContextSeqLen, 1, entryDim,
                    dtype: TorchSharp.torch.ScalarType.Float32);
                _historyMask = torch.zeros(1, sacConfig.ContextSeqLen,
                    dtype: TorchSharp.torch.ScalarType.Float32);
            }

            if (sacConfig.PerJointOUSigmaEnabled)
                BuildPerJointOUSigma(mjModel);

            if (sacConfig.DragForceEnabled)
                InitDragForce(mjModel, mjData, nb);

            if (sacConfig.ActionGuideEnabled)
                InitActionGuide(mjModel, mjData);

            Debug.Log($"ContinuousLearningV2: Initialized — " +
                $"encoder={sacConfig.LatentDim > 0} (latent={sacConfig.LatentDim})" +
                $", head={(_headBodyIdx > 0 ? $"body{_headBodyIdx} (standZ={_standingHeadZ:F3})" : "NOT FOUND")}" +
                (sacConfig.ContextDim > 0 ? $", context={sacConfig.ContextDim}, seqLen={sacConfig.ContextSeqLen}" : "") +
                (sacConfig.DragForceEnabled ? $", drag={sacConfig.DragForceNewtons}N→{sacConfig.DragForceFloor}N " +
                    $"over {sacConfig.DragTaperSteps} steps (OU σ={sacConfig.DragForceOUSigma})" +
                    $", upperBody={_upperBodyIdx?.Length ?? 0} ({sacConfig.DragUpperBodyFraction:P0})" : "") +
                (sacConfig.ActionGuideEnabled ? $", guide={sacConfig.ActionGuideProb:F2}→{sacConfig.ActionGuideProbFloor:F2}" +
                    $" / {sacConfig.ActionGuideTaperSteps} steps" : "") +
                (sacConfig.PerJointOUSigmaEnabled ? ", perJointOU=ON" : "") +
                $", weights: H={rewardWeights.Height:F2} O={rewardWeights.Orientation:F2} " +
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
            if (sacConfig.DragForceEnabled)
                ApplyDragForce(MjScene.Instance.Data);

            return _reward.Compute(MjScene.Instance.Data, MjScene.Instance.Model,
                in rewardWeights, _contact, _bodyWeight);
        }

        /// <summary>
        /// Build per-joint OU sigma array by matching MuJoCo actuator names to
        /// body-part keywords. Assigns differentiated exploration scales so large
        /// joints (hips, knees) explore more while delicate joints (ankles, waist) explore less.
        /// </summary>
        private unsafe void BuildPerJointOUSigma(MujocoLib.mjModel_* model)
        {
            var indices = _filter.includedActuatorIdx;
            int actDim = indices.Length;
            var sigmaArray = new float[actDim];

            var keywordMap = new (string keyword, float sigma)[]
            {
                ("hip", sacConfig.OUSigmaHip),
                ("knee", sacConfig.OUSigmaKnee),
                ("ankle", sacConfig.OUSigmaAnkle),
                ("shoulder", sacConfig.OUSigmaShoulder),
                ("elbow", sacConfig.OUSigmaElbow),
                ("waist", sacConfig.OUSigmaWaist),
                ("torso", sacConfig.OUSigmaWaist),
                ("abdomen", sacConfig.OUSigmaWaist),
            };

            int matched = 0;
            for (int i = 0; i < actDim; i++)
            {
                int mjIdx = indices[i];
                // Read actuator name directly from model->names buffer.
                // Cannot use MujocoLib.mj_id2name — its [return: MarshalAs(LPStr)]
                // causes the marshaller to free MuJoCo's internal name pointer, crashing Unity.
                string actuatorName = ReadMjName(model, model->name_actuatoradr[mjIdx]);

                float sigma = sacConfig.OUSigmaDefault;
                if (!string.IsNullOrEmpty(actuatorName))
                {
                    string lower = actuatorName.ToLowerInvariant();
                    foreach (var (keyword, kwSigma) in keywordMap)
                    {
                        if (lower.Contains(keyword))
                        {
                            sigma = kwSigma;
                            matched++;
                            break;
                        }
                    }
                }
                sigmaArray[i] = sigma;
            }

            SACTrainer?.Agent?.SetPerJointOUSigma(sigmaArray);

            float minS = float.MaxValue, maxS = float.MinValue;
            for (int i = 0; i < actDim; i++)
            {
                if (sigmaArray[i] < minS) minS = sigmaArray[i];
                if (sigmaArray[i] > maxS) maxS = sigmaArray[i];
            }
            Debug.Log($"ContinuousLearningV2: Per-joint OU sigma built — " +
                $"{matched}/{actDim} matched keywords, range=[{minS:F2}, {maxS:F2}]");
        }

        /// <summary>
        /// Find a MuJoCo body index by searching for a substring in body names.
        /// Returns -1 if no match found.
        /// </summary>
        private static unsafe int FindBodyByName(MujocoLib.mjModel_* model, int nbody, string keyword)
        {
            string lower = keyword.ToLowerInvariant();
            for (int i = 1; i < nbody; i++)
            {
                int nameAdr = model->name_bodyadr[i];
                byte* basePtr = (byte*)model->names;
                string name = System.Runtime.InteropServices.Marshal.PtrToStringAnsi(
                    (IntPtr)(basePtr + nameAdr));
                if (!string.IsNullOrEmpty(name) && name.ToLowerInvariant().Contains(lower))
                    return i;
            }
            return -1;
        }

        /// <summary>
        /// Read a null-terminated name string from model->names at the given byte offset.
        /// Safe alternative to MujocoLib.mj_id2name whose [return: MarshalAs(LPStr)]
        /// causes the marshaller to free MuJoCo's internal pointer, crashing Unity.
        /// Note: model->names is declared as char* in C# bindings but MuJoCo stores
        /// single-byte ASCII, so we cast to IntPtr and use PtrToStringAnsi.
        /// </summary>
        private static unsafe string ReadMjName(MujocoLib.mjModel_* model, int nameAdr)
        {
            if (nameAdr < 0 || model->names == null) return null;
            // Cast to byte* first: model->names is char* (2-byte in C#) but MuJoCo
            // stores single-byte ASCII and nameAdr is a byte offset.
            byte* basePtr = (byte*)model->names;
            return System.Runtime.InteropServices.Marshal.PtrToStringAnsi((IntPtr)(basePtr + nameAdr));
        }

        /// <summary>
        /// Discover upper-body MuJoCo body indices and initialize OU drag state.
        /// </summary>
        private unsafe void InitDragForce(MujocoLib.mjModel_* model, MujocoLib.mjData_* data, int nbody)
        {
            _dragOU = sacConfig.DragForceNewtons;
            _dragRng = new Random(42);

            var upperKeywords = new[] { "chest", "spine", "abdomen",
                "lshoulder", "rshoulder", "lcollar", "rcollar" };
            var idxList = new List<int>();
            var zList = new List<float>();
            var nameList = new List<string>();

            foreach (var kw in upperKeywords)
            {
                int idx = FindBodyByName(model, nbody, kw);
                if (idx > 0)
                {
                    int nameAdr = model->name_bodyadr[idx];
                    string bodyName = ReadMjName(model, nameAdr) ?? $"body{idx}";
                    idxList.Add(idx);
                    zList.Add((float)data->xpos[idx * 3 + 2]);
                    nameList.Add($"{bodyName}(id={idx}, z={zList[zList.Count-1]:F3})");
                }
            }

            // Also include head in the upper body array
            if (_headBodyIdx > 0)
            {
                int headNameAdr = model->name_bodyadr[_headBodyIdx];
                string headName = ReadMjName(model, headNameAdr) ?? $"body{_headBodyIdx}";
                idxList.Add(_headBodyIdx);
                zList.Add(_standingHeadZ);
                nameList.Add($"{headName}(id={_headBodyIdx}, z={_standingHeadZ:F3})");
            }

            _upperBodyIdx = idxList.ToArray();
            _upperBodyStandingZ = zList.ToArray();

            Debug.Log($"ContinuousLearningV2: Drag targets — " +
                $"{_upperBodyIdx.Length} bodies:\n  {string.Join("\n  ", nameList)}");
        }

        /// <summary>
        /// Apply upward drag force to upper body bones with gradual taper.
        /// Force linearly decays from DragForceNewtons to DragForceFloor over
        /// DragTaperSteps, with OU noise for variation. The agent's own actions
        /// always matter because force is comparable to (not overwhelming) body weight.
        /// All forces are world-frame: +Z = up regardless of body orientation.
        /// </summary>
        private unsafe void ApplyDragForce(MujocoLib.mjData_* data)
        {
            _dragDecisionCount++;

            // Linear taper: start → floor over TaperSteps
            float t = sacConfig.DragTaperSteps > 0
                ? Mathf.Clamp01((float)_dragDecisionCount / sacConfig.DragTaperSteps)
                : 1f;
            float taperMean = Mathf.Lerp(sacConfig.DragForceNewtons, sacConfig.DragForceFloor, t);

            // OU noise around the tapering mean
            float noise = (float)(_dragRng.NextDouble() * 2.0 - 1.0) * 1.7320508f;
            _dragOU += sacConfig.DragForceOUTheta * (taperMean - _dragOU) + sacConfig.DragForceOUSigma * noise;
            _dragOU = Mathf.Max(0f, _dragOU);

            if (_upperBodyIdx != null)
            {
                for (int i = 0; i < _upperBodyIdx.Length; i++)
                {
                    int bIdx = _upperBodyIdx[i];
                    float bZ = (float)data->xpos[bIdx * 3 + 2];
                    float bGate = Mathf.Max(0f, 1f - bZ / _upperBodyStandingZ[i]);
                    data->xfrc_applied[bIdx * 6 + 2] = _dragOU * sacConfig.DragUpperBodyFraction * bGate;
                }
            }
        }

        /// <summary>
        /// Build the guide action profile. For Motor actuators, ctrl is direct torque
        /// (ctrl × gear = N·m), so we need actual torque patterns, not position targets.
        /// </summary>
        private unsafe void InitActionGuide(MujocoLib.mjModel_* model, MujocoLib.mjData_* data)
        {
            _guideRng = new Random(123);
            _guideAction = new float[_filter.actDim];

            switch (sacConfig.ActionGuideProfile)
            {
                case GuideProfileType.ProneRecovery:
                    BuildProneRecoveryProfile(model);
                    break;
                default:
                    break; // StandingPose: all zeros (no torque)
            }

            float absMax = 0f;
            int nonZero = 0;
            for (int i = 0; i < _guideAction.Length; i++)
            {
                absMax = Mathf.Max(absMax, Mathf.Abs(_guideAction[i]));
                if (Mathf.Abs(_guideAction[i]) > 1e-6f) nonZero++;
            }

            Debug.Log($"ContinuousLearningV2: Action guide [{sacConfig.ActionGuideProfile}] — " +
                $"actDim={_filter.actDim}, nonZero={nonZero}, |max|={absMax:F4}, " +
                $"prob={sacConfig.ActionGuideProb:F2}→{sacConfig.ActionGuideProbFloor:F2} " +
                $"over {sacConfig.ActionGuideTaperSteps} steps");
        }

        /// <summary>
        /// Prone recovery profile: maximum extension torque on sagittal-plane joints.
        /// Spine/chest/pelvis/hips X → negative ctrl (back extension, arching).
        /// UpperLeg X → negative ctrl (hip extension, pushing pelvis up).
        /// LowerLeg X → negative ctrl (knee extension, straightening legs).
        /// UpperArm X → positive ctrl (shoulder flexion, pushing off ground).
        /// </summary>
        private unsafe void BuildProneRecoveryProfile(MujocoLib.mjModel_* model)
        {
            float aScale = sacConfig.ActionScale;

            // keyword → (X ctrl sign, axis filter). Only X-axis matters for sagittal recovery.
            var profileMap = new (string keyword, float xCtrl)[]
            {
                ("spine",      -aScale),   // back extension
                ("chest",      -aScale),   // back extension
                ("abdomen",    -aScale),   // back extension
                ("pelvis",     -aScale),   // back extension
                ("hip",        -aScale),   // hip extension
                ("neck",       -aScale),   // head up
                ("thigh",      -aScale),   // hip extension (alternate naming)
                ("shin",       -aScale),   // knee extension
                ("knee",       -aScale),   // knee extension (alternate naming)
                ("shoulder",    aScale),   // shoulder flexion (push off ground)
                ("upperarm",    aScale),   // push down/back
                ("lowerarm",   -aScale),   // elbow extension (push off ground)
                ("elbow",      -aScale),   // elbow extension (alternate naming)
            };

            var logLines = new System.Collections.Generic.List<string>();

            for (int i = 0; i < _filter.actDim; i++)
            {
                int mjIdx = _filter.includedActuatorIdx[i];
                string name = ReadMjName(model, model->name_actuatoradr[mjIdx]);
                if (string.IsNullOrEmpty(name)) continue;

                string lower = name.ToLowerInvariant();

                // Only apply to X-axis actuators (sagittal plane)
                bool isXAxis = lower.EndsWith("x") || lower.EndsWith("jointx");

                if (!isXAxis) continue;

                foreach (var (keyword, xCtrl) in profileMap)
                {
                    if (lower.Contains(keyword))
                    {
                        _guideAction[i] = xCtrl;
                        logLines.Add($"  [{i}] {name} → ctrl={xCtrl:F2}");
                        break;
                    }
                }
            }

            Debug.Log($"ContinuousLearningV2: ProneRecovery profile — " +
                $"{logLines.Count} X-axis actuators assigned:\n{string.Join("\n", logLines)}");
        }

        /// <summary>
        /// Probabilistically replace the agent's action with the standing-pose guide.
        /// Called on both random and policy actions, so the replay buffer gets
        /// high-reward demonstrations even during the random exploration phase.
        /// The critic learns Q(s, guide) > Q(s, random), creating an action-space
        /// gradient that the actor can follow.
        /// </summary>
        protected override void PostProcessAction(float[] rawAction)
        {
            if (!sacConfig.ActionGuideEnabled || _guideAction == null) return;

            _guideDecisionCount++;
            float t = sacConfig.ActionGuideTaperSteps > 0
                ? Mathf.Clamp01((float)_guideDecisionCount / sacConfig.ActionGuideTaperSteps)
                : 1f;
            float prob = Mathf.Lerp(sacConfig.ActionGuideProb, sacConfig.ActionGuideProbFloor, t);

            if (_guideRng.NextDouble() < prob)
            {
                Buffer.BlockCopy(_guideAction, 0, rawAction, 0,
                    rawAction.Length * sizeof(float));
                _lastStepWasGuided = true;
            }
            else
            {
                _lastStepWasGuided = false;
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

        // ── Temporal context integration ─────────────────────────────────

        protected override float[] InferAction(float[] fullObs)
        {
            if (_historyBuffer == null)
                return _trainer.GetAction(fullObs);

            _historyBuffer.PackIntoTensor(_historyTensor);
            _historyBuffer.WritePaddingMask(_historyMask);

            return SACTrainer.GetActionWithContext(fullObs, _historyTensor, _historyMask);
        }

        protected override float[] InferDeterministicAction(float[] fullObs)
        {
            if (_historyBuffer == null)
                return _trainer.GetDeterministicAction(fullObs);

            _historyBuffer.PackIntoTensor(_historyTensor);
            _historyBuffer.WritePaddingMask(_historyMask);

            return SACTrainer.GetDeterministicActionWithContext(fullObs, _historyTensor, _historyMask);
        }

        protected override void OnTransitionStored(float reward, bool done)
        {
            // Push the previous transition into the history buffer. _prevObs and
            // _prevAction are safe copies (Buffer.BlockCopy'd in BaseTrainingSkill),
            // so no aliasing issues. This runs during both random and policy actions,
            // ensuring the history buffer is populated before policy inference begins.
            if (_historyBuffer != null)
                _historyBuffer.Push(_prevObs, _prevAction, reward * rewardScale);

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
                    SACTrainer?.LastWorldModelLoss ?? 0f,
                    sacConfig.DragForceEnabled ? _dragOU : 0f,
                    _lastStepWasGuided ? 1f : 0f);
            }
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
