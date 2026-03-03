using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using TorchSharp;
using static TorchSharp.torch;
using Mujoco;
using Genesis.Sentience.Synth;
using Random = System.Random;
using Debug = UnityEngine.Debug;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// L0 skill for continuous, non-episodic learning.
    ///
    /// Performance-critical design:
    ///   - Act() is zero-allocation (pre-allocated obs/action buffers)
    ///   - Save runs on a background task (never blocks main thread)
    ///   - Training thread is throttled to leave CPU for the game loop
    ///   - Lock-free inference via ping-pong actor swap
    ///   - Adaptive throttle backs off training when frames drop
    /// </summary>
    public class ContinuousLearningSkill : MonoBehaviour, ISynthSkill
    {
        [Header("Learning")]
        [Tooltip("Use MPS (Metal) for training if available")]
        public bool preferMPS = true;

        [Tooltip("Random actions for initial exploration")]
        public int learningStarts = 5000;

        [Tooltip("SAC hyperparameters")]
        public SACConfig sacConfig = new SACConfig();

        [Header("Reference Motion")]
        [Tooltip("Animation clips used as reward attractors (nearest-frame matching)")]
        public AnimationClip[] referenceClips;

        [Tooltip("Sampling rate for motion extraction (full fidelity)")]
        public float extractionFps = 30f;

        [Tooltip("FPS used for nearest-frame matching. Lower = fewer frames to search. " +
            "5fps is sufficient since we only need coarse pose similarity.")]
        public float matchingFps = 5f;

        [Tooltip("Only recompute nearest-frame every N decision steps. " +
            "Cached distance is reused between searches. Reduces per-frame cost linearly.")]
        [Range(1, 16)]
        public int nearestFrameInterval = 4;

        [Tooltip("Whether clips loop")]
        public bool clipsAreLooping = true;

        [Header("Decision")]
        [Tooltip("Physics sub-steps per decision")]
        [Range(1, 10)]
        public int frameSkip = 2;

        [Header("Training Throttle")]
        [Tooltip("Max training steps/sec (0 = unlimited). Lower = more CPU for game loop.")]
        public int maxTrainingSPS = 200;

        [Header("Quest / Mobile Overrides")]
        [Tooltip("Auto-detect Quest and apply mobile-optimized defaults")]
        public bool autoDetectMobile = true;

        [Tooltip("frameSkip override on Quest (higher = fewer inference calls per FixedUpdate)")]
        [Range(1, 10)]
        public int questFrameSkip = 6;

        [Tooltip("Max training SPS on Quest (lower = more CPU for game loop)")]
        public int questMaxTrainingSPS = 30;

        [Tooltip("Training batch size on Quest (smaller = faster per training step)")]
        public int questBatchSize = 128;

        [Header("Action Smoothing")]
        [Tooltip("Exponential smoothing factor. 0=no smoothing (instant), 1=fully smoothed (frozen). " +
            "Smoothed action = alpha * prev + (1-alpha) * new. Prevents chaotic torque oscillation.")]
        [Range(0f, 0.95f)]
        public float actionSmoothingAlpha = 0.5f;

        [Header("Reward Scaling")]
        [Tooltip("Multiplier for raw reward before storing in replay buffer. " +
            "With 225-dim actions, SAC's entropy bonus is ~400/step. Reward must be comparable " +
            "or Q-networks learn the value of randomness instead of the task.")]
        [Range(1f, 100f)]
        public float rewardScale = 5f;

        [Header("Persistence")]
        [Tooltip("Auto-save every N minutes (0 = disabled)")]
        public float autoSaveMinutes = 1f;

        [Tooltip("Save subdirectory under persistentDataPath")]
        public string saveSubdirectory = "ContinuousLearning";

        [Tooltip("Delete all saved state on next Play. Use when model architecture changes. Auto-resets after deletion.")]
        public bool deleteSavesOnStart;

        [Header("Assisted Poses")]
        [Tooltip("Periodically teleport to reference clip poses when stuck in Fallen phase. " +
            "Like a parent picking up a baby — gives the agent experience of being upright.")]
        public bool enableAssistedPoses = true;

        [Tooltip("Seconds in Fallen phase before teleporting to a reference pose.")]
        [Range(5f, 300f)]
        public float assistIntervalSeconds = 300f;

        [Tooltip("Random noise added to joint angles after teleport (radians). Varies the starting condition.")]
        [Range(0f, 0.1f)]
        public float assistPoseNoise = 0.02f;

        [Tooltip("Decision steps to hold pose with zero torques after teleport. " +
            "Joint stiffness/damping keeps the synth roughly upright, giving the agent " +
            "several high-reward transitions to learn from.")]
        [Range(0, 600)]
        public int assistHoldFrames = 600; 

        // --- ISynthSkill ---
        public string Name => "ContinuousLearning";
        public bool IsReady => _initialized;
        public int FrameSkip => frameSkip;

        // --- Internal state ---
        private bool _initialized;
        private bool _initFailed;
        private bool _isMobile;

        private SynthProprioception _proprioSense;
        private BoneFilterConfig _filter;

        private SACAgent _agent;
        private ReplayBuffer _replayBuffer;
        private ContinuingReward _reward;
        private ObservationNormalizer _obsNormalizer;
        private TrainingThread _trainingThread;
        private StatePersister _persister;

        // Pre-allocated buffers (zero-allocation Act() hot path)
        private float[] _normalizedObs;
        private float[] _prevObs;
        private float[] _prevAction;
        private float[] _smoothedAction;
        private int _physicsObsDim;
        private bool _hasPrevTransition;
        private int _totalDecisions;
        private Random _rng;
        private float _lastAutoSaveTime;

        // Assisted pose state
        private double[] _standingQpos;
        private double[] _assistQposBuf;
        private float _fallenTimer;
        private float _lastFallenStartTime;
        private bool _wasFallen;
        private int _assistCount;
        private int _assistHoldRemaining;
        private float[] _zeroAction;

        // Async save state
        private volatile bool _saveInProgress;
        private volatile bool _quitSaveStarted;
        private volatile bool _quitSaveFinished;

        // Frame time tracking for adaptive training throttle
        private volatile float _lastFrameMs;

        // --- Diagnostics ---
        public bool IsMobile => _isMobile;
        public int TotalDecisions => _totalDecisions;
        public int ReplayBufferCount => _replayBuffer?.Count ?? 0;
        public AgentPhase CurrentPhase => _reward?.LastPhase ?? AgentPhase.Fallen;
        public float RawReward => _reward?.LastRawReward ?? 0f;
        public float CenteredReward => _reward?.LastCenteredReward ?? 0f;
        public float RewardBar => _reward?.RewardBar ?? 0f;
        public float NearestFrameDistance => _reward?.LastNearestFrameDistance ?? 0f;
        public float Alpha => _agent?.Alpha ?? 0f;
        public float LastQLoss => _agent?.LastQLoss ?? 0f;
        public float LastActorLoss => _agent?.LastActorLoss ?? 0f;
        public int TrainSteps => _agent?.TrainSteps ?? 0;
        public float TrainingSPS => _trainingThread?.SPS ?? 0f;
        public bool SaveInProgress => _saveInProgress;
        public int AssistCount => _assistCount;
        public float FallenTimer => _fallenTimer;
        public int AssistHoldRemaining => _assistHoldRemaining;

        public unsafe bool Initialize()
        {
            if (_initialized) return true;
            if (_initFailed) return false;

            if (_proprioSense == null)
            {
                _proprioSense = GetComponent<SynthProprioception>();
                if (_proprioSense == null)
                    _proprioSense = GetComponentInParent<SynthProprioception>();
            }
            if (_proprioSense == null || !_proprioSense.IsReady)
                return false;

            if (!MjScene.InstanceExists || MjScene.Instance.Model == null)
                return false;

            _filter = _proprioSense.Filter;
            if (!_filter.IsValid)
                return false;

            _physicsObsDim = _filter.physicsObsDim;
            int actDim = _filter.actDim;
            int obsDim = _physicsObsDim + actDim;

            Device device;
            try
            {
                if (preferMPS && torch.mps_is_available())
                {
                    device = torch.device("mps");
                    Debug.Log("ContinuousLearningSkill: Using MPS (Metal) for training");
                }
                else
                {
                    device = torch.CPU;
                    Debug.Log("ContinuousLearningSkill: Using CPU for training");
                }
            }
            catch
            {
                device = torch.CPU;
                Debug.Log("ContinuousLearningSkill: MPS check failed, using CPU");
            }

            // Apply mobile overrides before creating the agent
#if UNITY_ANDROID && !UNITY_EDITOR
            _isMobile = true;
#endif
            if (_isMobile && autoDetectMobile)
            {
                frameSkip = questFrameSkip;
                maxTrainingSPS = questMaxTrainingSPS;
                sacConfig.BatchSize = questBatchSize;
                Debug.Log($"ContinuousLearningSkill: Mobile detected — " +
                    $"frameSkip={frameSkip}, maxSPS={maxTrainingSPS}, " +
                    $"batch={sacConfig.BatchSize}, hidden={sacConfig.Hidden1}" +
                    " (unified network, model portable across devices)");
            }

            try
            {
                var sw = Stopwatch.StartNew();

                _agent = new SACAgent(obsDim, actDim, sacConfig, device);
                _replayBuffer = new ReplayBuffer(sacConfig.BufferSize, obsDim, actDim, sacConfig.PERAlpha);
                _obsNormalizer = new ObservationNormalizer(_physicsObsDim);
                _rng = new Random();
                long msAgent = sw.ElapsedMilliseconds;

                _normalizedObs = new float[obsDim];
                _prevObs = new float[obsDim];
                _prevAction = new float[actDim];
                _smoothedAction = new float[actDim];

                float standingZ = (float)MjScene.Instance.Data->qpos[2];

                _reward = new ContinuingReward(
                    standingZ,
                    _filter.includedQposIdx,
                    _filter.includedQvelIdx,
                    _filter.nbody);
                _reward.SetNearestFrameInterval(nearestFrameInterval);

                long msReward = sw.ElapsedMilliseconds;

                if (referenceClips != null && referenceClips.Length > 0)
                    IndexReferenceClips();

                long msMotion = sw.ElapsedMilliseconds;

                _zeroAction = new float[actDim];

                int nqInit = (int)MjScene.Instance.Model->nq;
                _standingQpos = new double[nqInit];
                for (int i = 0; i < nqInit; i++) _standingQpos[i] = MjScene.Instance.Data->qpos[i];
                _assistQposBuf = new double[nqInit];

                string synthName = gameObject.name;
                _persister = new StatePersister(
                    Path.Combine(Application.persistentDataPath, saveSubdirectory, synthName));

                if (deleteSavesOnStart)
                {
                    _persister.DeleteAll();
                    deleteSavesOnStart = false;
                    Debug.Log("ContinuousLearningSkill: Deleted saved state (deleteSavesOnStart was checked)");
                    #if UNITY_EDITOR
                    UnityEditor.EditorUtility.SetDirty(this);
                    #endif
                }

                if (_persister.HasSavedState())
                {
                    try
                    {
                        _persister.Load(_agent, _replayBuffer, _obsNormalizer, _reward);
                        _totalDecisions = _persister.LoadedDecisionCount;

                        if (_persister.LoadPhysicsState(MjScene.Instance.Data))
                        {
                            MujocoLib.mj_forward(MjScene.Instance.Model, MjScene.Instance.Data);
                            Debug.Log("ContinuousLearningSkill: Physics state restored");
                        }

                        Debug.Log($"ContinuousLearningSkill: Restored state — " +
                                  $"{_totalDecisions} decisions, {_replayBuffer.Count} replay entries");
                    }
                    catch (Exception loadEx)
                    {
                        Debug.LogWarning($"ContinuousLearningSkill: Saved state corrupted " +
                            $"({loadEx.GetType().Name}: {loadEx.Message}). " +
                            $"Deleting and starting fresh.");
                        _persister.DeleteAll();

                        // Recreate agent and buffer to ensure clean state
                        _agent.Dispose();
                        _agent = new SACAgent(obsDim, actDim, sacConfig, device);
                        _replayBuffer = new ReplayBuffer(sacConfig.BufferSize, obsDim, actDim, sacConfig.PERAlpha);
                        _obsNormalizer = new ObservationNormalizer(_physicsObsDim);
                        _totalDecisions = 0;
                    }
                }

                long msState = sw.ElapsedMilliseconds;

                _trainingThread = new TrainingThread(_agent, _replayBuffer, sacConfig, learningStarts, _isMobile);
                _trainingThread.MaxStepsPerSecond = maxTrainingSPS;
                _trainingThread.Start();

                Debug.Log($"ContinuousLearningSkill: Init timing — " +
                    $"agent+buffer={msAgent}ms, reward={msReward - msAgent}ms, " +
                    $"motion={msMotion - msReward}ms, state={msState - msMotion}ms, " +
                    $"total={sw.ElapsedMilliseconds}ms");
            }
            catch (Exception e)
            {
                var inner = e;
                while (inner.InnerException != null) inner = inner.InnerException;
                Debug.LogError($"ContinuousLearningSkill: Init failed.\n" +
                               $"  Outer: {e.GetType().Name}: {e.Message}\n" +
                               $"  Root:  {inner.GetType().Name}: {inner.Message}\n" +
                               $"  Stack: {e.StackTrace}");
                _initFailed = true;
                return false;
            }

            _initialized = true;
            _lastAutoSaveTime = Time.realtimeSinceStartup;

            Debug.Log($"ContinuousLearningSkill: Initialized — " +
                      $"obs_dim={obsDim}, act_dim={actDim}, " +
                      $"frameSkip={frameSkip}, " +
                      $"maxTrainSPS={maxTrainingSPS}, " +
                      $"refs={referenceClips?.Length ?? 0} clips, " +
                      $"device={device}");

            return true;
        }

        /// <summary>
        /// Zero-allocation hot path: normalize into pre-allocated buffer,
        /// compute reward, add to replay, get action.
        /// </summary>
        public unsafe float[] Act()
        {
            if (!_initialized || _proprioSense == null || !_proprioSense.IsReady)
                return null;

            var rawObs = _proprioSense.GetObservation();
            if (rawObs == null || rawObs.Length == 0)
                return null;

            if (rawObs.Length != _physicsObsDim)
            {
                Debug.LogError($"ContinuousLearningSkill: obs dimension mismatch — " +
                    $"rawObs.Length={rawObs.Length}, expected _physicsObsDim={_physicsObsDim}. " +
                    $"Filter may have changed since init.");
                return null;
            }

            // Normalize physics obs into first part of buffer, then append smoothed prev action
            _obsNormalizer.NormalizeAndUpdateInPlace(rawObs, _normalizedObs);
            Buffer.BlockCopy(_smoothedAction, 0, _normalizedObs,
                _physicsObsDim * sizeof(float), _smoothedAction.Length * sizeof(float));

            // Store previous transition
            if (_hasPrevTransition)
            {
                float reward = _reward.Compute(MjScene.Instance.Data, MjScene.Instance.Model) * rewardScale;
                _replayBuffer.Add(_prevObs, _prevAction, reward, _normalizedObs);
            }

            // Assisted hold: physically pin the synth in the assisted pose each frame.
            // Re-sets qpos/qvel so gravity can't pull it down. Transitions are stored
            // so the agent learns the value of being upright (action=0, reward=high).
            if (_assistHoldRemaining > 0)
            {
                _assistHoldRemaining--;
                ResetToHeldPose();

                Buffer.BlockCopy(_normalizedObs, 0, _prevObs, 0, _normalizedObs.Length * sizeof(float));
                Buffer.BlockCopy(_zeroAction, 0, _prevAction, 0, _zeroAction.Length * sizeof(float));
                _hasPrevTransition = true;
                _totalDecisions++;

                return _zeroAction;
            }

            // Assisted poses: teleport when stuck fallen too long (wall-clock time)
            if (enableAssistedPoses)
            {
                bool isFallen = _reward.LastPhase == AgentPhase.Fallen;
                if (isFallen && !_wasFallen)
                    _lastFallenStartTime = Time.realtimeSinceStartup;

                _wasFallen = isFallen;
                _fallenTimer = isFallen
                    ? Time.realtimeSinceStartup - _lastFallenStartTime
                    : 0f;

                if (_fallenTimer >= assistIntervalSeconds && _assistQposBuf != null)
                {
                    TeleportToAssistedPose();
                    _lastFallenStartTime = Time.realtimeSinceStartup;
                    return null;
                }
            }

            // Get action (writes into agent's pre-allocated _actionBuffer)
            float[] rawAction;
            if (_totalDecisions < learningStarts)
                rawAction = _agent.GetRandomAction(_rng);
            else
                rawAction = _agent.GetAction(_normalizedObs);

            // Exponential action smoothing: prevents chaotic torque oscillation
            float a = actionSmoothingAlpha;
            float b = 1f - a;
            for (int i = 0; i < rawAction.Length; i++)
                _smoothedAction[i] = a * _smoothedAction[i] + b * rawAction[i];

            // Save for next transition (copy into our pre-allocated buffers)
            Buffer.BlockCopy(_normalizedObs, 0, _prevObs, 0, _normalizedObs.Length * sizeof(float));
            Buffer.BlockCopy(_smoothedAction, 0, _prevAction, 0, _smoothedAction.Length * sizeof(float));
            _hasPrevTransition = true;
            _totalDecisions++;

            // Auto-save (async, non-blocking)
            if (autoSaveMinutes > 0 &&
                Time.realtimeSinceStartup - _lastAutoSaveTime > autoSaveMinutes * 60f)
            {
                RequestAsyncSave();
                _lastAutoSaveTime = Time.realtimeSinceStartup;
            }

            return _smoothedAction;
        }

        public void AdvanceTime(float dt) { }
        public void Reset() { }

        private unsafe void TeleportToAssistedPose()
        {
            // Alternate between standing pose and random reference frames
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
            _assistQposBuf[0] = data->qpos[0]; // root X
            _assistQposBuf[1] = data->qpos[1]; // root Y

            if (assistPoseNoise > 0f)
            {
                for (int i = 7; i < _assistQposBuf.Length; i++)
                    _assistQposBuf[i] += ((_rng.NextDouble() * 2.0 - 1.0) * assistPoseNoise);
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

            double jointSumSq = 0;
            for (int i = 7; i < _assistQposBuf.Length; i++)
            {
                double d = _assistQposBuf[i] - _standingQpos[i];
                jointSumSq += d * d;
            }
            float diffFromStanding = (float)Math.Sqrt(jointSumSq);
            Debug.Log($"[ContinuousLearning] Assisted pose #{_assistCount} — {poseDesc}, " +
                $"holding {assistHoldFrames} steps, diffFromStanding={diffFromStanding:F3}, " +
                $"rootZ={_assistQposBuf[2]:F3}");
        }

        /// <summary>
        /// Re-applies the held pose each frame during assistHoldRemaining countdown.
        /// Resets qpos to _assistQposBuf and zeros qvel so the synth stays pinned.
        /// </summary>
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

        private const int MOTION_CACHE_VERSION = 3;

        private unsafe void IndexReferenceClips()
        {
            string cachePath = GetMotionCachePath();
            if (cachePath != null && TryLoadMotionCache(cachePath))
                return;

            var humanoidRoot = ResolveHumanoidRoot();
            var extractor = new MotionClipExtractor();
            var motionData = new MotionReferenceData[referenceClips.Length];
            var extractSw = Stopwatch.StartNew();

            Debug.Log($"ContinuousLearningSkill: Extracting {referenceClips.Length} clips (no valid cache)...");
            for (int i = 0; i < referenceClips.Length; i++)
            {
                if (referenceClips[i] == null) continue;
                motionData[i] = extractor.Extract(
                    referenceClips[i], humanoidRoot, MjScene.Instance.Model,
                    extractionFps, clipsAreLooping, Array.Empty<string>());
                if ((i + 1) % 10 == 0 || i == referenceClips.Length - 1)
                    Debug.Log($"ContinuousLearningSkill: Extracted {i + 1}/{referenceClips.Length} clips " +
                        $"({extractSw.ElapsedMilliseconds}ms)");
            }

            int validCount = 0;
            for (int i = 0; i < motionData.Length; i++)
                if (motionData[i] != null) validCount++;

            var validData = new MotionReferenceData[validCount];
            int idx = 0;
            for (int i = 0; i < motionData.Length; i++)
                if (motionData[i] != null)
                    validData[idx++] = motionData[i];

            _reward.IndexMotionClips(validData, matchingFps);

            if (cachePath != null && _reward.HasReferenceFrames && _reward.NumReferenceFrames >= 2)
            {
                int uniquePoses = _reward.ValidateFrameVariance();
                if (uniquePoses >= 2)
                    SaveMotionCache(cachePath);
                else
                    Debug.LogWarning($"ContinuousLearningSkill: Not caching — " +
                        $"only {uniquePoses} unique poses (extraction likely failed)");
            }
            else if (cachePath != null)
                Debug.LogWarning("ContinuousLearningSkill: Not caching — reference data appears invalid");
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
                    Debug.LogWarning("ContinuousLearningSkill: Cached data has < 2 frames, discarding");
                    return false;
                }

                Debug.Log($"ContinuousLearningSkill: Loaded motion cache — " +
                    $"{_reward.NumReferenceFrames} frames from {path}");
                return true;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"ContinuousLearningSkill: Motion cache load failed — {e.Message}");
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
                Debug.Log($"ContinuousLearningSkill: Saved motion cache — " +
                    $"{_reward.NumReferenceFrames} frames to {path}");
            }
            catch (Exception e)
            {
                Debug.LogWarning($"ContinuousLearningSkill: Motion cache save failed — {e.Message}");
            }
        }

        private GameObject ResolveHumanoidRoot()
        {
            var entity = GetComponent<SynthEntity>();
            if (entity != null) return entity.gameObject;
            entity = GetComponentInParent<SynthEntity>();
            if (entity != null) return entity.gameObject;
            return gameObject;
        }

        /// <summary>
        /// Non-blocking save: pauses training, snapshots physics on main thread,
        /// then writes everything to disk on a background task.
        /// </summary>
        private void RequestAsyncSave()
        {
            if (_saveInProgress || !_initialized || _persister == null) return;
            _saveInProgress = true;

            // Pause training so buffer/agent state is consistent
            _trainingThread?.Pause();

            double[] qposSnap = null, qvelSnap = null, ctrlSnap = null;
            try
            {
                unsafe
                {
                    if (MjScene.InstanceExists && MjScene.Instance.Data != null && MjScene.Instance.Model != null)
                    {
                        var data = MjScene.Instance.Data;
                        var model = MjScene.Instance.Model;
                        int nq = (int)model->nq;
                        int nv = (int)model->nv;
                        int nu = (int)model->nu;
                        qposSnap = new double[nq];
                        qvelSnap = new double[nv];
                        ctrlSnap = new double[nu];
                        for (int i = 0; i < nq; i++) qposSnap[i] = data->qpos[i];
                        for (int i = 0; i < nv; i++) qvelSnap[i] = data->qvel[i];
                        for (int i = 0; i < nu; i++) ctrlSnap[i] = data->ctrl[i];
                    }
                }
            }
            catch { }

            int decisions = _totalDecisions;

            // Fire off background save
            Task.Run(() =>
            {
                try
                {
                    _persister.SaveWithSnapshot(_agent, _replayBuffer, _obsNormalizer,
                        _reward, decisions, qposSnap, qvelSnap, ctrlSnap);
                }
                catch (Exception e)
                {
                    UnityEngine.Debug.LogError($"ContinuousLearningSkill: Async save failed — {e.Message}");
                }
                finally
                {
                    // Reclaim save-path temporaries while training is still paused
                    // (the pause makes this "free" — no frame impact).
                    if (_isMobile)
                        GC.Collect(0, GCCollectionMode.Optimized);

                    _trainingThread?.Resume();
                    _saveInProgress = false;
                }
            });
        }

        /// <summary>
        /// Public trigger for manual save (e.g. from Inspector button).
        /// Always async, never blocks.
        /// </summary>
        public void SaveState()
        {
            RequestAsyncSave();
        }

        void OnEnable()
        {
            Application.wantsToQuit += OnWantsToQuit;
        }

        void OnDisable()
        {
            Application.wantsToQuit -= OnWantsToQuit;
            if (_initialized && !_saveInProgress)
                RequestAsyncSave();
        }

        void OnApplicationPause(bool pause)
        {
            if (pause && _initialized && !_saveInProgress)
                RequestAsyncSave();
        }

        /// <summary>
        /// Intercept quit: fire async save, defer quit until done.
        /// Unity calls this before actually quitting. Returning false
        /// cancels the quit; we re-request quit once save finishes.
        /// </summary>
        private bool OnWantsToQuit()
        {
            if (!_initialized || _persister == null) return true;

            if (!_saveInProgress && !_quitSaveStarted)
            {
                _quitSaveStarted = true;
                _trainingThread?.Pause();

                double[] qposSnap = null, qvelSnap = null, ctrlSnap = null;
                try
                {
                    unsafe
                    {
                        if (MjScene.InstanceExists && MjScene.Instance.Data != null && MjScene.Instance.Model != null)
                        {
                            var data = MjScene.Instance.Data;
                            var model = MjScene.Instance.Model;
                            int nq = (int)model->nq;
                            int nv = (int)model->nv;
                            int nu = (int)model->nu;
                            qposSnap = new double[nq];
                            qvelSnap = new double[nv];
                            ctrlSnap = new double[nu];
                            for (int i = 0; i < nq; i++) qposSnap[i] = data->qpos[i];
                            for (int i = 0; i < nv; i++) qvelSnap[i] = data->qvel[i];
                            for (int i = 0; i < nu; i++) ctrlSnap[i] = data->ctrl[i];
                        }
                    }
                }
                catch { }

                int decisions = _totalDecisions;
                _saveInProgress = true;

                Task.Run(() =>
                {
                    try
                    {
                        _persister.SaveWithSnapshot(_agent, _replayBuffer, _obsNormalizer,
                            _reward, decisions, qposSnap, qvelSnap, ctrlSnap);
                        Debug.Log($"ContinuousLearningSkill: Quit save complete — {decisions} decisions");
                    }
                    catch (Exception e)
                    {
                        Debug.LogError($"ContinuousLearningSkill: Quit save failed — {e.Message}");
                    }
                    finally
                    {
                        _saveInProgress = false;
                        _quitSaveFinished = true;
                    }
                });

                // Cancel quit for now, we'll re-request after save
                return false;
            }

            if (_saveInProgress && !_quitSaveFinished)
                return false; // still saving, keep deferring

            return true; // save done, allow quit
        }

        void Update()
        {
            _lastFrameMs = Time.unscaledDeltaTime * 1000f;
            if (_trainingThread != null)
                _trainingThread.LastFrameMs = _lastFrameMs;

            // Re-trigger quit after async save completes
            if (_quitSaveFinished)
            {
                _quitSaveFinished = false;
                #if UNITY_EDITOR
                UnityEditor.EditorApplication.isPlaying = false;
                #else
                Application.Quit();
                #endif
            }
        }

        void OnDestroy()
        {
            // Wait for any in-flight async save before disposing
            int waitMs = 0;
            while (_saveInProgress && waitMs < 5000)
            {
                Thread.Sleep(10);
                waitMs += 10;
            }

            _trainingThread?.Stop();
            _agent?.Dispose();
        }

        #if UNITY_EDITOR
        void OnValidate()
        {
            if (sacConfig == null)
                sacConfig = new SACConfig();
        }
        #endif
    }
}
