using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
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
    /// Algorithm-agnostic abstract base for training skills. Owns the decision loop
    /// (observe → reward → terminate → act), observation normalization, action
    /// smoothing, state persistence, and diagnostics. Subclasses define:
    ///   - Which trainer to use (SAC, PPO, etc.) via CreateTrainer()
    ///   - How to compute reward via ComputeReward()
    ///   - How to augment observations via BuildFullObs()
    ///   - Episode termination logic via CheckTermination() / OnTermination()
    /// </summary>
    public abstract class BaseTrainingSkill : MonoBehaviour, ISynthSkill
    {
        [Header("Device")]
        [Tooltip("Use MPS (Metal) for training if available")]
        public bool preferMPS = true;

        [Header("Decision")]
        [Tooltip("Physics sub-steps per decision")]
        [Range(1, 10)]
        public int frameSkip = 2;

        [Tooltip("Random actions for initial exploration")]
        public int learningStarts = 5000;

        [Header("Training Throttle")]
        [Tooltip("Max training steps/sec (0 = unlimited). Lower = more CPU for game loop.")]
        public int maxTrainingSPS = 200;

        [Header("Quest / Mobile Overrides")]
        [Tooltip("Auto-detect Quest and apply mobile-optimized defaults")]
        public bool autoDetectMobile = true;

        [Tooltip("frameSkip override on Quest")]
        [Range(1, 10)]
        public int questFrameSkip = 6;

        [Tooltip("Max training SPS on Quest")]
        public int questMaxTrainingSPS = 30;

        [Header("Action Smoothing")]
        [Tooltip("Exponential smoothing. 0=instant, 1=frozen. " +
            "Smoothed = alpha * prev + (1-alpha) * new.")]
        [Range(0f, 0.95f)]
        public float actionSmoothingAlpha = 0.5f;

        [Header("Reward Scaling")]
        [Tooltip("Multiplier for raw reward before storing in buffer.")]
        [Range(1f, 200f)]
        public float rewardScale = 50f;

        [Header("Inference")]
        [Tooltip("Run policy without training — load saved weights")]
        public bool inferenceOnly;

        [Tooltip("Use deterministic (mean) actions. Disable to use stochastic " +
                 "(noisy) actions like during training — useful for undertrained policies.")]
        public bool deterministicInference = true;

        [Header("Persistence")]
        [Tooltip("Auto-save every N minutes (0 = disabled)")]
        public float autoSaveMinutes = 1f;

        [Tooltip("Save subdirectory under persistentDataPath")]
        public string saveSubdirectory = "Training";

        [Tooltip("Delete all saved state on next Play. Auto-resets after deletion.")]
        public bool deleteSavesOnStart;

        // ── ISynthSkill ─────────────────────────────────────────────────

        public abstract string Name { get; }
        public bool IsReady => _initialized;
        public int FrameSkip => frameSkip;

        // ── Internal state ──────────────────────────────────────────────

        protected bool _initialized;
        protected bool _initFailed;
        protected bool _isMobile;

        protected ISkillTrainer _trainer;
        protected ObservationNormalizer _obsNormalizer;
        protected StatePersister _persister;

        protected SynthProprioception _proprioSense;
        protected SynthContact _contact;
        protected SynthEntity _entity;

        protected float[] _normalizedObs;
        protected float[] _prevObs;
        protected float[] _prevAction;
        protected float[] _smoothedAction;
        protected int _physicsObsDim;
        protected int _actDim;
        protected bool _hasPrevTransition;
        protected int _totalDecisions;
        protected Random _rng;
        protected BoneFilterConfig _filter;

        private float _lastAutoSaveTime;
        private volatile bool _saveInProgress;
        private volatile bool _quitSaveStarted;
        private volatile bool _quitSaveFinished;
        private volatile bool _destroyed;
        private volatile float _lastFrameMs;

        protected TrainingMetrics _metrics;
        private float _lastMetricsSampleTime;
        private const float METRICS_SAMPLE_INTERVAL = 0.1f;

        // ── Diagnostics ─────────────────────────────────────────────────

        public bool IsMobile => _isMobile;
        public int TotalDecisions => _totalDecisions;
        public ISkillTrainer Trainer => _trainer;
        public TrainingMetrics Metrics => _metrics;
        public bool SaveInProgress => _saveInProgress;
        public BoneFilterConfig Filter => _filter;

        // ── Abstract hooks ──────────────────────────────────────────────

        /// <summary>Create the ISkillTrainer for this skill (SAC, PPO, etc.).</summary>
        protected abstract ISkillTrainer CreateTrainer();

        /// <summary>Return (physicsObsDim + augmented dims, actionDim).</summary>
        protected abstract (int obsDim, int actDim) GetDimensions();

        /// <summary>
        /// Build the full raw observation by combining physics obs with
        /// skill-specific data (reference obs, smoothed action, etc.).
        /// Normalization is applied AFTER this, covering all dimensions.
        /// Must write into a pre-allocated array of size obsDim.
        /// </summary>
        protected abstract float[] BuildFullObs(float[] rawPhysicsObs);

        /// <summary>Compute the raw (unscaled) reward for the current step.</summary>
        protected abstract float ComputeReward();

        /// <summary>Called after trainer is initialized. Set up reward, curriculum, etc.</summary>
        protected abstract void OnSkillInitialize();

        /// <summary>Save any skill-specific extra state (reward bar, curriculum, etc.).</summary>
        protected abstract void SaveExtraState(string directory);

        /// <summary>Load any skill-specific extra state.</summary>
        protected abstract void LoadExtraState(string directory);

        // ── Virtual hooks (override to customize) ───────────────────────

        /// <summary>
        /// Transform normalized observations before passing to the actor/buffer.
        /// Default: identity (returns input unchanged).
        /// </summary>
        protected virtual float[] TransformObservation(float[] normalizedObs) => normalizedObs;

        /// <summary>Check if the current episode should terminate. Default: false (continuous).</summary>
        protected virtual bool CheckTermination() => false;

        /// <summary>Handle episode termination (reset MuJoCo state, counters, etc.).</summary>
        protected virtual void OnTermination() { }

        /// <summary>Called after a transition is stored. Update metrics, curriculum, etc.</summary>
        protected virtual void OnTransitionStored(float reward, bool done) { }

        /// <summary>Return true to skip the normal decision this step (e.g. assisted hold).</summary>
        protected virtual bool ShouldSkipDecision() => false;

        /// <summary>Handle the skipped decision step (e.g. pin pose, return zero action).</summary>
        protected virtual float[] OnSkipDecision() => null;

        /// <summary>Skill-specific diagnostics for the dashboard.</summary>
        public virtual Dictionary<string, float> GetDiagnostics() => null;

        /// <summary>
        /// Apply mobile overrides to trainer-specific config before initialization.
        /// Called when mobile is detected and autoDetectMobile is enabled.
        /// </summary>
        protected virtual void ApplyMobileOverrides() { }

        // ── Initialization ──────────────────────────────────────────────

        public unsafe bool Initialize()
        {
            if (_initialized) return true;
            if (_initFailed) return false;

            // Wait for background model extraction (started at app launch)
            ModelBootstrap.Start(); // idempotent safety net
            if (!ModelBootstrap.IsComplete)
                return false;

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

            _entity = GetComponent<SynthEntity>() ?? GetComponentInParent<SynthEntity>();
            _contact = GetComponent<SynthContact>();
            if (_contact == null) _contact = GetComponentInParent<SynthContact>();
            if (_contact == null) _contact = GetComponentInChildren<SynthContact>(true);

            var (obsDim, actDim) = GetDimensions();
            _physicsObsDim = _filter.physicsObsDim;
            _actDim = actDim;

            Device device;
            try
            {
                if (preferMPS && torch.mps_is_available())
                {
                    device = torch.device("mps");
                    Debug.Log($"{Name}: Using MPS (Metal) for training");
                }
                else
                {
                    device = torch.CPU;
                    Debug.Log($"{Name}: Using CPU for training");
                }
            }
            catch
            {
                device = torch.CPU;
                Debug.Log($"{Name}: MPS check failed, using CPU");
            }

#if UNITY_ANDROID && !UNITY_EDITOR
            _isMobile = true;
#endif
            if (_isMobile && autoDetectMobile)
            {
                frameSkip = questFrameSkip;
                maxTrainingSPS = questMaxTrainingSPS;
                ApplyMobileOverrides();
                Debug.Log($"{Name}: Mobile detected — frameSkip={frameSkip}, maxSPS={maxTrainingSPS}");
            }

            try
            {
                var sw = Stopwatch.StartNew();

                _trainer = CreateTrainer();
                _trainer.Initialize(obsDim, actDim, device);

                _obsNormalizer = new ObservationNormalizer(obsDim);
                _rng = new Random();

                _normalizedObs = new float[obsDim];
                _prevObs = new float[obsDim];
                _prevAction = new float[actDim];
                _smoothedAction = new float[actDim];

                long msTrainer = sw.ElapsedMilliseconds;

                OnSkillInitialize();

                long msSkill = sw.ElapsedMilliseconds;

                string synthName = gameObject.name;
                _persister = new StatePersister(
                    Path.Combine(Application.persistentDataPath, saveSubdirectory, synthName));

                if (deleteSavesOnStart)
                {
                    _persister.DeleteAll();
                    deleteSavesOnStart = false;
                    Debug.Log($"{Name}: Deleted saved state (deleteSavesOnStart was checked)");
#if UNITY_EDITOR
                    UnityEditor.EditorUtility.SetDirty(this);
#endif
                }

                if (_persister.HasSavedState())
                {
                    try
                    {
                        LoadState();
                        Debug.Log($"{Name}: Restored state — " +
                                  $"{_totalDecisions} decisions, {_trainer.ExperienceCount} experiences");
                    }
                    catch (Exception loadEx)
                    {
                        Debug.LogWarning($"{Name}: Saved state corrupted " +
                            $"({loadEx.GetType().Name}: {loadEx.Message}). Starting fresh.");
                        _persister.DeleteAll();
                        _trainer.Dispose();
                        _trainer = CreateTrainer();
                        _trainer.Initialize(obsDim, actDim, device);
                        _obsNormalizer = new ObservationNormalizer(obsDim);
                        _totalDecisions = 0;
                    }
                }

                long msState = sw.ElapsedMilliseconds;

                if (inferenceOnly)
                {
                    _totalDecisions = 0;
                }
                else
                {
                    if (_trainer is BaseSkillTrainer bst)
                        bst.MaxStepsPerSecond = maxTrainingSPS;
                    _trainer.StartTraining();
                }

                _metrics = new TrainingMetrics();

                Debug.Log($"{Name}: Init timing — trainer={msTrainer}ms, " +
                    $"skill={msSkill - msTrainer}ms, state={msState - msSkill}ms, " +
                    $"total={sw.ElapsedMilliseconds}ms");
            }
            catch (Exception e)
            {
                var inner = e;
                while (inner.InnerException != null) inner = inner.InnerException;
                Debug.LogError($"{Name}: Init failed.\n" +
                               $"  Outer: {e.GetType().Name}: {e.Message}\n" +
                               $"  Root:  {inner.GetType().Name}: {inner.Message}\n" +
                               $"  Stack: {e.StackTrace}");
                _initFailed = true;
                return false;
            }

            _initialized = true;
            _lastAutoSaveTime = Time.realtimeSinceStartup;

            string mode = inferenceOnly ? "INFERENCE" : $"TRAINING (maxSPS={maxTrainingSPS})";
            Debug.Log($"{Name}: Initialized [{mode}] — obs={obsDim}, act={actDim}, " +
                      $"frameSkip={frameSkip}");
            return true;
        }

        // ── Decision loop ───────────────────────────────────────────────

        public unsafe float[] Act()
        {
            if (_destroyed || !_initialized || _proprioSense == null || !_proprioSense.IsReady)
                return null;

            // Skill-specific skip (e.g. assisted hold) — training only
            if (!inferenceOnly && ShouldSkipDecision())
                return OnSkipDecision();

            var rawObs = _proprioSense.GetObservation();
            if (rawObs == null || rawObs.Length == 0)
                return null;

            if (rawObs.Length != _physicsObsDim)
            {
                Debug.LogError($"{Name}: obs dimension mismatch — " +
                    $"rawObs.Length={rawObs.Length}, expected {_physicsObsDim}.");
                return null;
            }

            // Build the raw full observation (physics + reference + action),
            // then normalize ALL dimensions together. This matches the legacy
            // pipeline where the normalizer covers physics AND reference,
            // keeping all inputs on the same scale for the policy network.
            var rawFullObs = BuildFullObs(rawObs);

            if (inferenceOnly)
                _obsNormalizer.NormalizeInPlace(rawFullObs, _normalizedObs);
            else
                _obsNormalizer.NormalizeAndUpdateInPlace(rawFullObs, _normalizedObs);
            var fullObs = TransformObservation(_normalizedObs);

            if (ContainsNaN(fullObs))
            {
                Debug.LogWarning($"{Name}: NaN in observations at decision {_totalDecisions}, skipping step.");
                return _smoothedAction;
            }

            // ── Inference-only path: run policy without training ──
            if (inferenceOnly)
            {
                float[] infAction = deterministicInference
                    ? _trainer.GetDeterministicAction(fullObs)
                    : _trainer.GetAction(fullObs);
                if (ContainsNaN(infAction))
                {
                    Debug.LogWarning($"{Name}: NaN in inference action at decision {_totalDecisions}, zeroing.");
                    Array.Clear(infAction, 0, infAction.Length);
                }
                PostProcessAction(infAction);
                for (int i = 0; i < infAction.Length; i++)
                    _smoothedAction[i] = infAction[i];
                _totalDecisions++;

                if (_totalDecisions <= 5 || _totalDecisions % 200 == 0)
                {
                    float actMax = 0f;
                    for (int i = 0; i < infAction.Length; i++)
                        actMax = Math.Max(actMax, Math.Abs(infAction[i]));

                    float obsMin = float.MaxValue, obsMax = float.MinValue;
                    for (int i = 0; i < fullObs.Length; i++)
                    {
                        obsMin = Math.Min(obsMin, fullObs[i]);
                        obsMax = Math.Max(obsMax, fullObs[i]);
                    }

                    Debug.Log($"{Name}: [inference] d={_totalDecisions} " +
                        $"|act|={actMax:F4} obs[{obsMin:F3}..{obsMax:F3}] " +
                        $"raw0={rawObs[0]:F4} raw1={rawObs[1]:F4} raw2={rawObs[2]:F4}");
                }
                return _smoothedAction;
            }

            // ── Training path ──
            if (_hasPrevTransition)
            {
                float reward = ComputeReward();
                bool done = CheckTermination();
                _trainer.StoreTransition(_prevObs, _prevAction,
                    reward * rewardScale, fullObs, done);
                OnTransitionStored(reward, done);

                if (done)
                {
                    OnTermination();
                    _hasPrevTransition = false;
                    return null;
                }
            }

            float[] rawAction;
            if (_totalDecisions < learningStarts)
                rawAction = _trainer.GetRandomAction(_rng);
            else
                rawAction = _trainer.GetAction(fullObs);

            if (ContainsNaN(rawAction))
            {
                Debug.LogWarning($"{Name}: NaN detected in raw action at decision {_totalDecisions}, zeroing.");
                Array.Clear(rawAction, 0, rawAction.Length);
            }

            PostProcessAction(rawAction);

            float a = actionSmoothingAlpha;
            float b = 1f - a;
            for (int i = 0; i < rawAction.Length; i++)
                _smoothedAction[i] = a * _smoothedAction[i] + b * rawAction[i];

            Buffer.BlockCopy(fullObs, 0, _prevObs, 0, fullObs.Length * sizeof(float));
            Buffer.BlockCopy(_smoothedAction, 0, _prevAction, 0,
                _smoothedAction.Length * sizeof(float));
            _hasPrevTransition = true;
            _totalDecisions++;

            if (autoSaveMinutes > 0 &&
                Time.realtimeSinceStartup - _lastAutoSaveTime > autoSaveMinutes * 60f)
            {
                RequestAsyncSave();
                _lastAutoSaveTime = Time.realtimeSinceStartup;
            }

            return _smoothedAction;
        }

        /// <summary>
        /// Post-process raw action before smoothing (e.g. curriculum masking).
        /// Default: no-op. Override to apply masks, clamping, etc.
        /// </summary>
        protected virtual void PostProcessAction(float[] rawAction) { }

        public virtual void AdvanceTime(float dt) { }
        public virtual void Reset() { }

        // ── Persistence ─────────────────────────────────────────────────

        private void LoadState()
        {
            var dir = _persister.SaveDirectory;

            _trainer.Load(dir);

            string normPath = Path.Combine(dir, "normalizer.bin");
            if (File.Exists(normPath))
            {
                try
                {
                    using var br = new BinaryReader(File.OpenRead(normPath));
                    _obsNormalizer.Load(br);
                }
                catch (InvalidOperationException)
                {
                    Debug.LogWarning($"{Name}: Normalizer dimension changed " +
                        $"(saved state used physics-only dim). Starting normalizer fresh.");
                }
            }

            string metaPath = Path.Combine(dir, "meta.json");
            if (File.Exists(metaPath))
            {
                var meta = JsonUtility.FromJson<LearningMetadata>(
                    File.ReadAllText(metaPath));
                _totalDecisions = meta.totalDecisions;
            }

            unsafe
            {
                if (MjScene.InstanceExists && MjScene.Instance.Data != null)
                {
                    if (_persister.LoadPhysicsState(MjScene.Instance.Data))
                    {
                        MujocoLib.mj_forward(MjScene.Instance.Model, MjScene.Instance.Data);
                        Debug.Log($"{Name}: Physics state restored");
                    }
                }
            }

            LoadExtraState(dir);
        }

        protected void RequestAsyncSave()
        {
            if (_saveInProgress || !_initialized || _persister == null || inferenceOnly) return;
            _saveInProgress = true;

            _trainer.PauseTraining();

            double[] qposSnap = null, qvelSnap = null, ctrlSnap = null;
            SnapshotPhysics(ref qposSnap, ref qvelSnap, ref ctrlSnap);

            int decisions = _totalDecisions;
            string dir = _persister.SaveDirectory;

            Task.Run(() =>
            {
                try
                {
                    Directory.CreateDirectory(dir);

                    _trainer.Save(dir);

                    WriteBinaryTmp(Path.Combine(dir, "normalizer.bin"),
                        bw => _obsNormalizer.Save(bw));

                    SaveExtraState(dir);

                    if (qposSnap != null)
                        WritePhysicsSnapshotTmp(dir, qposSnap, qvelSnap, ctrlSnap);

                    WriteMetaTmp(dir, decisions);

                    PromoteAllTmpFiles(dir);
                }
                catch (Exception e)
                {
                    Debug.LogError($"{Name}: Async save failed — {e.Message}");
                }
                finally
                {
                    if (_isMobile) GC.Collect(0, GCCollectionMode.Optimized);
                    if (!_destroyed) _trainer.ResumeTraining();
                    _saveInProgress = false;
                }
            });
        }

        public void SaveState() => RequestAsyncSave();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static bool ContainsNaN(float[] arr)
        {
            for (int i = 0; i < arr.Length; i++)
                if (float.IsNaN(arr[i]) || float.IsInfinity(arr[i]))
                    return true;
            return false;
        }

        private static unsafe void SnapshotPhysics(ref double[] qpos, ref double[] qvel, ref double[] ctrl)
        {
            try
            {
                if (MjScene.InstanceExists && MjScene.Instance.Data != null &&
                    MjScene.Instance.Model != null)
                {
                    var data = MjScene.Instance.Data;
                    var model = MjScene.Instance.Model;
                    int nq = (int)model->nq;
                    int nv = (int)model->nv;
                    int nu = (int)model->nu;
                    qpos = new double[nq];
                    qvel = new double[nv];
                    ctrl = new double[nu];
                    for (int i = 0; i < nq; i++) qpos[i] = data->qpos[i];
                    for (int i = 0; i < nv; i++) qvel[i] = data->qvel[i];
                    for (int i = 0; i < nu; i++) ctrl[i] = data->ctrl[i];
                }
            }
            catch { }
        }

        private static void WriteBinaryTmp(string finalPath, Action<BinaryWriter> write)
        {
            WriteBinaryTmpStatic(finalPath, write);
        }

        internal static void WriteBinaryTmpStatic(string finalPath, Action<BinaryWriter> write)
        {
            string tmpPath = finalPath + ".tmp";
            using var fs = new FileStream(tmpPath, FileMode.Create);
            using var bw = new BinaryWriter(fs);
            write(bw);
            fs.Flush(true);
        }

        private static void WritePhysicsSnapshotTmp(string dir, double[] qpos, double[] qvel, double[] ctrl)
        {
            WriteBinaryTmp(Path.Combine(dir, "physics_state.bin"), bw =>
            {
                bw.Write(qpos.Length);
                bw.Write(qvel?.Length ?? 0);
                bw.Write(ctrl?.Length ?? 0);
                foreach (var v in qpos) bw.Write(v);
                if (qvel != null) foreach (var v in qvel) bw.Write(v);
                if (ctrl != null) foreach (var v in ctrl) bw.Write(v);
            });
        }

        private void WriteMetaTmp(string dir, int decisions)
        {
            var meta = new LearningMetadata
            {
                totalDecisions = decisions,
                trainSteps = (int)_trainer.TotalTrainSteps,
                alpha = 0f,
                replayCount = _trainer.ExperienceCount,
                timestamp = DateTime.UtcNow.ToString("o"),
                version = 2
            };
            string tmpPath = Path.Combine(dir, "meta.json.tmp");
            File.WriteAllText(tmpPath, JsonUtility.ToJson(meta, prettyPrint: true));
        }

        private static void PromoteAllTmpFiles(string dir)
        {
            foreach (var tmpPath in Directory.GetFiles(dir, "*.tmp"))
            {
                string finalPath = tmpPath.Substring(0, tmpPath.Length - 4);
                if (File.Exists(finalPath)) File.Delete(finalPath);
                File.Move(tmpPath, finalPath);
            }
        }

        // ── Unity lifecycle ─────────────────────────────────────────────

        void OnEnable()
        {
            Application.wantsToQuit += OnWantsToQuit;
        }

        void OnDisable()
        {
            Application.wantsToQuit -= OnWantsToQuit;
            if (_initialized && !_saveInProgress && !_destroyed && !inferenceOnly)
                RequestAsyncSave();
        }

        void OnApplicationPause(bool pause)
        {
            if (pause && _initialized && !_saveInProgress && !inferenceOnly)
                RequestAsyncSave();
        }

        private bool OnWantsToQuit()
        {
            if (!_initialized || _persister == null || inferenceOnly) return true;

            if (!_saveInProgress && !_quitSaveStarted)
            {
                _quitSaveStarted = true;
                _trainer.PauseTraining();

                double[] qposSnap = null, qvelSnap = null, ctrlSnap = null;
                SnapshotPhysics(ref qposSnap, ref qvelSnap, ref ctrlSnap);

                int decisions = _totalDecisions;
                string dir = _persister.SaveDirectory;
                _saveInProgress = true;

                Task.Run(() =>
                {
                    try
                    {
                        Directory.CreateDirectory(dir);
                        _trainer.Save(dir);
                        WriteBinaryTmp(Path.Combine(dir, "normalizer.bin"),
                            bw => _obsNormalizer.Save(bw));
                        SaveExtraState(dir);
                        if (qposSnap != null)
                            WritePhysicsSnapshotTmp(dir, qposSnap, qvelSnap, ctrlSnap);
                        WriteMetaTmp(dir, decisions);
                        PromoteAllTmpFiles(dir);
                        Debug.Log($"{Name}: Quit save complete — {decisions} decisions");
                    }
                    catch (Exception e)
                    {
                        Debug.LogError($"{Name}: Quit save failed — {e.Message}");
                    }
                    finally
                    {
                        _saveInProgress = false;
                        _quitSaveFinished = true;
                    }
                });

                return false;
            }

            if (_saveInProgress && !_quitSaveFinished)
                return false;

            return true;
        }

        void Update()
        {
            if (_destroyed) return;
            _lastFrameMs = Time.unscaledDeltaTime * 1000f;
            if (_trainer is BaseSkillTrainer bst)
                bst.LastFrameMs = _lastFrameMs;

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
            _destroyed = true;

            _trainer?.StopTraining();

            int waitMs = 0;
            while (_saveInProgress && waitMs < 5000)
            {
                Thread.Sleep(10);
                waitMs += 10;
            }

            _trainer?.Dispose();
        }

#if UNITY_EDITOR
        void OnValidate()
        {
            OnSkillValidate();
        }
#endif

        /// <summary>Override for skill-specific OnValidate logic.</summary>
        protected virtual void OnSkillValidate() { }
    }
}
