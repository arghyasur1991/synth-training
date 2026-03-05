using System;
using System.Diagnostics;
using System.Threading;
using Debug = UnityEngine.Debug;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Common background-thread infrastructure for all training algorithms.
    /// Provides thread lifecycle, mobile frame-budget throttling, SPS tracking,
    /// and pause/resume. Subclasses implement ReadyToTrain() and DoTrainStep().
    /// </summary>
    public abstract class BaseSkillTrainer : ISkillTrainer
    {
        protected Thread _thread;
        protected volatile bool _running;
        private volatile bool _paused;
        protected readonly ManualResetEventSlim _pauseEvent = new ManualResetEventSlim(true);
        protected readonly bool _isMobile;

        protected long _totalSteps;
        private float _sps;
        private readonly Stopwatch _spsWatch = new Stopwatch();
        private long _spsStepCounter;
        private float _lastTrainStepMs;
        private const double SPS_TIME_WINDOW_SEC = 2.0;
        private const int MEM_LOG_INTERVAL = 3000;

        public float LastGcMemMB { get; private set; }

        /// <summary>
        /// Max training steps/sec. 0 = unlimited.
        /// Default 200 keeps CPU headroom for the game loop.
        /// </summary>
        public int MaxStepsPerSecond = 200;

        /// <summary>
        /// Written by the main thread each frame for adaptive throttling.
        /// </summary>
        public volatile float LastFrameMs;

        /// <summary>
        /// Target frame time in ms. 72fps Quest = 13.9ms, 90fps = 11.1ms.
        /// </summary>
        public float TargetFrameMs = 13.9f;

        public bool IsTraining => _running && !_paused;
        public long TotalTrainSteps => Interlocked.Read(ref _totalSteps);
        public float LastTrainStepMs => _lastTrainStepMs;
        public int StepsPerSecond => (int)_sps;

        public abstract int ExperienceCount { get; }

        protected BaseSkillTrainer(bool isMobile)
        {
            _isMobile = isMobile;
        }

        public abstract void Initialize(int obsDim, int actDim, TorchSharp.torch.Device device);
        public abstract float[] GetAction(float[] obs);
        public abstract float[] GetRandomAction(Random rng);
        public abstract void StoreTransition(float[] obs, float[] action, float reward,
                                             float[] nextObs, bool done);
        public abstract void Save(string directory);
        public abstract bool Load(string directory);

        /// <summary>Can the trainer run a step right now? (e.g. enough data in buffer)</summary>
        protected abstract bool ReadyToTrain();

        /// <summary>Execute one training step (or batch of steps for PPO).</summary>
        protected abstract void DoTrainStep();

        public void StartTraining()
        {
            if (_running) return;
            _running = true;
            _paused = false;
            _pauseEvent.Set();

            var priority = _isMobile
                ? System.Threading.ThreadPriority.Lowest
                : System.Threading.ThreadPriority.BelowNormal;

            _thread = new Thread(TrainLoop)
            {
                Name = $"{GetType().Name}-Thread",
                IsBackground = true,
                Priority = priority
            };
            _thread.Start();
            Debug.Log($"{GetType().Name}: Started (maxSPS={MaxStepsPerSecond}, " +
                $"mobile={_isMobile}, priority={priority})");
        }

        public void StopTraining()
        {
            _running = false;
            _pauseEvent.Set();
            if (_thread != null && _thread.IsAlive)
                _thread.Join(timeout: TimeSpan.FromSeconds(5));
            _thread = null;
            Debug.Log($"{GetType().Name}: Stopped");
        }

        public void PauseTraining()
        {
            _paused = true;
            _pauseEvent.Reset();
        }

        public void ResumeTraining()
        {
            _paused = false;
            _pauseEvent.Set();
        }

        private void TrainLoop()
        {
            try
            {
                OnTrainLoopStart();

                _spsWatch.Restart();
                _spsStepCounter = 0;

                var throttleWatch = new Stopwatch();
                long minTicksPerStep = MaxStepsPerSecond > 0
                    ? Stopwatch.Frequency / MaxStepsPerSecond
                    : 0;
                throttleWatch.Start();

                var stepTimer = new Stopwatch();

                while (_running)
                {
                    if (_paused)
                    {
                        _pauseEvent.Wait();
                        if (!_running) break;
                        _spsWatch.Restart();
                        _spsStepCounter = 0;
                        throttleWatch.Restart();
                    }

                    if (!ReadyToTrain())
                    {
                        Thread.Sleep(50);
                        continue;
                    }

                    if (_isMobile)
                    {
                        float frameMs = LastFrameMs;
                        if (frameMs > 0f && TargetFrameMs > 0f)
                        {
                            float ratio = frameMs / TargetFrameMs;
                            if (ratio > 0.9f)
                            {
                                int sleepMs = (int)((ratio - 0.7f) * 20f);
                                if (sleepMs < 2) sleepMs = 2;
                                if (sleepMs > 50) sleepMs = 50;
                                Thread.Sleep(sleepMs);
                            }
                        }
                    }

                    if (minTicksPerStep > 0)
                    {
                        long elapsed = throttleWatch.ElapsedTicks;
                        if (elapsed < minTicksPerStep)
                        {
                            long waitTicks = minTicksPerStep - elapsed;
                            long waitMs = waitTicks * 1000 / Stopwatch.Frequency;
                            if (waitMs > 1)
                                Thread.Sleep((int)(waitMs - 1));

                            if (!_isMobile)
                            {
                                while (throttleWatch.ElapsedTicks < minTicksPerStep)
                                    Thread.SpinWait(4);
                            }
                        }
                        throttleWatch.Restart();
                    }

                    stepTimer.Restart();
                    DoTrainStep();
                    _lastTrainStepMs = (float)stepTimer.Elapsed.TotalMilliseconds;

                    long steps = Interlocked.Increment(ref _totalSteps);
                    _spsStepCounter++;

                    double elapsedSec = _spsWatch.Elapsed.TotalSeconds;
                    if (elapsedSec >= SPS_TIME_WINDOW_SEC)
                    {
                        _sps = (float)(_spsStepCounter / elapsedSec);
                        _spsWatch.Restart();
                        _spsStepCounter = 0;
                    }

                    if (steps % MEM_LOG_INTERVAL == 0)
                    {
                        LastGcMemMB = GC.GetTotalMemory(false) / (1024f * 1024f);
                        Debug.Log($"{GetType().Name}: step={steps}, SPS={_sps:F0}, " +
                            $"GC_heap={LastGcMemMB:F1}MB");
                    }

                    if (_isMobile)
                        Thread.Sleep(1);
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"{GetType().Name}: Exception — {e.GetType().Name}: " +
                    $"{e.Message}\n{e.StackTrace}");
            }
        }

        /// <summary>Called once at the start of the training loop thread, before the main loop.</summary>
        protected virtual void OnTrainLoopStart() { }

        public virtual void Dispose()
        {
            StopTraining();
            _pauseEvent?.Dispose();
        }
    }
}
