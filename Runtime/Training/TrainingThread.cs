using System;
using System.Diagnostics;
using System.Threading;
using Debug = UnityEngine.Debug;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Background thread for SAC training. Optimized to avoid starving the main thread:
    ///   - Configurable max training steps per second (throttle)
    ///   - Pre-allocated Batch (zero allocation in hot loop)
    ///   - Adaptive frame-budget throttle (backs off when frames drop)
    ///   - Platform-aware thread priority and yield behavior
    ///   - SpinWait for sub-ms throttle precision (desktop only)
    /// </summary>
    public class TrainingThread
    {
        private readonly SACAgent _agent;
        private readonly ReplayBuffer _buffer;
        private readonly SACConfig _config;
        private readonly int _learningStarts;
        private readonly bool _isMobile;

        private Thread _thread;
        private volatile bool _running;
        private volatile bool _paused;
        private readonly ManualResetEventSlim _pauseEvent = new ManualResetEventSlim(true);

        private long _trainSteps;
        private float _sps;
        private readonly Stopwatch _spsWatch = new Stopwatch();
        private long _spsStepCounter;
        private const int SPS_WINDOW = 100;
        private const int MEM_LOG_INTERVAL = 3000;

        /// <summary>Last logged managed heap size in MB (for diagnostics).</summary>
        public float LastGcMemMB { get; private set; }

        /// <summary>
        /// Max training steps/sec. 0 = unlimited. Default 200 keeps CPU
        /// headroom for the game loop on typical hardware.
        /// </summary>
        public int MaxStepsPerSecond = 200;

        /// <summary>
        /// Written by the main thread (ContinuousLearningSkill.Update).
        /// Read by the training thread for adaptive throttling.
        /// </summary>
        public volatile float LastFrameMs;

        /// <summary>
        /// Target frame time in ms. 72fps Quest = 13.9ms, 90fps = 11.1ms.
        /// </summary>
        public float TargetFrameMs = 13.9f;

        public float SPS => _sps;
        public long TrainSteps => Interlocked.Read(ref _trainSteps);
        public bool IsRunning => _running;

        public TrainingThread(SACAgent agent, ReplayBuffer buffer, SACConfig config,
            int learningStarts, bool isMobile = false)
        {
            _agent = agent;
            _buffer = buffer;
            _config = config;
            _learningStarts = learningStarts;
            _isMobile = isMobile;
        }

        public void Start()
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
                Name = "SAC-Training",
                IsBackground = true,
                Priority = priority
            };
            _thread.Start();
            Debug.Log($"TrainingThread: Started (maxSPS={MaxStepsPerSecond}, " +
                $"mobile={_isMobile}, priority={priority})");
        }

        public void Stop()
        {
            _running = false;
            _pauseEvent.Set();
            if (_thread != null && _thread.IsAlive)
                _thread.Join(timeout: TimeSpan.FromSeconds(5));
            _thread = null;
            Debug.Log("TrainingThread: Stopped");
        }

        public void Pause()
        {
            _paused = true;
            _pauseEvent.Reset();
        }

        public void Resume()
        {
            _paused = false;
            _pauseEvent.Set();
        }

        private void TrainLoop()
        {
            try
            {
                var batch = new Batch(_config.BatchSize, _buffer.ObsDim, _buffer.ActDim);

                float perBeta = _config.PERBetaStart;
                float perBetaIncrement = _config.PERBetaAnnealSteps > 0
                    ? (1f - _config.PERBetaStart) / _config.PERBetaAnnealSteps
                    : 0f;

                _spsWatch.Restart();
                _spsStepCounter = 0;

                var throttleWatch = new Stopwatch();
                long minTicksPerStep = MaxStepsPerSecond > 0
                    ? Stopwatch.Frequency / MaxStepsPerSecond
                    : 0;
                throttleWatch.Start();

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

                    if (_buffer.Count < _learningStarts)
                    {
                        Thread.Sleep(50);
                        continue;
                    }

                    // Adaptive throttle: only on mobile where frame budget is tight
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

                    // SPS-based throttle
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

                    _buffer.SampleInto(batch, perBeta);
                    _agent.TrainStep(batch);
                    _buffer.UpdatePriorities(batch.Indices, batch.TDErrors, batch.Size);

                    perBeta = Math.Min(1f, perBeta + perBetaIncrement);

                    long steps = Interlocked.Increment(ref _trainSteps);
                    _spsStepCounter++;

                    if (steps % _config.WeightSyncFrequency == 0)
                        _agent.SyncInferenceWeights();

                    if (_spsStepCounter >= SPS_WINDOW)
                    {
                        double elapsedSec = _spsWatch.Elapsed.TotalSeconds;
                        if (elapsedSec > 0)
                            _sps = (float)(_spsStepCounter / elapsedSec);
                        _spsWatch.Restart();
                        _spsStepCounter = 0;
                    }

                    if (steps % MEM_LOG_INTERVAL == 0)
                    {
                        LastGcMemMB = GC.GetTotalMemory(false) / (1024f * 1024f);
                        Debug.Log($"TrainingThread: step={steps}, SPS={_sps:F0}, " +
                            $"GC_heap={LastGcMemMB:F1}MB");
                    }

                    // On mobile, yield after every step to prevent core saturation
                    if (_isMobile)
                        Thread.Sleep(1);
                }
            }
            catch (Exception e)
            {
                Debug.LogError($"TrainingThread: Exception — {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
            }
        }
    }
}
