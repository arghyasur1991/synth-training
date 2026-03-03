using UnityEngine;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Periodic console logger for the continuous learning system.
    /// Attach to the same GameObject as ContinuousLearningSkill.
    /// Logs key metrics every logIntervalSeconds to the Unity console.
    /// </summary>
    [RequireComponent(typeof(ContinuousLearningSkill))]
    public class LearningLogger : MonoBehaviour
    {
        [Tooltip("Log interval in seconds")]
        public float logIntervalSeconds = 10f;

        [Tooltip("Enable console logging")]
        public bool enableLogging = true;

        private ContinuousLearningSkill _skill;
        private float _lastLogTime;

        // Phase distribution within the log window
        private int[] _windowPhaseCounts = new int[4];
        private int _windowPhaseTotal;

        // Running reward EMA
        private float _rewardEMA;
        private bool _emaInitialized;

        void Awake()
        {
            _skill = GetComponent<ContinuousLearningSkill>();
        }

        void FixedUpdate()
        {
            if (!enableLogging || _skill == null || !_skill.IsReady)
                return;

            // Track phase distribution
            _windowPhaseCounts[(int)_skill.CurrentPhase]++;
            _windowPhaseTotal++;

            // Track reward EMA
            float raw = _skill.RawReward;
            if (!_emaInitialized)
            {
                _rewardEMA = raw;
                _emaInitialized = true;
            }
            else
            {
                _rewardEMA = 0.99f * _rewardEMA + 0.01f * raw;
            }

            // Periodic log
            if (Time.realtimeSinceStartup - _lastLogTime >= logIntervalSeconds)
            {
                _lastLogTime = Time.realtimeSinceStartup;
                LogMetrics();
                _windowPhaseCounts = new int[4];
                _windowPhaseTotal = 0;
            }
        }

        private void LogMetrics()
        {
            string phaseDistStr = "---";
            if (_windowPhaseTotal > 0)
            {
                float pF = _windowPhaseCounts[0] * 100f / _windowPhaseTotal;
                float pR = _windowPhaseCounts[1] * 100f / _windowPhaseTotal;
                float pS = _windowPhaseCounts[2] * 100f / _windowPhaseTotal;
                float pM = _windowPhaseCounts[3] * 100f / _windowPhaseTotal;
                phaseDistStr = $"F:{pF:F0}% R:{pR:F0}% S:{pS:F0}% M:{pM:F0}%";
            }

            Debug.Log(
                $"[ContinuousLearning] " +
                $"decisions={_skill.TotalDecisions:N0} | " +
                $"train={_skill.TrainSteps:N0} | " +
                $"trainSPS={_skill.TrainingSPS:F0} | " +
                $"reward={_rewardEMA:F3} (bar={_skill.RewardBar:F3}) | " +
                $"alpha={_skill.Alpha:F3} | " +
                $"qLoss={_skill.LastQLoss:F3} | " +
                $"actLoss={_skill.LastActorLoss:F3} | " +
                $"nearDist={_skill.NearestFrameDistance:F3} | " +
                $"phase={_skill.CurrentPhase} [{phaseDistStr}] | " +
                $"replay={_skill.ReplayBufferCount:N0}"
            );
        }
    }
}
