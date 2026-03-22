using System;
using System.Collections.Generic;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Fixed-capacity ring buffer for a single scalar metric over time.
    /// Zero-allocation after construction. Thread-safe for single-writer / single-reader.
    /// </summary>
    public class MetricRingBuffer
    {
        private readonly float[] _data;
        private int _head;
        private int _count;

        public int Count => _count;
        public int Capacity => _data.Length;

        public MetricRingBuffer(int capacity)
        {
            _data = new float[capacity];
        }

        public void Push(float value)
        {
            _data[_head] = value;
            _head = (_head + 1) % _data.Length;
            if (_count < _data.Length) _count++;
        }

        /// <summary>Index from oldest (0) to newest (Count-1).</summary>
        public float this[int index]
        {
            get
            {
                int start = (_head - _count + _data.Length) % _data.Length;
                return _data[(start + index) % _data.Length];
            }
        }

        /// <summary>Most recent value, or 0 if empty.</summary>
        public float Latest => _count > 0 ? this[_count - 1] : 0f;

        public void Clear()
        {
            _head = 0;
            _count = 0;
        }

        /// <summary>
        /// Compute min/max over the last <paramref name="window"/> samples.
        /// Returns (min, max) with padding so graphs never collapse to a line.
        /// </summary>
        public (float min, float max) ComputeRange(int window)
        {
            float lo = float.MaxValue, hi = float.MinValue;
            int start = Math.Max(0, _count - window);
            for (int i = start; i < _count; i++)
            {
                float v = this[i];
                if (v < lo) lo = v;
                if (v > hi) hi = v;
            }
            if (lo >= hi) { lo -= 0.5f; hi += 0.5f; }
            return (lo, hi);
        }
    }

    /// <summary>
    /// Collects time-series training metrics in ring buffers for live visualization.
    /// Sampled at a fixed wall-clock interval from the training skill.
    /// </summary>
    public class TrainingMetrics
    {
        public const int DEFAULT_CAPACITY = 18000; // ~30 min at 10 Hz

        // --- Reward overview ---
        public readonly MetricRingBuffer RawReward;
        public readonly MetricRingBuffer CenteredReward;
        public readonly MetricRingBuffer RewardBar;

        // --- Reward components (weighted contributions) ---
        public readonly MetricRingBuffer Alive;
        public readonly MetricRingBuffer Height;
        public readonly MetricRingBuffer Orientation;
        public readonly MetricRingBuffer Energy;
        public readonly MetricRingBuffer Recovery;
        public readonly MetricRingBuffer Imitation;
        public readonly MetricRingBuffer VelocityUp;
        public readonly MetricRingBuffer Comfort;
        public readonly MetricRingBuffer FootSupport;
        public readonly MetricRingBuffer HandBrace;
        public readonly MetricRingBuffer ActiveSupport;
        public readonly MetricRingBuffer PhaseBonus;
        public readonly MetricRingBuffer StandBlend;

        // --- Training ---
        public readonly MetricRingBuffer QLoss;
        public readonly MetricRingBuffer ActorLoss;
        public readonly MetricRingBuffer AlphaLoss;
        public readonly MetricRingBuffer Alpha;
        public readonly MetricRingBuffer TrainingSPS;

        // --- State ---
        public readonly MetricRingBuffer Phase;
        public readonly MetricRingBuffer RootZ;

        // --- Curriculum ---
        public readonly MetricRingBuffer CurriculumStage;
        public readonly MetricRingBuffer ActiveJoints;

        // --- Buffer ---
        public readonly MetricRingBuffer ReplayCount;

        // --- World Model / Dreaming ---
        public readonly MetricRingBuffer WorldModelLoss;

        // --- V2 metrics ---
        public readonly MetricRingBuffer HeightFraction;
        public readonly MetricRingBuffer ContactReward;
        public readonly MetricRingBuffer DiscoveryGate;
        public readonly MetricRingBuffer AvgHeightFraction;
        public readonly MetricRingBuffer DragForce;
        public readonly MetricRingBuffer GuideActive;

        // --- Dynamic / skill-specific metrics ---
        private readonly Dictionary<string, MetricRingBuffer> _dynamic
            = new Dictionary<string, MetricRingBuffer>(16);
        private readonly int _capacity;

        public IReadOnlyDictionary<string, MetricRingBuffer> DynamicMetrics => _dynamic;

        public int TotalSamples { get; private set; }

        public TrainingMetrics(int capacity = DEFAULT_CAPACITY)
        {
            _capacity = capacity;
            RawReward = new MetricRingBuffer(capacity);
            CenteredReward = new MetricRingBuffer(capacity);
            RewardBar = new MetricRingBuffer(capacity);

            Alive = new MetricRingBuffer(capacity);
            Height = new MetricRingBuffer(capacity);
            Orientation = new MetricRingBuffer(capacity);
            Energy = new MetricRingBuffer(capacity);
            Recovery = new MetricRingBuffer(capacity);
            Imitation = new MetricRingBuffer(capacity);
            VelocityUp = new MetricRingBuffer(capacity);
            Comfort = new MetricRingBuffer(capacity);
            FootSupport = new MetricRingBuffer(capacity);
            HandBrace = new MetricRingBuffer(capacity);
            ActiveSupport = new MetricRingBuffer(capacity);
            PhaseBonus = new MetricRingBuffer(capacity);
            StandBlend = new MetricRingBuffer(capacity);

            QLoss = new MetricRingBuffer(capacity);
            ActorLoss = new MetricRingBuffer(capacity);
            AlphaLoss = new MetricRingBuffer(capacity);
            Alpha = new MetricRingBuffer(capacity);
            TrainingSPS = new MetricRingBuffer(capacity);

            Phase = new MetricRingBuffer(capacity);
            RootZ = new MetricRingBuffer(capacity);

            CurriculumStage = new MetricRingBuffer(capacity);
            ActiveJoints = new MetricRingBuffer(capacity);

            ReplayCount = new MetricRingBuffer(capacity);

            WorldModelLoss = new MetricRingBuffer(capacity);

            HeightFraction = new MetricRingBuffer(capacity);
            ContactReward = new MetricRingBuffer(capacity);
            DiscoveryGate = new MetricRingBuffer(capacity);
            AvgHeightFraction = new MetricRingBuffer(capacity);
            DragForce = new MetricRingBuffer(capacity);
            GuideActive = new MetricRingBuffer(capacity);
        }

        /// <summary>
        /// Record one sample of all metrics. Called at fixed wall-clock interval from the skill.
        /// </summary>
        public void Sample(
            in RewardSnapshot reward,
            float alpha, float qLoss, float actorLoss, float alphaLoss,
            float sps, int replayCount,
            int currStage, int activeJoints,
            float worldModelLoss = 0f)
        {
            RawReward.Push(reward.RawReward);
            CenteredReward.Push(reward.CenteredReward);
            RewardBar.Push(reward.RewardBar);

            Alive.Push(reward.Alive);
            Height.Push(reward.Height);
            Orientation.Push(reward.Orientation);
            Energy.Push(reward.Energy);
            Recovery.Push(reward.Recovery);
            Imitation.Push(reward.Imitation);
            VelocityUp.Push(reward.VelocityUp);
            Comfort.Push(reward.Comfort);
            FootSupport.Push(reward.FootSupport);
            HandBrace.Push(reward.HandBrace);
            ActiveSupport.Push(reward.ActiveSupport);
            PhaseBonus.Push(reward.PhaseBonus);
            StandBlend.Push(reward.StandBlend);

            Alpha.Push(alpha);
            QLoss.Push(qLoss);
            ActorLoss.Push(actorLoss);
            AlphaLoss.Push(alphaLoss);
            TrainingSPS.Push(sps);

            Phase.Push((float)reward.Phase);
            RootZ.Push(reward.RootZ);

            CurriculumStage.Push(currStage);
            ActiveJoints.Push(activeJoints);

            ReplayCount.Push(replayCount);

            WorldModelLoss.Push(worldModelLoss);

            TotalSamples++;
        }

        /// <summary>
        /// Record one sample of V2 metrics. Used by ContinuousLearningSkillV2.
        /// </summary>
        public void Sample(
            in RewardSnapshotV2 reward,
            float alpha, float qLoss, float actorLoss, float alphaLoss,
            float sps, int replayCount,
            float worldModelLoss, float dragForce = 0f,
            float guideActive = 0f)
        {
            RawReward.Push(reward.RawReward);
            CenteredReward.Push(reward.CenteredReward);
            RewardBar.Push(reward.RewardBar);

            Height.Push(reward.Height);
            Orientation.Push(reward.Orientation);
            Energy.Push(reward.Energy);
            Imitation.Push(reward.Imitation);
            RootZ.Push(reward.RootZ);

            ContactReward.Push(reward.Contact);
            HeightFraction.Push(reward.HeightFraction);
            DiscoveryGate.Push(reward.DiscoveryGate);
            AvgHeightFraction.Push(reward.AvgHeightFraction);
            DragForce.Push(dragForce);
            GuideActive.Push(guideActive);

            Alpha.Push(alpha);
            QLoss.Push(qLoss);
            ActorLoss.Push(actorLoss);
            AlphaLoss.Push(alphaLoss);
            TrainingSPS.Push(sps);

            ReplayCount.Push(replayCount);

            WorldModelLoss.Push(worldModelLoss);

            TotalSamples++;
        }

        /// <summary>
        /// Record common metrics plus arbitrary skill-specific key/value pairs.
        /// Use this for skills that don't have ContinuousLearning's RewardSnapshot.
        /// </summary>
        public void SampleGeneric(float rawReward, float sps, int expCount,
            Dictionary<string, float> custom = null)
        {
            RawReward.Push(rawReward);
            TrainingSPS.Push(sps);
            ReplayCount.Push(expCount);

            if (custom != null)
            {
                foreach (var kv in custom)
                {
                    if (!_dynamic.TryGetValue(kv.Key, out var buf))
                    {
                        buf = new MetricRingBuffer(_capacity);
                        _dynamic[kv.Key] = buf;
                    }
                    buf.Push(kv.Value);
                }
            }

            TotalSamples++;
        }

        /// <summary>Push a single named dynamic metric.</summary>
        public void PushDynamic(string name, float value)
        {
            if (!_dynamic.TryGetValue(name, out var buf))
            {
                buf = new MetricRingBuffer(_capacity);
                _dynamic[name] = buf;
            }
            buf.Push(value);
        }
    }
}
