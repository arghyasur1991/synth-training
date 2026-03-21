using System;
using UnityEngine;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Configuration for Population-Based Training.
    /// Sequential PBT: one lifetime at a time, K configs evolved via rank-based selection.
    /// </summary>
    [Serializable]
    public class PBTConfig
    {
        [Tooltip("Population size")]
        public int PopulationSize = 8;

        [Tooltip("Decision steps per agent lifetime")]
        public int LifetimeSteps = 500000;

        [Tooltip("Top fraction that survives each generation (top K/2)")]
        [Range(0.25f, 0.75f)]
        public float SurvivalRate = 0.5f;

        [Tooltip("Gaussian mutation sigma as fraction of parameter range")]
        [Range(0.01f, 0.5f)]
        public float MutationSigma = 0.2f;

        [Tooltip("Minimum base reward weight")]
        public float MinWeight = 0.01f;

        [Tooltip("Maximum base reward weight")]
        public float MaxWeight = 1.0f;

        [Tooltip("Enable PBT (otherwise uses fixed reward weights)")]
        public bool Enabled = false;

        [Tooltip("Auto-save population state every N completed lifetimes")]
        public int SaveInterval = 4;
    }

    /// <summary>
    /// A single member of the PBT population: reward weights + hyperparams + fitness.
    /// </summary>
    [Serializable]
    public class PBTMember
    {
        public RewardWeightsV2 Weights;
        public float GainRampRate;
        public float EncoderLr;

        // Fitness components
        public float TimeToFirstStand;
        public float StandingStability;
        public float TotalRewardImprovement;
        public float Fitness;

        // Tracking
        public int Generation;
        public int LifetimeStepsCompleted;
        public bool IsAlive;

        public PBTMember()
        {
            Weights = RewardWeightsV2.Default;
            GainRampRate = 1e-5f;
            EncoderLr = 5e-4f;
            IsAlive = true;
        }

        public PBTMember Clone()
        {
            return new PBTMember
            {
                Weights = Weights,
                GainRampRate = GainRampRate,
                EncoderLr = EncoderLr,
                Generation = Generation,
            };
        }
    }
}
