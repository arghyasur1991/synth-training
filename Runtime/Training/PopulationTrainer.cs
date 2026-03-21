using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Random = System.Random;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Sequential Population-Based Training over reward weights and hyperparameters.
    ///
    /// Runs one lifetime at a time on the same machine. Each lifetime trains a
    /// ContinuousLearningSkillV2 with a specific set of reward weights, curriculum
    /// ramp rate, and encoder learning rate. After all members complete one generation,
    /// rank by fitness → top half survive → copy to bottom half with Gaussian mutation.
    ///
    /// Fitness: weighted combination of time_to_first_stand, standing_stability,
    /// and total_reward_improvement.
    ///
    /// Population state is fully serializable for pause/resume across sessions.
    /// </summary>
    public class PopulationTrainer
    {
        private readonly PBTConfig _config;
        private readonly Random _rng;
        private List<PBTMember> _population;
        private int _currentMemberIdx;
        private int _currentGeneration;

        // Fitness tracking for current lifetime
        private float _lifetimeStartReward;
        private float _lifetimeRewardEMA;
        private bool _hasStoodOnce;
        private float _timeToFirstStand;
        private int _standingSteps;
        private int _totalStepsThisLifetime;

        public PBTConfig Config => _config;
        public int CurrentMemberIdx => _currentMemberIdx;
        public int CurrentGeneration => _currentGeneration;
        public int PopulationSize => _population?.Count ?? 0;
        public PBTMember CurrentMember => _population != null && _currentMemberIdx < _population.Count
            ? _population[_currentMemberIdx] : null;
        public IReadOnlyList<PBTMember> Population => _population;

        public PopulationTrainer(PBTConfig config, int seed = 42)
        {
            _config = config;
            _rng = new Random(seed);
        }

        /// <summary>
        /// Initialize the population with randomized reward weights.
        /// </summary>
        public void InitializePopulation()
        {
            _population = new List<PBTMember>(_config.PopulationSize);
            var baseWeights = RewardWeightsV2.Default;

            for (int i = 0; i < _config.PopulationSize; i++)
            {
                var member = new PBTMember
                {
                    Weights = new RewardWeightsV2
                    {
                        Height = Mutate(baseWeights.Height, 0.5f),
                        Orientation = Mutate(baseWeights.Orientation, 0.5f),
                        Contact = Mutate(baseWeights.Contact, 0.5f),
                        Energy = Mutate(baseWeights.Energy, 0.5f),
                        Imitation = Mutate(baseWeights.Imitation, 0.5f),
                    },
                    GainRampRate = Mutate(1e-5f, 0.5f),
                    EncoderLr = Mutate(5e-4f, 0.5f),
                    Generation = 0,
                };
                _population.Add(member);
            }

            // First member uses default weights as anchor
            _population[0].Weights = baseWeights;
            _population[0].GainRampRate = 1e-5f;
            _population[0].EncoderLr = 5e-4f;

            _currentMemberIdx = 0;
            _currentGeneration = 0;

            Debug.Log($"PopulationTrainer: Initialized {_config.PopulationSize} members");
        }

        /// <summary>
        /// Get the reward weights and hyperparams for the current member.
        /// Apply these to ContinuousLearningSkillV2 before starting a lifetime.
        /// </summary>
        public (RewardWeightsV2 weights, float rampRate, float encoderLr) GetCurrentConfig()
        {
            var m = CurrentMember;
            return (m.Weights, m.GainRampRate, m.EncoderLr);
        }

        /// <summary>
        /// Start a new lifetime. Resets fitness tracking.
        /// </summary>
        public void BeginLifetime(float initialReward)
        {
            _lifetimeStartReward = initialReward;
            _lifetimeRewardEMA = initialReward;
            _hasStoodOnce = false;
            _timeToFirstStand = float.MaxValue;
            _standingSteps = 0;
            _totalStepsThisLifetime = 0;

            Debug.Log($"PopulationTrainer: Begin lifetime — " +
                $"gen={_currentGeneration}, member={_currentMemberIdx}/{_config.PopulationSize}, " +
                $"H={CurrentMember.Weights.Height:F2} O={CurrentMember.Weights.Orientation:F2} " +
                $"C={CurrentMember.Weights.Contact:F2} E={CurrentMember.Weights.Energy:F2} " +
                $"I={CurrentMember.Weights.Imitation:F2}");
        }

        /// <summary>
        /// Called each decision step during a lifetime. Tracks fitness signals.
        /// rootZ: current root height. reward: current centered reward.
        /// Returns true if the lifetime should end.
        /// </summary>
        public bool StepLifetime(float rootZ, float reward, float standingZ = 0.7f)
        {
            _totalStepsThisLifetime++;
            _lifetimeRewardEMA += 0.001f * (reward - _lifetimeRewardEMA);

            bool isStanding = rootZ > standingZ * 0.9f;
            if (isStanding)
            {
                _standingSteps++;
                if (!_hasStoodOnce)
                {
                    _hasStoodOnce = true;
                    _timeToFirstStand = _totalStepsThisLifetime;
                    Debug.Log($"PopulationTrainer: First stand at step {_totalStepsThisLifetime}");
                }
            }

            return _totalStepsThisLifetime >= _config.LifetimeSteps;
        }

        /// <summary>
        /// End the current lifetime, compute fitness, and advance to next member.
        /// Returns true if a full generation completed (all members evaluated).
        /// </summary>
        public bool EndLifetime()
        {
            var member = CurrentMember;
            member.LifetimeStepsCompleted = _totalStepsThisLifetime;

            float normTTFS = _hasStoodOnce
                ? 1f - Mathf.Clamp01(_timeToFirstStand / _config.LifetimeSteps)
                : 0f;
            member.TimeToFirstStand = normTTFS;

            int lastWindowSteps = Mathf.Min(_totalStepsThisLifetime, _config.LifetimeSteps / 5);
            member.StandingStability = lastWindowSteps > 0
                ? Mathf.Clamp01((float)_standingSteps / _totalStepsThisLifetime)
                : 0f;

            member.TotalRewardImprovement = Mathf.Max(0, _lifetimeRewardEMA - _lifetimeStartReward);

            member.Fitness = 0.3f * normTTFS
                           + 0.4f * member.StandingStability
                           + 0.3f * Mathf.Clamp01(member.TotalRewardImprovement);

            Debug.Log($"PopulationTrainer: Lifetime complete — " +
                $"member={_currentMemberIdx}, fitness={member.Fitness:F3} " +
                $"(ttfs={normTTFS:F2}, stability={member.StandingStability:F2}, " +
                $"reward_imp={member.TotalRewardImprovement:F3})");

            _currentMemberIdx++;

            bool generationComplete = _currentMemberIdx >= _config.PopulationSize;
            if (generationComplete)
            {
                Evolve();
                _currentMemberIdx = 0;
                _currentGeneration++;
            }

            return generationComplete;
        }

        /// <summary>
        /// Rank-based selection: top half survive, bottom half are replaced
        /// by mutated copies of the top half.
        /// </summary>
        private void Evolve()
        {
            _population.Sort((a, b) => b.Fitness.CompareTo(a.Fitness));

            int survivors = Mathf.Max(1, (int)(_config.PopulationSize * _config.SurvivalRate));
            float sigma = _config.MutationSigma;

            Debug.Log($"PopulationTrainer: Evolution — gen={_currentGeneration}, " +
                $"best fitness={_population[0].Fitness:F3}, " +
                $"worst={_population[^1].Fitness:F3}, survivors={survivors}");

            for (int i = survivors; i < _config.PopulationSize; i++)
            {
                int parentIdx = _rng.Next(survivors);
                var parent = _population[parentIdx];
                var child = parent.Clone();

                child.Weights = new RewardWeightsV2
                {
                    Height = ClampWeight(MutateFrom(parent.Weights.Height, sigma)),
                    Orientation = ClampWeight(MutateFrom(parent.Weights.Orientation, sigma)),
                    Contact = ClampWeight(MutateFrom(parent.Weights.Contact, sigma)),
                    Energy = ClampWeight(MutateFrom(parent.Weights.Energy, sigma)),
                    Imitation = ClampWeight(MutateFrom(parent.Weights.Imitation, sigma)),
                };
                child.GainRampRate = Mathf.Max(1e-7f, MutateFrom(parent.GainRampRate, sigma));
                child.EncoderLr = Mathf.Max(1e-6f, MutateFrom(parent.EncoderLr, sigma));
                child.Generation = _currentGeneration + 1;
                child.Fitness = 0;

                _population[i] = child;
            }

            for (int i = 0; i < survivors; i++)
                _population[i].Generation = _currentGeneration + 1;
        }

        private float Mutate(float value, float sigma)
        {
            float range = _config.MaxWeight - _config.MinWeight;
            return value + (float)(_rng.NextDouble() * 2 - 1) * sigma * range;
        }

        private float MutateFrom(float value, float sigma)
        {
            return value * (1f + (float)(_rng.NextDouble() * 2 - 1) * sigma);
        }

        private float ClampWeight(float v)
        {
            return Mathf.Clamp(v, _config.MinWeight, _config.MaxWeight);
        }

        // ── Persistence ─────────────────────────────────────────────────

        private const int SAVE_VERSION = 1;

        public void Save(string directory)
        {
            string path = Path.Combine(directory, "pbt_population.bin");
            try
            {
                Directory.CreateDirectory(directory);
                using var bw = new BinaryWriter(File.Create(path));
                bw.Write(SAVE_VERSION);
                bw.Write(_currentGeneration);
                bw.Write(_currentMemberIdx);
                bw.Write(_population.Count);

                foreach (var m in _population)
                {
                    bw.Write(m.Weights.Height);
                    bw.Write(m.Weights.Orientation);
                    bw.Write(m.Weights.Contact);
                    bw.Write(m.Weights.Energy);
                    bw.Write(m.Weights.Imitation);
                    bw.Write(m.GainRampRate);
                    bw.Write(m.EncoderLr);
                    bw.Write(m.Fitness);
                    bw.Write(m.TimeToFirstStand);
                    bw.Write(m.StandingStability);
                    bw.Write(m.TotalRewardImprovement);
                    bw.Write(m.Generation);
                    bw.Write(m.LifetimeStepsCompleted);
                }

                Debug.Log($"PopulationTrainer: Saved — gen={_currentGeneration}, " +
                    $"{_population.Count} members");
            }
            catch (Exception e)
            {
                Debug.LogWarning($"PopulationTrainer: Save failed — {e.Message}");
            }
        }

        public bool Load(string directory)
        {
            string path = Path.Combine(directory, "pbt_population.bin");
            if (!File.Exists(path)) return false;

            try
            {
                using var br = new BinaryReader(File.OpenRead(path));
                int ver = br.ReadInt32();
                if (ver != SAVE_VERSION)
                {
                    Debug.LogWarning("PopulationTrainer: Version mismatch, starting fresh");
                    return false;
                }

                _currentGeneration = br.ReadInt32();
                _currentMemberIdx = br.ReadInt32();
                int count = br.ReadInt32();

                _population = new List<PBTMember>(count);
                for (int i = 0; i < count; i++)
                {
                    var m = new PBTMember
                    {
                        Weights = new RewardWeightsV2
                        {
                            Height = br.ReadSingle(),
                            Orientation = br.ReadSingle(),
                            Contact = br.ReadSingle(),
                            Energy = br.ReadSingle(),
                            Imitation = br.ReadSingle(),
                        },
                        GainRampRate = br.ReadSingle(),
                        EncoderLr = br.ReadSingle(),
                        Fitness = br.ReadSingle(),
                        TimeToFirstStand = br.ReadSingle(),
                        StandingStability = br.ReadSingle(),
                        TotalRewardImprovement = br.ReadSingle(),
                        Generation = br.ReadInt32(),
                        LifetimeStepsCompleted = br.ReadInt32(),
                    };
                    _population.Add(m);
                }

                Debug.Log($"PopulationTrainer: Loaded — gen={_currentGeneration}, " +
                    $"{count} members, resuming at member {_currentMemberIdx}");
                return true;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"PopulationTrainer: Load failed — {e.Message}");
                return false;
            }
        }

        /// <summary>
        /// Summary string for display in dashboard or editor window.
        /// </summary>
        public string GetSummary()
        {
            if (_population == null || _population.Count == 0)
                return "PBT: Not initialized";

            float bestFitness = 0f;
            for (int i = 0; i < _population.Count; i++)
                bestFitness = Mathf.Max(bestFitness, _population[i].Fitness);

            return $"PBT: Gen {_currentGeneration} | Member {_currentMemberIdx}/{_config.PopulationSize} | " +
                   $"Best fitness: {bestFitness:F3} | Steps: {_totalStepsThisLifetime}/{_config.LifetimeSteps}";
        }
    }
}
