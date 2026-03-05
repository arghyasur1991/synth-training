using System;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Algorithm-agnostic interface for training backends used by BaseTrainingSkill.
    /// Implementations wrap a specific RL algorithm (SAC, PPO, etc.) and its
    /// associated data structures (replay buffer, rollout buffer, etc.).
    /// </summary>
    public interface ISkillTrainer : IDisposable
    {
        void Initialize(int obsDim, int actDim, TorchSharp.torch.Device device);

        /// <summary>Sample a stochastic action from the current policy.</summary>
        float[] GetAction(float[] obs);

        /// <summary>Sample a random action for warmup / exploration.</summary>
        float[] GetRandomAction(Random rng);

        /// <summary>Store one transition. The trainer decides how to buffer it.</summary>
        void StoreTransition(float[] obs, float[] action, float reward,
                             float[] nextObs, bool done);

        void StartTraining();
        void StopTraining();
        void PauseTraining();
        void ResumeTraining();

        void Save(string directory);
        bool Load(string directory);

        bool IsTraining { get; }
        long TotalTrainSteps { get; }
        float LastTrainStepMs { get; }
        int StepsPerSecond { get; }
        int ExperienceCount { get; }
    }
}
