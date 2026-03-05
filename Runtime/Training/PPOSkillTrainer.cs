using System;
using System.Threading;
using Debug = UnityEngine.Debug;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// ISkillTrainer wrapping PPOAgent + RolloutBuffer.
    /// Collects transitions sequentially; when the rollout is full, computes GAE
    /// and runs K epochs of mini-batch PPO updates.
    /// </summary>
    public class PPOSkillTrainer : BaseSkillTrainer
    {
        private PPOAgent _agent;
        private RolloutBuffer _rollout;
        private readonly PPOConfig _config;

        private float _lastLogProb;
        private float _lastValue;
        private float[] _lastObs;
        private long _totalIterations;

        // Mini-batch pre-allocated arrays
        private float[] _mbObs, _mbActions, _mbLogProbs, _mbAdvantages, _mbReturns, _mbValues;
        private int[] _batchIndices;

        public PPOAgent Agent => _agent;
        public RolloutBuffer Rollout => _rollout;
        public override int ExperienceCount => _rollout?.Count ?? 0;
        public float LastPolicyLoss { get; private set; }
        public float LastValueLoss { get; private set; }
        public float LastEntropy { get; private set; }
        public float LastApproxKL { get; private set; }
        public float LastClipFrac { get; private set; }

        public PPOSkillTrainer(PPOConfig config, bool isMobile = false)
            : base(isMobile)
        {
            _config = config;
        }

        public override void Initialize(int obsDim, int actDim, TorchSharp.torch.Device device)
        {
            _agent = new PPOAgent(obsDim, actDim, _config, device);
            _rollout = new RolloutBuffer(_config.NumSteps, obsDim, actDim);

            _lastObs = new float[obsDim];

            int mbSize = _config.MiniBatchSize;
            _mbObs = new float[mbSize * obsDim];
            _mbActions = new float[mbSize * actDim];
            _mbLogProbs = new float[mbSize];
            _mbAdvantages = new float[mbSize];
            _mbReturns = new float[mbSize];
            _mbValues = new float[mbSize];
            _batchIndices = new int[_config.NumSteps];
            for (int i = 0; i < _config.NumSteps; i++)
                _batchIndices[i] = i;

            Debug.Log($"PPOSkillTrainer: Initialized (obs={obsDim}, act={actDim}, " +
                $"device={device}, rollout={_config.NumSteps})");
        }

        public override float[] GetAction(float[] obs)
        {
            var (action, logProb, value) = _agent.GetActionAndValue(obs);
            _lastLogProb = logProb;
            _lastValue = value;
            Array.Copy(obs, _lastObs, obs.Length);
            return action;
        }

        public override float[] GetRandomAction(Random rng)
        {
            var buf = new float[_agent.ActDim];
            for (int i = 0; i < buf.Length; i++)
                buf[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            _lastLogProb = 0f;
            _lastValue = 0f;
            return buf;
        }

        public override void StoreTransition(float[] obs, float[] action, float reward,
                                             float[] nextObs, bool done)
        {
            _rollout.Add(obs, action, reward, done ? 1f : 0f, _lastLogProb, _lastValue);
        }

        protected override bool ReadyToTrain() => _rollout.IsFull;

        protected override void DoTrainStep()
        {
            using var scope = TorchSharp.torch.NewDisposeScope();

            float lastDone;
            _rollout.GetLast(_lastObs, out lastDone);
            float lastValue;
            using (TorchSharp.torch.no_grad())
            {
                var obsTensor = TorchSharp.torch.tensor(_lastObs)
                    .reshape(1, _agent.ObsDim).to(_agent.Device);
                lastValue = _agent.GetValue(obsTensor).item<float>();
            }

            _rollout.ComputeGAE(lastValue, lastDone, _config.Gamma, _config.GAELambda);

            if (_config.AnnealLR && _totalIterations > 0)
            {
                float frac = 1.0f - Math.Min(_totalIterations / 10000f, 0.98f);
                float lr = Math.Max(_config.LearningRate * frac, _config.MinLearningRate);
                _agent.SetLearningRate(lr);
            }

            int numSteps = _config.NumSteps;
            int mbSize = _config.MiniBatchSize;
            bool earlyStop = false;

            for (int epoch = 0; epoch < _config.NumEpochs; epoch++)
            {
                if (earlyStop) break;
                Shuffle(_batchIndices, numSteps);

                for (int start = 0; start < numSteps; start += mbSize)
                {
                    int end = Math.Min(start + mbSize, numSteps);
                    int actualMb = end - start;

                    _rollout.GetMiniBatch(_batchIndices, actualMb,
                        _mbObs, _mbActions, _mbLogProbs,
                        _mbAdvantages, _mbReturns, _mbValues);

                    var dev = _agent.Device;
                    var obsT = TorchSharp.torch.tensor(_mbObs[..(actualMb * _agent.ObsDim)])
                        .reshape(actualMb, _agent.ObsDim).to(dev);
                    var actT = TorchSharp.torch.tensor(_mbActions[..(actualMb * _agent.ActDim)])
                        .reshape(actualMb, _agent.ActDim).to(dev);
                    var logPT = TorchSharp.torch.tensor(_mbLogProbs[..actualMb]).to(dev);
                    var advT = TorchSharp.torch.tensor(_mbAdvantages[..actualMb]).to(dev);
                    var retT = TorchSharp.torch.tensor(_mbReturns[..actualMb]).to(dev);
                    var valT = TorchSharp.torch.tensor(_mbValues[..actualMb]).to(dev);

                    var (pg, vl, ent, kl, cf) = _agent.UpdateStep(
                        obsT, actT, logPT, advT, retT, valT, _config);

                    if (float.IsNaN(pg) || float.IsNaN(vl) || float.IsInfinity(pg))
                    {
                        Debug.LogWarning("PPOSkillTrainer: NaN/Inf detected in loss, skipping update round.");
                        earlyStop = true;
                        break;
                    }

                    LastPolicyLoss = pg;
                    LastValueLoss = vl;
                    LastEntropy = ent;
                    LastApproxKL = kl;
                    LastClipFrac = cf;

                    Interlocked.Increment(ref _totalSteps);

                    if (_config.TargetKL > 0 && kl > _config.TargetKL * 1.5f)
                    {
                        earlyStop = true;
                        break;
                    }
                }
            }

            _agent.SyncInferenceWeights();
            _rollout.Clear();
            _totalIterations++;
        }

        [ThreadStatic] private static Random _shuffleRng;

        private static void Shuffle(int[] array, int count)
        {
            if (_shuffleRng == null)
                _shuffleRng = new Random(Environment.TickCount ^
                    Thread.CurrentThread.ManagedThreadId);

            for (int i = count - 1; i > 0; i--)
            {
                int j = _shuffleRng.Next(0, i + 1);
                (array[i], array[j]) = (array[j], array[i]);
            }
        }

        public override void Save(string directory)
        {
            _agent?.Save(directory);
        }

        public override bool Load(string directory)
        {
            try
            {
                _agent?.Load(directory);
                return true;
            }
            catch (Exception e)
            {
                Debug.LogWarning($"PPOSkillTrainer: Load failed — {e.Message}");
                return false;
            }
        }

        public override void Dispose()
        {
            base.Dispose();
            _agent?.Dispose();
        }
    }
}
