using System;
using System.IO;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Tensor = TorchSharp.torch.Tensor;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Dyna-style world model: predicts (next_obs, reward) from (obs, action).
    /// Two-headed MLP trained on real transitions from the replay buffer.
    /// Used to generate imagined transitions for critic-only SAC training.
    /// </summary>
    public class WorldModelNetwork : Module<Tensor, Tensor, (Tensor nextObs, Tensor reward)>
    {
        private readonly Linear fc1, fc2, headObs, headReward;

        public WorldModelNetwork(int obsDim, int actDim,
            int hidden1 = 256, int hidden2 = 128)
            : base("WorldModelNetwork")
        {
            fc1 = Linear(obsDim + actDim, hidden1);
            fc2 = Linear(hidden1, hidden2);
            headObs = Linear(hidden2, obsDim);
            headReward = Linear(hidden2, 1);

            using (no_grad())
            {
                headObs.weight.mul_(0.01f);
                headObs.bias?.fill_(0f);
                headReward.weight.mul_(0.01f);
                headReward.bias?.fill_(0f);
            }

            RegisterComponents();
        }

        public override (Tensor nextObs, Tensor reward) forward(Tensor obs, Tensor action)
        {
            var cat = torch.cat(new[] { obs, action }, dim: 1);
            var x = functional.relu(fc1.forward(cat));
            x = functional.relu(fc2.forward(x));
            return (headObs.forward(x), headReward.forward(x));
        }
    }

    /// <summary>
    /// Manages world model training and dream batch generation.
    /// All operations run on the training thread — no cross-thread concerns.
    /// </summary>
    public class WorldModel : IDisposable
    {
        private readonly WorldModelNetwork _net;
        private readonly optim.Optimizer _optimizer;
        private readonly Device _device;
        private readonly int _obsDim;
        private readonly int _actDim;

        private float _lastLoss;
        private int _trainSteps;

        public float LastLoss => _lastLoss;
        public int TrainSteps => _trainSteps;

        public WorldModel(int obsDim, int actDim, float lr, Device device,
            int hidden1 = 256, int hidden2 = 128)
        {
            _obsDim = obsDim;
            _actDim = actDim;
            _device = device;

            _net = new WorldModelNetwork(obsDim, actDim, hidden1, hidden2);
            _net.to(device);
            _optimizer = optim.Adam(_net.parameters(), lr: lr);
        }

        /// <summary>
        /// Train on a batch of real transitions. Returns combined MSE loss.
        /// </summary>
        public float TrainStep(Batch batch)
        {
            using var scope = NewDisposeScope();

            var obs = torch.tensor(batch.Obs).reshape(batch.Size, batch.ObsDim).to(_device);
            var actions = torch.tensor(batch.Actions).reshape(batch.Size, batch.ActDim).to(_device);
            var nextObs = torch.tensor(batch.NextObs).reshape(batch.Size, batch.ObsDim).to(_device);
            var rewards = torch.tensor(batch.Rewards).reshape(batch.Size, 1).to(_device);

            var (predNextObs, predReward) = _net.forward(obs, actions);

            var obsLoss = functional.mse_loss(predNextObs, nextObs);
            var rewardLoss = functional.mse_loss(predReward, rewards);
            var loss = obsLoss + rewardLoss;

            _optimizer.zero_grad();
            loss.backward();
            torch.nn.utils.clip_grad_norm_(_net.parameters(), 1.0);
            _optimizer.step();

            _lastLoss = loss.item<float>();
            _trainSteps++;
            return _lastLoss;
        }

        /// <summary>
        /// Generate a dream batch by taking start states from the real batch,
        /// picking new actions via the actor, and predicting next_obs + reward.
        /// All imagined transitions are non-terminal (done=0).
        /// </summary>
        public void GenerateDreamBatch(Batch realBatch, SACActorNetwork actor,
            Batch dreamBatch)
        {
            using var scope = NewDisposeScope();
            using (no_grad())
            {
                int dreamSize = dreamBatch.Size;
                int realSize = realBatch.Size;

                var obsT = torch.tensor(realBatch.Obs)
                    .reshape(realSize, realBatch.ObsDim).to(_device);

                // If dream batch is smaller than real batch, take the first N rows.
                // If larger, wrap around.
                Tensor startObs;
                if (dreamSize <= realSize)
                {
                    startObs = obsT.slice(0, 0, dreamSize, 1);
                }
                else
                {
                    var indices = torch.randint(realSize, new long[] { dreamSize },
                        dtype: ScalarType.Int64, device: _device);
                    startObs = obsT.index_select(0, indices);
                }

                var (dreamActions, _, _) = actor.forward(startObs);
                var (predNextObs, predReward) = _net.forward(startObs, dreamActions);

                var startObsCpu = startObs.cpu();
                var actionsCpu = dreamActions.cpu();
                var nextObsCpu = predNextObs.cpu();
                var rewardsCpu = predReward.cpu();

                CopyToFlat(startObsCpu, dreamBatch.Obs, dreamSize, _obsDim);
                CopyToFlat(actionsCpu, dreamBatch.Actions, dreamSize, _actDim);
                CopyToFlat(nextObsCpu, dreamBatch.NextObs, dreamSize, _obsDim);

                var rewardData = rewardsCpu.data<float>();
                for (int i = 0; i < dreamSize; i++)
                {
                    dreamBatch.Rewards[i] = rewardData[i];
                    dreamBatch.Dones[i] = 0f;
                    dreamBatch.ISWeights[i] = 1f;
                }
            }
        }

        private static void CopyToFlat(Tensor src, float[] dst, int rows, int cols)
        {
            var data = src.data<float>();
            for (int i = 0; i < rows * cols; i++)
                dst[i] = data[i];
        }

        public void Save(string directory)
        {
            _net.save(Path.Combine(directory, "world_model.pt"));
            var statePath = Path.Combine(directory, "world_model_state.bin");
            using var bw = new BinaryWriter(File.Create(statePath));
            bw.Write(_trainSteps);
            bw.Write(_lastLoss);
        }

        public bool Load(string directory)
        {
            var modelPath = Path.Combine(directory, "world_model.pt");
            if (!File.Exists(modelPath))
                return false;

            try
            {
                _net.load(modelPath);
                var statePath = Path.Combine(directory, "world_model_state.bin");
                if (File.Exists(statePath))
                {
                    using var br = new BinaryReader(File.OpenRead(statePath));
                    _trainSteps = br.ReadInt32();
                    _lastLoss = br.ReadSingle();
                }
                return true;
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogWarning(
                    $"WorldModel: Load failed — {e.Message}. Starting fresh.");
                return false;
            }
        }

        public void Dispose()
        {
            _net?.Dispose();
            _optimizer?.Dispose();
        }
    }
}
