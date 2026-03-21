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
    /// Structured world model: predicts only smooth dynamics (qpos, qvel, qfrc_actuator),
    /// NOT contact or strain observations. Contact features are discontinuous and
    /// essentially unpredictable — predicting them injects noise into dream batches.
    ///
    /// Architecture:
    ///   Encoder: obs(obsDim) → z(zDim) via shared StateEncoder or independent layers
    ///   Dynamics: (z, action) → delta_z (residual: z_next = z + delta_z)
    ///   Decoder: z_next → (qpos_pred, qvel_pred, qfrc_pred) — smooth dims only
    ///   Reward head: z_next → reward_pred
    ///
    /// Dream generation zeroes contact/strain dims in predicted next_obs.
    /// The critic learns that dream batches have uncertain contacts.
    ///
    /// Obs layout (from BoneFilterConfig):
    ///   [0..smoothDim-1]                = qpos(5+N) + qvel(6+N) + qfrc(N)  (smooth)
    ///   [smoothDim..smoothDim+39]        = contact (40 floats, discontinuous)
    ///   [smoothDim+40..smoothDim+40+N-1] = strain (N floats, high-frequency)
    /// </summary>
    public class StructuredWorldModelNetwork : Module<Tensor, Tensor, (Tensor deltaZ, Tensor reward)>
    {
        private readonly Linear encFc1, encFc2;
        private readonly Linear dynFc1, dynFc2, dynHead;
        private readonly Linear rewardHead;
        private readonly Linear decFc1, decHead;
        private readonly int _zDim;

        public StructuredWorldModelNetwork(int obsDim, int actDim, int zDim = 32,
            int hidden = 256)
            : base("StructuredWorldModelNetwork")
        {
            _zDim = zDim;

            encFc1 = Linear(obsDim, hidden);
            encFc2 = Linear(hidden, zDim);

            dynFc1 = Linear(zDim + actDim, hidden);
            dynFc2 = Linear(hidden, hidden / 2);
            dynHead = Linear(hidden / 2, zDim);

            rewardHead = Linear(zDim, 1);

            decFc1 = Linear(zDim, hidden);
            decHead = Linear(hidden, obsDim);

            using (no_grad())
            {
                dynHead.weight.mul_(0.01f);
                dynHead.bias?.fill_(0f);
                rewardHead.weight.mul_(0.01f);
                rewardHead.bias?.fill_(0f);
                decHead.weight.mul_(0.01f);
                decHead.bias?.fill_(0f);
            }

            RegisterComponents();
        }

        public Tensor Encode(Tensor obs)
        {
            var h = functional.relu(encFc1.forward(obs));
            return encFc2.forward(h);
        }

        public Tensor Decode(Tensor z)
        {
            var h = functional.relu(decFc1.forward(z));
            return decHead.forward(h);
        }

        public override (Tensor deltaZ, Tensor reward) forward(Tensor obs, Tensor action)
        {
            var z = Encode(obs);
            var catZA = torch.cat(new[] { z, action }, dim: 1);
            var h = functional.relu(dynFc1.forward(catZA));
            h = functional.relu(dynFc2.forward(h));
            var deltaZ = dynHead.forward(h);
            var zNext = z + deltaZ;
            var reward = rewardHead.forward(zNext);
            return (deltaZ, reward);
        }

        /// <summary>
        /// Full forward: encode → dynamics → decode. Returns predicted next_obs and reward.
        /// </summary>
        public (Tensor nextObsPred, Tensor reward) ForwardFull(Tensor obs, Tensor action)
        {
            var z = Encode(obs);
            var catZA = torch.cat(new[] { z, action }, dim: 1);
            var h = functional.relu(dynFc1.forward(catZA));
            h = functional.relu(dynFc2.forward(h));
            var deltaZ = dynHead.forward(h);
            var zNext = z + deltaZ;
            var nextObsPred = Decode(zNext);
            var reward = rewardHead.forward(zNext);
            return (nextObsPred, reward);
        }
    }

    /// <summary>
    /// Manages structured world model training and dream batch generation.
    /// Only trains on smooth observation dimensions; zeroes contacts/strain in dreams.
    /// </summary>
    public class StructuredWorldModel : IDisposable
    {
        private readonly StructuredWorldModelNetwork _net;
        private readonly optim.Optimizer _optimizer;
        private readonly Device _device;
        private readonly int _obsDim;
        private readonly int _actDim;
        private readonly int _smoothDim;
        private readonly int _contactStart;
        private readonly int _contactDim;
        private readonly int _strainDim;

        private float _lastLoss;
        private float _lastDynLoss;
        private float _lastRewardLoss;
        private int _trainSteps;

        public float LastLoss => _lastLoss;
        public float LastDynLoss => _lastDynLoss;
        public float LastRewardLoss => _lastRewardLoss;
        public int TrainSteps => _trainSteps;

        /// <param name="smoothDim">Number of smooth obs dims (qpos + qvel + qfrc_actuator).</param>
        /// <param name="contactDim">Number of contact obs dims (typically 40).</param>
        /// <param name="strainDim">Number of strain obs dims.</param>
        public StructuredWorldModel(int obsDim, int actDim, int smoothDim,
            int contactDim, int strainDim, float lr, Device device,
            int zDim = 32, int hidden = 256)
        {
            _obsDim = obsDim;
            _actDim = actDim;
            _smoothDim = smoothDim;
            _contactStart = smoothDim;
            _contactDim = contactDim;
            _strainDim = strainDim;
            _device = device;

            _net = new StructuredWorldModelNetwork(obsDim, actDim, zDim, hidden);
            _net.to(device);
            _optimizer = optim.Adam(_net.parameters(), lr: lr);
        }

        /// <summary>
        /// Train on a batch of real transitions. Loss only on smooth dims + reward.
        /// Contact and strain prediction errors are excluded from training loss.
        /// </summary>
        public float TrainStep(Batch batch)
        {
            using var scope = NewDisposeScope();

            var obs = torch.tensor(batch.Obs).reshape(batch.Size, batch.ObsDim).to(_device);
            var actions = torch.tensor(batch.Actions).reshape(batch.Size, batch.ActDim).to(_device);
            var nextObs = torch.tensor(batch.NextObs).reshape(batch.Size, batch.ObsDim).to(_device);
            var rewards = torch.tensor(batch.Rewards).reshape(batch.Size, 1).to(_device);

            var (nextObsPred, predReward) = _net.ForwardFull(obs, actions);

            // Only compute loss on smooth dims (qpos + qvel + qfrc_actuator)
            var nextObsSmooth = nextObs.slice(1, 0, _smoothDim, 1);
            var predObsSmooth = nextObsPred.slice(1, 0, _smoothDim, 1);

            var dynLoss = functional.mse_loss(predObsSmooth, nextObsSmooth);
            var rewardLoss = functional.mse_loss(predReward, rewards);
            var loss = dynLoss + rewardLoss;

            _optimizer.zero_grad();
            loss.backward();
            torch.nn.utils.clip_grad_norm_(_net.parameters(), 1.0);
            _optimizer.step();

            _lastLoss = loss.item<float>();
            _lastDynLoss = dynLoss.item<float>();
            _lastRewardLoss = rewardLoss.item<float>();
            _trainSteps++;
            return _lastLoss;
        }

        /// <summary>
        /// Generate a dream batch. Contact and strain dims are zeroed in predicted next_obs.
        /// </summary>
        public void GenerateDreamBatch(Batch realBatch, SACActorNetwork actor, Batch dreamBatch)
        {
            using var scope = NewDisposeScope();
            using (no_grad())
            {
                int dreamSize = dreamBatch.Size;
                int realSize = realBatch.Size;

                var obsT = torch.tensor(realBatch.Obs)
                    .reshape(realSize, realBatch.ObsDim).to(_device);

                Tensor startObs;
                if (dreamSize <= realSize)
                    startObs = obsT.slice(0, 0, dreamSize, 1);
                else
                {
                    var indices = torch.randint(realSize, new long[] { dreamSize },
                        dtype: ScalarType.Int64, device: _device);
                    startObs = obsT.index_select(0, indices);
                }

                var (dreamActions, _, _) = actor.forward(startObs);
                var (predNextObs, predReward) = _net.ForwardFull(startObs, dreamActions);

                // Zero out contact and strain dims — they're unreliable predictions.
                // The critic learns that dream transitions have no contact info.
                if (_contactStart < _obsDim)
                {
                    var contactSlice = predNextObs.slice(1, _contactStart, _obsDim, 1);
                    contactSlice.fill_(0f);
                }

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
            _net.save(Path.Combine(directory, "structured_world_model.pt"));
            var statePath = Path.Combine(directory, "structured_wm_state.bin");
            using var bw = new BinaryWriter(File.Create(statePath));
            bw.Write(_trainSteps);
            bw.Write(_lastLoss);
            bw.Write(_lastDynLoss);
            bw.Write(_lastRewardLoss);
        }

        public bool Load(string directory)
        {
            var modelPath = Path.Combine(directory, "structured_world_model.pt");
            if (!File.Exists(modelPath)) return false;
            try
            {
                _net.load(modelPath);
                var statePath = Path.Combine(directory, "structured_wm_state.bin");
                if (File.Exists(statePath))
                {
                    using var br = new BinaryReader(File.OpenRead(statePath));
                    _trainSteps = br.ReadInt32();
                    _lastLoss = br.ReadSingle();
                    _lastDynLoss = br.ReadSingle();
                    _lastRewardLoss = br.ReadSingle();
                }
                return true;
            }
            catch (Exception e)
            {
                UnityEngine.Debug.LogWarning(
                    $"StructuredWorldModel: Load failed — {e.Message}. Starting fresh.");
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
