using System;
using System.Collections.Generic;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Tensor = TorchSharp.torch.Tensor;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Structured actor network with kinematic-chain action heads.
    ///
    /// Instead of a flat MLP producing all ~90 actions, a shared encoder
    /// feeds separate per-chain heads. Each head is a small MLP responsible
    /// for its kinematic chain (spine, left leg, right leg, left arm, etc.).
    ///
    /// This provides inductive bias matching the body's structure:
    ///   - Legs share the same head architecture but have independent weights
    ///   - The shared encoder captures body-wide coordination
    ///   - Each head only needs to coordinate a small set of joints
    ///
    /// Chain layout (configurable via chainDims):
    ///   [0] Spine/Trunk  — hips, spine, chest, upper chest (~12 actions)
    ///   [1] Left leg      — upper leg, lower leg, foot, toes (~12 actions)
    ///   [2] Right leg     — upper leg, lower leg, foot, toes (~12 actions)
    ///   [3] Left arm      — shoulder, upper arm, lower arm, hand (~12 actions)
    ///   [4] Right arm     — shoulder, upper arm, lower arm, hand (~12 actions)
    ///   [5] Head/Neck     — neck, head, jaw, eyes (~6+ actions)
    ///   [6] Auxiliary      — pectorals, gluteals, etc. (remainder)
    /// </summary>
    public class StructuredActorNetwork : Module<Tensor, (Tensor action, Tensor logProb, Tensor mean)>
    {
        private const float LOG_STD_MIN = -5f;
        private const float LOG_STD_MAX = 2f;

        private readonly Linear _enc1, _enc2;
        private readonly List<(Linear fc, Linear mean, Linear logStd)> _heads;
        private readonly int[] _chainDims;
        private readonly int _totalActDim;
        private readonly float _actionScale;
        private readonly float _actionBias;

        /// <summary>
        /// Create a structured actor.
        /// chainDims: action dimension per kinematic chain head. Sum must equal actDim.
        /// </summary>
        public StructuredActorNetwork(
            int obsDim, int actDim, int[] chainDims,
            int encoderHidden = 512, int headHidden = 128,
            float actionScale = 1f, float actionBias = 0f)
            : base("StructuredActorNetwork")
        {
            _actionScale = actionScale;
            _actionBias = actionBias;
            _chainDims = chainDims;
            _totalActDim = actDim;

            int chainSum = 0;
            foreach (int d in chainDims) chainSum += d;
            if (chainSum != actDim)
                throw new ArgumentException(
                    $"StructuredActor: chain dims sum ({chainSum}) != actDim ({actDim})");

            _enc1 = Linear(obsDim, encoderHidden);
            _enc2 = Linear(encoderHidden, encoderHidden);

            _heads = new List<(Linear, Linear, Linear)>();
            for (int i = 0; i < chainDims.Length; i++)
            {
                int cd = chainDims[i];
                var fc = Linear(encoderHidden, headHidden);
                var mean = Linear(headHidden, cd);
                var logStd = Linear(headHidden, cd);
                _heads.Add((fc, mean, logStd));
            }

            RegisterComponents();
        }

        public override (Tensor action, Tensor logProb, Tensor mean) forward(Tensor obs)
        {
            var h = functional.relu(_enc1.forward(obs));
            h = functional.relu(_enc2.forward(h));

            var meanParts = new Tensor[_heads.Count];
            var logStdParts = new Tensor[_heads.Count];

            for (int i = 0; i < _heads.Count; i++)
            {
                var (fc, meanLayer, logStdLayer) = _heads[i];
                var headOut = functional.relu(fc.forward(h));
                meanParts[i] = meanLayer.forward(headOut);
                logStdParts[i] = logStdLayer.forward(headOut).clamp(LOG_STD_MIN, LOG_STD_MAX);
            }

            var fullMean = torch.cat(meanParts, dim: -1);
            var fullLogStd = torch.cat(logStdParts, dim: -1);
            var std = fullLogStd.exp();

            var noise = torch.randn_like(fullMean);
            var xT = fullMean + std * noise;
            var action = torch.tanh(xT) * _actionScale + _actionBias;

            var logProbRaw = -0.5f * (((xT - fullMean) / std).pow(2) + 2f * fullLogStd
                            + (float)Math.Log(2.0 * Math.PI));
            var logProb = logProbRaw.sum(-1);
            logProb = logProb - (2f * ((float)Math.Log(2.0) - xT
                     - functional.softplus(-2f * xT))).sum(1);

            var meanAction = torch.tanh(fullMean) * _actionScale + _actionBias;

            return (action, logProb, meanAction);
        }
    }
}
