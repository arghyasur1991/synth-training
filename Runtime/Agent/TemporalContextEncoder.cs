using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Tensor = TorchSharp.torch.Tensor;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Transformer-based temporal context encoder. Takes a sequence of recent
    /// (obs, action, reward) tuples and produces a fixed-size context vector
    /// that is prepended to observations for both actor and critic.
    ///
    /// Architecture:
    ///   input projection → positional encoding → TransformerEncoder → last token → output projection
    ///
    /// TorchSharp's MultiheadAttention is NOT batch_first, so all sequences
    /// use (seqLen, batch, features) layout throughout.
    /// </summary>
    public class TemporalContextEncoder : Module<Tensor, Tensor, Tensor>
    {
        private readonly Linear _inputProj;
        private readonly Linear _outputProj;
        private readonly TransformerEncoder _encoder;
        private readonly Parameter _posEncoding;
        private readonly Tensor _causalMask;
        private readonly LayerNorm _inputNorm;
        private readonly int _seqLen;
        private readonly int _contextDim;

        public int ContextDim => _contextDim;

        /// <param name="inputDim">Per-timestep input size (obsDim + actDim + 1 for reward)</param>
        /// <param name="contextDim">Output context vector dimension</param>
        /// <param name="seqLen">Maximum sequence length (history window)</param>
        /// <param name="dModel">Transformer hidden dimension</param>
        /// <param name="nHeads">Number of attention heads</param>
        /// <param name="nLayers">Number of TransformerEncoderLayer stacks</param>
        /// <param name="dimFeedforward">Feedforward hidden size per layer</param>
        /// <param name="dropout">Attention dropout rate</param>
        public TemporalContextEncoder(
            int inputDim, int contextDim, int seqLen,
            int dModel = 128, int nHeads = 4, int nLayers = 2,
            int dimFeedforward = 256, double dropout = 0.0)
            : base("TemporalContextEncoder")
        {
            _seqLen = seqLen;
            _contextDim = contextDim;

            _inputProj = Linear(inputDim, dModel);
            _inputNorm = LayerNorm(dModel);

            var layer = TransformerEncoderLayer(
                d_model: dModel,
                nhead: nHeads,
                dim_feedforward: dimFeedforward,
                dropout: dropout,
                activation: Activations.ReLU);
            _encoder = TransformerEncoder(layer, num_layers: nLayers);

            _outputProj = Linear(dModel, contextDim);

            _posEncoding = Parameter(torch.randn(seqLen, 1, dModel) * 0.02f);

            _causalMask = BuildCausalMask(seqLen);

            RegisterComponents();
        }

        private static Tensor BuildCausalMask(int size)
        {
            var mask = torch.full(size, size, float.NegativeInfinity);
            for (int i = 0; i < size; i++)
                for (int j = 0; j <= i; j++)
                    mask[i, j] = 0f;
            return mask;
        }

        /// <summary>
        /// Encode a history sequence into a context vector.
        /// </summary>
        /// <param name="sequence">Shape: (seqLen, batch, inputDim)</param>
        /// <param name="paddingMask">Shape: (batch, seqLen), true where invalid/padded</param>
        /// <returns>Context vector, shape: (batch, contextDim)</returns>
        public override Tensor forward(Tensor sequence, Tensor paddingMask)
        {
            var projected = _inputProj.forward(sequence);

            int actualSeq = (int)projected.shape[0];
            if (actualSeq <= _seqLen)
                projected = projected + _posEncoding.narrow(0, 0, actualSeq);

            projected = _inputNorm.forward(projected);

            var mask = _causalMask;
            if (mask.device != projected.device)
                mask = mask.to(projected.device);
            if (actualSeq < _seqLen)
                mask = mask.narrow(0, 0, actualSeq).narrow(1, 0, actualSeq);

            Tensor padMask = null;
            if (paddingMask is not null && paddingMask.numel() > 0)
            {
                padMask = paddingMask.to(projected.device);
            }

            var encoded = _encoder.forward(projected, mask, padMask);

            var lastToken = encoded[actualSeq - 1];

            return _outputProj.forward(lastToken);
        }
    }
}
