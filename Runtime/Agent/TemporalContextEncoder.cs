using System;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using Tensor = TorchSharp.torch.Tensor;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Conv1D temporal context encoder inspired by HumanUP's StateHistoryEncoder.
    /// Takes a sequence of recent (obs, action, reward) tuples and produces a
    /// fixed-size context vector prepended to observations for actor and critic.
    ///
    /// Architecture:
    ///   per-timestep projection -> Conv1D stack -> flatten -> output projection
    ///
    /// Input layout: (seqLen, batch, inputDim) — same as the old transformer encoder.
    /// ~5K params vs ~320K for the transformer (60x reduction).
    /// </summary>
    public class TemporalContextEncoder : Module<Tensor, Tensor, Tensor>
    {
        private readonly Linear _inputProj;
        private readonly Conv1d _conv1;
        private readonly Conv1d _conv2;
        private readonly Linear _outputProj;
        private readonly int _seqLen;
        private readonly int _contextDim;
        private readonly int _flatDim;

        public int ContextDim => _contextDim;

        /// <param name="inputDim">Per-timestep input size (obsDim + actDim + 1 for reward)</param>
        /// <param name="contextDim">Output context vector dimension</param>
        /// <param name="seqLen">Maximum sequence length (history window)</param>
        /// <param name="channelSize">Base channel width (actual channels are 3x, 2x, 1x)</param>
        public TemporalContextEncoder(
            int inputDim, int contextDim, int seqLen, int channelSize = 10)
            : base("TemporalContextEncoder")
        {
            _seqLen = seqLen;
            _contextDim = contextDim;

            int ch3 = 3 * channelSize; // 30
            int ch2 = 2 * channelSize; // 20
            int ch1 = channelSize;     // 10

            _inputProj = Linear(inputDim, ch3);

            // Conv1D kernel/stride chosen for seqLen=16 target:
            //   Conv1(30->20, k=4, s=2): 16 -> 7
            //   Conv2(20->10, k=2, s=1): 7 -> 6
            // For other seqLen values the output length adapts automatically.
            _conv1 = Conv1d(ch3, ch2, kernelSize: 4, stride: 2);
            _conv2 = Conv1d(ch2, ch1, kernelSize: 2, stride: 1);

            // Compute flattened dimension after conv stack
            int afterConv1 = (seqLen - 4) / 2 + 1;
            int afterConv2 = afterConv1 - 2 + 1;
            _flatDim = ch1 * Math.Max(1, afterConv2);

            _outputProj = Linear(_flatDim, contextDim);

            RegisterComponents();
        }

        /// <summary>
        /// Encode a history sequence into a context vector.
        /// </summary>
        /// <param name="sequence">Shape: (seqLen, batch, inputDim)</param>
        /// <param name="paddingMask">Shape: (batch, seqLen) — ignored by Conv1D but kept for interface compat.
        /// Only the all-padded safety check is used.</param>
        /// <returns>Context vector, shape: (batch, contextDim)</returns>
        public override Tensor forward(Tensor sequence, Tensor paddingMask)
        {
            if (paddingMask is not null && paddingMask.numel() > 0)
            {
                bool allPadded;
                using (no_grad())
                {
                    long numTrue = paddingMask.sum().to_type(ScalarType.Int64).item<long>();
                    allPadded = numTrue == paddingMask.numel();
                }
                if (allPadded)
                {
                    int batch = (int)sequence.shape[1];
                    return torch.zeros(batch, _contextDim, dtype: sequence.dtype,
                        device: sequence.device);
                }
            }

            // sequence: (seqLen, batch, inputDim) -> (batch, seqLen, inputDim)
            var x = sequence.permute(1, 0, 2);
            int batchSize = (int)x.shape[0];
            int actualSeq = (int)x.shape[1];

            // Per-timestep projection: (batch * seqLen, inputDim) -> (batch * seqLen, ch3)
            var flat = x.reshape(-1, x.shape[2]);
            var projected = functional.relu(_inputProj.forward(flat));
            projected = projected.reshape(batchSize, actualSeq, -1);

            // Conv1D expects (batch, channels, length)
            var conv_in = projected.permute(0, 2, 1);
            var h = functional.relu(_conv1.forward(conv_in));
            h = functional.relu(_conv2.forward(h));

            // Flatten: (batch, ch1, reducedLen) -> (batch, flatDim)
            h = h.flatten(1);

            return functional.relu(_outputProj.forward(h));
        }
    }
}
