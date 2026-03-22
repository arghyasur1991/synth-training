using System;
using System.Runtime.CompilerServices;
using TorchSharp;
using Tensor = TorchSharp.torch.Tensor;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Fixed-capacity ring buffer for inference-time history. Stores the last
    /// N (obs, action, reward) tuples and packs them into a preallocated tensor
    /// for the TemporalContextEncoder.
    ///
    /// Pure C#, zero-allocation after construction. Used only on the game thread.
    /// </summary>
    public class HistoryRingBuffer
    {
        private readonly int _seqLen;
        private readonly int _obsDim;
        private readonly int _actDim;
        private readonly int _entryDim; // obsDim + actDim + 1

        private readonly float[] _data; // seqLen * entryDim, circular
        private readonly float[] _packBuf; // pre-allocated for PackIntoTensor
        private readonly float[] _maskBuf; // pre-allocated for WritePaddingMask
        private int _head;
        private int _count;

        public int SeqLen => _seqLen;
        public int EntryDim => _entryDim;
        public int Count => _count;

        public HistoryRingBuffer(int seqLen, int obsDim, int actDim)
        {
            _seqLen = seqLen;
            _obsDim = obsDim;
            _actDim = actDim;
            _entryDim = obsDim + actDim + 1;
            _data = new float[seqLen * _entryDim];
            _packBuf = new float[seqLen * _entryDim];
            _maskBuf = new float[seqLen];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Push(float[] obs, float[] action, float reward)
        {
            int offset = _head * _entryDim;
            Buffer.BlockCopy(obs, 0, _data, offset * sizeof(float), _obsDim * sizeof(float));
            Buffer.BlockCopy(action, 0, _data, (offset + _obsDim) * sizeof(float), _actDim * sizeof(float));
            _data[offset + _obsDim + _actDim] = reward;

            _head = (_head + 1) % _seqLen;
            if (_count < _seqLen) _count++;
        }

        /// <summary>
        /// Pack buffer contents into a preallocated tensor in chronological order
        /// (oldest first). Entries beyond _count are zero-padded.
        /// Tensor shape must be (seqLen, 1, entryDim) — seq-first for TorchSharp.
        /// </summary>
        public void PackIntoTensor(Tensor output)
        {
            int padCount = _seqLen - _count;
            if (padCount > 0)
                Array.Clear(_packBuf, 0, padCount * _entryDim);

            if (_count > 0)
            {
                int oldest = (_head - _count + _seqLen) % _seqLen;
                for (int i = 0; i < _count; i++)
                {
                    int srcIdx = (oldest + i) % _seqLen;
                    Buffer.BlockCopy(_data, srcIdx * _entryDim * sizeof(float),
                        _packBuf, (padCount + i) * _entryDim * sizeof(float),
                        _entryDim * sizeof(float));
                }
            }

            output.bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes<float>(_packBuf.AsSpan());
        }

        /// <summary>
        /// Write padding mask into a preallocated tensor. Shape: (1, seqLen).
        /// True (1) where padded/invalid, false (0) where valid data exists.
        /// Uses float tensor since TorchSharp bool tensor interop is limited.
        /// </summary>
        public void WritePaddingMask(Tensor mask)
        {
            int padCount = _seqLen - _count;
            for (int i = 0; i < padCount; i++)
                _maskBuf[i] = 1f;
            for (int i = padCount; i < _seqLen; i++)
                _maskBuf[i] = 0f;

            mask.bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes<float>(_maskBuf.AsSpan());
        }

        public void Reset()
        {
            _head = 0;
            _count = 0;
            Array.Clear(_data, 0, _data.Length);
        }
    }
}
