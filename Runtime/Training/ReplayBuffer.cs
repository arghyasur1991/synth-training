using System;
using System.IO;
using System.Runtime.CompilerServices;
using System.Threading;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Thread-safe circular replay buffer with Prioritized Experience Replay (PER).
    ///
    /// Uses a sum-tree (binary segment tree) for O(log n) proportional sampling:
    ///   - New transitions get max priority so they're always sampled at least once
    ///   - After training, priorities are updated to |TD-error| + epsilon
    ///   - Importance-sampling weights correct for the non-uniform sampling bias
    ///
    /// Thread safety architecture:
    ///   - Add (main thread): writes to a lock-free SPSC staging ring buffer.
    ///     Zero locks, zero contention, zero stalls.
    ///   - SampleInto (training thread): drains staging into PER buffer, then
    ///     samples indices. The lock is only held by the training thread itself.
    /// Buffer.BlockCopy for fast memcpy-style array copies.
    /// </summary>
    public class ReplayBuffer
    {
        private readonly int _capacity;
        private readonly int _obsDim;
        private readonly int _actDim;
        private readonly int _obsStride;
        private readonly int _actStride;

        private readonly float[] _obs;
        private readonly float[] _actions;
        private readonly float[] _rewards;
        private readonly float[] _nextObs;
        private readonly float[] _dones;

        // Sum-tree: 1-indexed binary tree with power-of-2 leaf count.
        // Leaves at [_treeCapacity, 2*_treeCapacity). Internal nodes store child sums.
        private readonly float[] _tree;
        private readonly int _treeCapacity;
        private float _maxPriority = 1.0f;
        private readonly float _perAlpha;
        private const float PER_EPSILON = 1e-5f;

        private int _head;
        private volatile int _count;
        private readonly object _lock = new object();
        private readonly Random _rng;

        // SPSC staging ring buffer: main thread writes, training thread drains.
        // Volatile write pointer has release semantics in C# — all preceding
        // array writes are visible to the reader after it reads _stageWrite.
        private const int STAGE_CAP = 512;
        private readonly float[] _stageObs;
        private readonly float[] _stageAct;
        private readonly float[] _stageRew;
        private readonly float[] _stageNextObs;
        private readonly float[] _stageDones;
        private volatile int _stageWrite;
        private int _stageRead;

        public int Count => _count;

        public int Capacity => _capacity;
        public int ObsDim => _obsDim;
        public int ActDim => _actDim;

        public ReplayBuffer(int capacity, int obsDim, int actDim, float perAlpha = 0.6f, int seed = 0)
        {
            _obsDim = obsDim;
            _actDim = actDim;
            _obsStride = obsDim * sizeof(float);
            _actStride = actDim * sizeof(float);
            _perAlpha = perAlpha;

            const long MAX_BYTES = 1_500_000_000L;
            int floatsPerEntry = 2 * obsDim + actDim + 1;
            int maxCapacity = (int)Math.Min(capacity, MAX_BYTES / (4L * floatsPerEntry));
            if (maxCapacity < capacity)
                UnityEngine.Debug.LogWarning(
                    $"ReplayBuffer: capped capacity {capacity} → {maxCapacity} " +
                    $"(obsDim={obsDim}, ~{(long)maxCapacity * floatsPerEntry * 4 / (1024 * 1024)}MB)");
            _capacity = maxCapacity;

            _obs = new float[_capacity * obsDim];
            _actions = new float[_capacity * actDim];
            _rewards = new float[_capacity];
            _nextObs = new float[_capacity * obsDim];
            _dones = new float[_capacity];

            _treeCapacity = 1;
            while (_treeCapacity < _capacity) _treeCapacity <<= 1;
            _tree = new float[2 * _treeCapacity];

            _head = 0;
            _count = 0;
            _rng = seed > 0 ? new Random(seed) : new Random();

            _stageObs = new float[STAGE_CAP * obsDim];
            _stageAct = new float[STAGE_CAP * actDim];
            _stageRew = new float[STAGE_CAP];
            _stageNextObs = new float[STAGE_CAP * obsDim];
            _stageDones = new float[STAGE_CAP];
            _stageWrite = 0;
            _stageRead = 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void TreeSet(int bufferIdx, float priority)
        {
            int idx = _treeCapacity + bufferIdx;
            _tree[idx] = priority;
            while (idx > 1)
            {
                idx >>= 1;
                _tree[idx] = _tree[idx << 1] + _tree[(idx << 1) | 1];
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int TreeFind(float value)
        {
            int idx = 1;
            while (idx < _treeCapacity)
            {
                int left = idx << 1;
                if (value <= _tree[left])
                    idx = left;
                else
                {
                    value -= _tree[left];
                    idx = left | 1;
                }
            }
            return idx - _treeCapacity;
        }

        /// <summary>
        /// Lock-free add: writes to SPSC staging buffer. The training thread
        /// drains staging into the PER buffer before each sample.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Add(float[] obs, float[] action, float reward, float[] nextObs)
        {
            Add(obs, action, reward, nextObs, 0f);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void Add(float[] obs, float[] action, float reward, float[] nextObs, float done)
        {
            int w = _stageWrite;
            int next = (w + 1) % STAGE_CAP;

            if (next == _stageRead)
            {
                AddDirect(obs, action, reward, nextObs, done);
                return;
            }

            Buffer.BlockCopy(obs, 0, _stageObs, w * _obsStride, _obsStride);
            Buffer.BlockCopy(action, 0, _stageAct, w * _actStride, _actStride);
            _stageRew[w] = reward;
            Buffer.BlockCopy(nextObs, 0, _stageNextObs, w * _obsStride, _obsStride);
            _stageDones[w] = done;

            _stageWrite = next;
        }

        private void AddDirect(float[] obs, float[] action, float reward, float[] nextObs, float done)
        {
            lock (_lock)
            {
                InsertEntry(obs, 0, action, 0, reward, nextObs, 0, done);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void InsertEntry(float[] obs, int obsOffset, float[] action, int actOffset,
            float reward, float[] nextObs, int nextObsOffset, float done = 0f)
        {
            int slot = _head;
            Buffer.BlockCopy(obs, obsOffset, _obs, slot * _obsStride, _obsStride);
            Buffer.BlockCopy(action, actOffset, _actions, slot * _actStride, _actStride);
            _rewards[slot] = reward;
            Buffer.BlockCopy(nextObs, nextObsOffset, _nextObs, slot * _obsStride, _obsStride);
            _dones[slot] = done;

            float priority = (float)Math.Pow(_maxPriority + PER_EPSILON, _perAlpha);
            TreeSet(slot, priority);

            _head = (_head + 1) % _capacity;
            if (_count < _capacity) _count++;
        }

        /// <summary>
        /// Drain all pending staging entries into the main PER buffer.
        /// Called by the training thread before sampling. Caller must hold _lock.
        /// </summary>
        private void DrainStaging()
        {
            int r = _stageRead;
            int w = _stageWrite; // volatile read — sees all stores before the write

            while (r != w)
            {
                InsertEntry(
                    _stageObs, r * _obsStride,
                    _stageAct, r * _actStride,
                    _stageRew[r],
                    _stageNextObs, r * _obsStride,
                    _stageDones[r]);
                r = (r + 1) % STAGE_CAP;
            }

            _stageRead = r;
        }

        /// <summary>
        /// Proportional prioritized sampling into a pre-allocated Batch.
        /// Lock is held only during index sampling (tree walks + IS weight math).
        /// The expensive data copies happen outside the lock — safe because the
        /// 200K-entry circular buffer takes ~800s to wrap, so sampled slots
        /// won't be overwritten during the brief unlocked copy window.
        /// </summary>
        public void SampleInto(Batch batch, float perBeta)
        {
            int n;
            int batchSize = batch.Size;

            // Phase 1: drain staging + sample indices under lock
            // Only the training thread enters this lock, so contention is zero.
            lock (_lock)
            {
                DrainStaging();
                n = _count;
                float total = _tree[1];

                if (total <= 0f)
                {
                    for (int i = 0; i < batchSize; i++)
                    {
                        batch.Indices[i] = _rng.Next(n);
                        batch.ISWeights[i] = 1f;
                    }
                }
                else
                {
                    float segment = total / batchSize;
                    for (int i = 0; i < batchSize; i++)
                    {
                        float lo = segment * i;
                        float hi = segment * (i + 1);
                        float value = lo + (float)(_rng.NextDouble() * (hi - lo));
                        if (value >= total) value = total * 0.999999f;

                        int idx = TreeFind(value);
                        if (idx >= n) idx = n - 1;
                        batch.Indices[i] = idx;

                        float prob = _tree[_treeCapacity + idx] / total;
                        if (prob < 1e-10f) prob = 1e-10f;
                        batch.ISWeights[i] = (float)Math.Pow(n * prob, -perBeta);
                    }

                    float maxW = batch.ISWeights[0];
                    for (int i = 1; i < batchSize; i++)
                        if (batch.ISWeights[i] > maxW) maxW = batch.ISWeights[i];
                    if (maxW > 0f)
                        for (int i = 0; i < batchSize; i++)
                            batch.ISWeights[i] /= maxW;
                }
            }

            for (int i = 0; i < batchSize; i++)
            {
                int idx = batch.Indices[i];
                Buffer.BlockCopy(_obs, idx * _obsStride, batch.Obs, i * _obsStride, _obsStride);
                Buffer.BlockCopy(_actions, idx * _actStride, batch.Actions, i * _actStride, _actStride);
                batch.Rewards[i] = _rewards[idx];
                Buffer.BlockCopy(_nextObs, idx * _obsStride, batch.NextObs, i * _obsStride, _obsStride);
                batch.Dones[i] = _dones[idx];
            }
        }

        /// <summary>
        /// Sample transitions with contiguous history windows for temporal context.
        /// For each sampled index i, reads the preceding seqLen-1 entries from the
        /// circular buffer. Episode boundaries (done flags) truncate the history
        /// and remaining entries are zero-padded with mask = 1.
        /// </summary>
        public void SampleSequencesInto(SequenceBatch batch, float perBeta)
        {
            int batchSize = batch.Size;
            int seqLen = batch.SeqLen;
            int entryDim = batch.EntryDim;
            int n;

            lock (_lock)
            {
                DrainStaging();
                n = _count;
                float total = _tree[1];

                if (total <= 0f)
                {
                    for (int i = 0; i < batchSize; i++)
                    {
                        batch.Indices[i] = _rng.Next(n);
                        batch.ISWeights[i] = 1f;
                    }
                }
                else
                {
                    float segment = total / batchSize;
                    for (int i = 0; i < batchSize; i++)
                    {
                        float lo = segment * i;
                        float hi = segment * (i + 1);
                        float value = lo + (float)(_rng.NextDouble() * (hi - lo));
                        if (value >= total) value = total * 0.999999f;

                        int idx = TreeFind(value);
                        if (idx >= n) idx = n - 1;
                        batch.Indices[i] = idx;

                        float prob = _tree[_treeCapacity + idx] / total;
                        if (prob < 1e-10f) prob = 1e-10f;
                        batch.ISWeights[i] = (float)Math.Pow(n * prob, -perBeta);
                    }

                    float maxW = batch.ISWeights[0];
                    for (int i = 1; i < batchSize; i++)
                        if (batch.ISWeights[i] > maxW) maxW = batch.ISWeights[i];
                    if (maxW > 0f)
                        for (int i = 0; i < batchSize; i++)
                            batch.ISWeights[i] /= maxW;
                }
            }

            for (int i = 0; i < batchSize; i++)
            {
                int idx = batch.Indices[i];

                Buffer.BlockCopy(_obs, idx * _obsStride, batch.Obs, i * _obsStride, _obsStride);
                Buffer.BlockCopy(_actions, idx * _actStride, batch.Actions, i * _actStride, _actStride);
                batch.Rewards[i] = _rewards[idx];
                Buffer.BlockCopy(_nextObs, idx * _obsStride, batch.NextObs, i * _obsStride, _obsStride);
                batch.Dones[i] = _dones[idx];

                int histBase = i * seqLen * entryDim;
                int maskBase = i * seqLen;

                // The last slot of the history window = the current transition
                // Walk backward to fill older entries
                for (int s = seqLen - 1; s >= 0; s--)
                {
                    int stepsBack = (seqLen - 1) - s;
                    int bufIdx = (idx - stepsBack + _capacity) % _capacity;
                    bool outOfRange = stepsBack >= n;

                    // Check for episode boundary: if any transition between bufIdx and idx
                    // has done=1, this entry is from a different episode
                    bool crossedEpisode = false;
                    if (!outOfRange && stepsBack > 0)
                    {
                        for (int k = 0; k < stepsBack; k++)
                        {
                            int checkIdx = (idx - k - 1 + _capacity) % _capacity;
                            if (_dones[checkIdx] > 0.5f)
                            {
                                crossedEpisode = true;
                                break;
                            }
                        }
                    }

                    int entryOffset = histBase + s * entryDim;
                    int maskOffset = maskBase + s;

                    if (outOfRange || crossedEpisode)
                    {
                        Array.Clear(batch.HistoryData, entryOffset, entryDim);
                        batch.HistoryMask[maskOffset] = 1f;
                    }
                    else
                    {
                        Buffer.BlockCopy(_obs, bufIdx * _obsStride,
                            batch.HistoryData, entryOffset * sizeof(float), _obsStride);
                        Buffer.BlockCopy(_actions, bufIdx * _actStride,
                            batch.HistoryData, (entryOffset + _obsDim) * sizeof(float), _actStride);
                        batch.HistoryData[entryOffset + _obsDim + _actDim] = _rewards[bufIdx];
                        batch.HistoryMask[maskOffset] = 0f;
                    }
                }
            }
        }

        /// <summary>
        /// Update priorities for sampled transitions after computing TD errors.
        /// </summary>
        public void UpdatePriorities(int[] indices, float[] absTDErrors, int count)
        {
            lock (_lock)
            {
                for (int i = 0; i < count; i++)
                {
                    float raw = absTDErrors[i] + PER_EPSILON;
                    if (raw > _maxPriority) _maxPriority = raw;
                    TreeSet(indices[i], (float)Math.Pow(raw, _perAlpha));
                }
            }
        }

        public unsafe void Save(BinaryWriter bw)
        {
            lock (_lock)
            {
                DrainStaging();
                bw.Write((int)-3); // version 3: adds dones
                bw.Write(_count);
                bw.Write(_head);
                WriteFloats(bw, _obs, _count * _obsDim);
                WriteFloats(bw, _actions, _count * _actDim);
                WriteFloats(bw, _rewards, _count);
                WriteFloats(bw, _nextObs, _count * _obsDim);
                WriteFloats(bw, _dones, _count);
                bw.Write(_treeCapacity);
                WriteFloats(bw, _tree, 2 * _treeCapacity);
                bw.Write(_maxPriority);
            }
        }

        public unsafe void Load(BinaryReader br)
        {
            lock (_lock)
            {
                int first = br.ReadInt32();
                int version;
                if (first < 0)
                {
                    version = -first;
                    _count = br.ReadInt32();
                }
                else
                {
                    version = 1;
                    _count = first;
                }

                _head = br.ReadInt32();
                ReadFloats(br, _obs, _count * _obsDim);
                ReadFloats(br, _actions, _count * _actDim);
                ReadFloats(br, _rewards, _count);
                ReadFloats(br, _nextObs, _count * _obsDim);

                if (version >= 3)
                {
                    ReadFloats(br, _dones, _count);
                }
                else
                {
                    Array.Clear(_dones, 0, _dones.Length);
                }

                if (version >= 2)
                {
                    int savedTreeCap = br.ReadInt32();
                    if (savedTreeCap == _treeCapacity)
                    {
                        ReadFloats(br, _tree, 2 * _treeCapacity);
                        _maxPriority = br.ReadSingle();
                    }
                    else
                    {
                        var temp = new float[2 * savedTreeCap];
                        ReadFloats(br, temp, 2 * savedTreeCap);
                        br.ReadSingle();
                        InitUniformPriorities();
                    }
                }
                else
                {
                    InitUniformPriorities();
                }
            }
        }

        private void InitUniformPriorities()
        {
            _maxPriority = 1.0f;
            float p = (float)Math.Pow(1.0f + PER_EPSILON, _perAlpha);
            for (int i = 0; i < _count; i++)
                _tree[_treeCapacity + i] = p;
            for (int i = _treeCapacity - 1; i >= 1; i--)
                _tree[i] = _tree[i << 1] + _tree[(i << 1) | 1];
        }

        // Reusable 64 KB chunk buffer — stays under the 85 KB LOH threshold.
        // Only accessed from the training thread (under _lock during Save/Load).
        [ThreadStatic] private static byte[] _ioChunk;
        private const int IO_CHUNK = 65536;

        private static unsafe void WriteFloats(BinaryWriter bw, float[] arr, int count)
        {
            if (_ioChunk == null) _ioChunk = new byte[IO_CHUNK];
            int totalBytes = count * sizeof(float);
            int written = 0;
            while (written < totalBytes)
            {
                int chunk = Math.Min(IO_CHUNK, totalBytes - written);
                Buffer.BlockCopy(arr, written, _ioChunk, 0, chunk);
                bw.Write(_ioChunk, 0, chunk);
                written += chunk;
            }
        }

        private static unsafe void ReadFloats(BinaryReader br, float[] arr, int count)
        {
            int totalBytes = count * sizeof(float);
            int read = 0;
            while (read < totalBytes)
            {
                int chunk = Math.Min(totalBytes - read, 4096);
                var bytes = br.ReadBytes(chunk);
                if (bytes.Length == 0) break;
                Buffer.BlockCopy(bytes, 0, arr, read, bytes.Length);
                read += bytes.Length;
            }
        }
    }

    public struct Batch
    {
        public readonly float[] Obs;
        public readonly float[] Actions;
        public readonly float[] Rewards;
        public readonly float[] NextObs;
        public readonly float[] Dones;
        public readonly int[] Indices;
        public readonly float[] ISWeights;
        public readonly float[] TDErrors;
        public readonly int Size;
        public readonly int ObsDim;
        public readonly int ActDim;

        public Batch(int size, int obsDim, int actDim)
        {
            Size = size;
            ObsDim = obsDim;
            ActDim = actDim;
            Obs = new float[size * obsDim];
            Actions = new float[size * actDim];
            Rewards = new float[size];
            NextObs = new float[size * obsDim];
            Dones = new float[size];
            Indices = new int[size];
            ISWeights = new float[size];
            TDErrors = new float[size];
        }
    }

    /// <summary>
    /// Extended batch that includes a history window for each sampled transition.
    /// Used by the TemporalContextEncoder for history-conditioned SAC.
    /// HistoryData layout: flat [batchSize * seqLen * entryDim] in (batch, seq, entry) order.
    /// </summary>
    public struct SequenceBatch
    {
        public readonly float[] Obs;
        public readonly float[] Actions;
        public readonly float[] Rewards;
        public readonly float[] NextObs;
        public readonly float[] Dones;
        public readonly int[] Indices;
        public readonly float[] ISWeights;
        public readonly float[] TDErrors;
        public readonly int Size;
        public readonly int ObsDim;
        public readonly int ActDim;

        public readonly float[] HistoryData;   // Size * SeqLen * EntryDim
        public readonly float[] HistoryMask;   // Size * SeqLen (1 = padded, 0 = valid)
        public readonly int SeqLen;
        public readonly int EntryDim;          // ObsDim + ActDim + 1

        public SequenceBatch(int size, int obsDim, int actDim, int seqLen)
        {
            Size = size;
            ObsDim = obsDim;
            ActDim = actDim;
            SeqLen = seqLen;
            EntryDim = obsDim + actDim + 1;
            Obs = new float[size * obsDim];
            Actions = new float[size * actDim];
            Rewards = new float[size];
            NextObs = new float[size * obsDim];
            Dones = new float[size];
            Indices = new int[size];
            ISWeights = new float[size];
            TDErrors = new float[size];
            HistoryData = new float[size * seqLen * EntryDim];
            HistoryMask = new float[size * seqLen];
        }
    }
}
