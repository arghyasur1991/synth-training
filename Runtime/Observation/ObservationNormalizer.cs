using System;
using System.IO;
using System.Runtime.CompilerServices;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Running mean/std observation normalizer using Welford's algorithm.
    ///
    /// Optimized: NormalizeInPlace writes directly into a pre-allocated buffer
    /// to avoid per-frame allocations on the main thread.
    /// </summary>
    public class ObservationNormalizer
    {
        private readonly int _dim;
        private readonly double[] _mean;
        private readonly double[] _variance;
        private double _count;

        private const float CLIP = 10f;
        private const double EPSILON = 1e-8;

        public int Dim => _dim;
        public double[] Mean => _mean;
        public double[] Variance => _variance;
        public double Count => _count;

        public ObservationNormalizer(int dim)
        {
            _dim = dim;
            _mean = new double[dim];
            _variance = new double[dim];
            for (int i = 0; i < dim; i++)
                _variance[i] = 1.0;
            _count = 1e-4;
        }

        /// <summary>
        /// Update running statistics from raw obs, write normalized result into dst.
        /// Zero-allocation hot path.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void NormalizeAndUpdateInPlace(float[] rawObs, float[] dst)
        {
            double totalCount = _count + 1.0;
            int dim = _dim;
            for (int i = 0; i < dim; i++)
            {
                double val = rawObs[i];
                double delta = val - _mean[i];
                double newMean = _mean[i] + delta / totalCount;
                double m2 = _variance[i] * _count + delta * (val - newMean);
                _mean[i] = newMean;
                _variance[i] = m2 / totalCount;

                float std = (float)Math.Sqrt(_variance[i] + EPSILON);
                float norm = ((float)(val - _mean[i])) / std;
                dst[i] = norm > CLIP ? CLIP : (norm < -CLIP ? -CLIP : norm);
            }
            _count = totalCount;
        }

        /// <summary>
        /// Normalize without updating stats (for replay / eval).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void NormalizeInPlace(float[] obs, float[] dst)
        {
            int dim = _dim;
            for (int i = 0; i < dim; i++)
            {
                float std = (float)Math.Sqrt(_variance[i] + EPSILON);
                float val = (obs[i] - (float)_mean[i]) / std;
                dst[i] = val > CLIP ? CLIP : (val < -CLIP ? -CLIP : val);
            }
        }

        public void Save(BinaryWriter bw)
        {
            bw.Write(_dim);
            bw.Write(_count);
            for (int i = 0; i < _dim; i++)
            {
                bw.Write(_mean[i]);
                bw.Write(_variance[i]);
            }
        }

        public void Load(BinaryReader br)
        {
            int dim = br.ReadInt32();
            if (dim != _dim)
                throw new InvalidOperationException(
                    $"ObservationNormalizer dimension mismatch: expected {_dim}, got {dim}");
            _count = br.ReadDouble();
            for (int i = 0; i < _dim; i++)
            {
                _mean[i] = br.ReadDouble();
                _variance[i] = br.ReadDouble();
            }
        }
    }
}
