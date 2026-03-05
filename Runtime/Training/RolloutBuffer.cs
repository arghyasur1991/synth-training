using System;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Fixed-size sequential buffer for on-policy PPO rollouts.
    /// Stores one step per Add() call. After the buffer is full,
    /// compute GAE, iterate mini-batches for the PPO update, then Clear().
    /// No persistence needed (on-policy data is discarded after each update).
    /// </summary>
    public class RolloutBuffer
    {
        private readonly int _capacity;
        private readonly int _obsDim;
        private readonly int _actDim;

        private readonly float[] _obs;
        private readonly float[] _actions;
        private readonly float[] _rewards;
        private readonly float[] _dones;
        private readonly float[] _logProbs;
        private readonly float[] _values;

        private float[] _advantages;
        private float[] _returns;

        private int _count;

        public int Count => _count;
        public int Capacity => _capacity;
        public int ObsDim => _obsDim;
        public int ActDim => _actDim;
        public bool IsFull => _count >= _capacity;

        public RolloutBuffer(int capacity, int obsDim, int actDim)
        {
            _capacity = capacity;
            _obsDim = obsDim;
            _actDim = actDim;

            _obs = new float[capacity * obsDim];
            _actions = new float[capacity * actDim];
            _rewards = new float[capacity];
            _dones = new float[capacity];
            _logProbs = new float[capacity];
            _values = new float[capacity];

            _advantages = new float[capacity];
            _returns = new float[capacity];
        }

        /// <summary>Append one transition. Returns true if buffer became full.</summary>
        public bool Add(float[] obs, float[] action, float reward, float done,
                        float logProb, float value)
        {
            if (_count >= _capacity) return true;

            int obsOff = _count * _obsDim;
            int actOff = _count * _actDim;
            Buffer.BlockCopy(obs, 0, _obs, obsOff * sizeof(float), _obsDim * sizeof(float));
            Buffer.BlockCopy(action, 0, _actions, actOff * sizeof(float), _actDim * sizeof(float));
            _rewards[_count] = reward;
            _dones[_count] = done;
            _logProbs[_count] = logProb;
            _values[_count] = value;

            _count++;
            return _count >= _capacity;
        }

        /// <summary>
        /// Compute Generalized Advantage Estimation in-place.
        /// Call after the buffer is full, before iterating mini-batches.
        /// </summary>
        public void ComputeGAE(float lastValue, float lastDone, float gamma, float gaeLambda)
        {
            float lastGaeLam = 0f;
            for (int t = _count - 1; t >= 0; t--)
            {
                float nextNonTerminal;
                float nextValue;
                if (t == _count - 1)
                {
                    nextNonTerminal = 1f - lastDone;
                    nextValue = lastValue;
                }
                else
                {
                    nextNonTerminal = 1f - _dones[t + 1];
                    nextValue = _values[t + 1];
                }

                float delta = _rewards[t] + gamma * nextValue * nextNonTerminal - _values[t];
                lastGaeLam = delta + gamma * gaeLambda * nextNonTerminal * lastGaeLam;
                _advantages[t] = lastGaeLam;
                _returns[t] = _advantages[t] + _values[t];
            }
        }

        /// <summary>
        /// Copy a mini-batch of data at the given indices into pre-allocated arrays.
        /// </summary>
        public void GetMiniBatch(int[] indices, int indicesOffset, int mbSize,
            float[] mbObs, float[] mbActions, float[] mbLogProbs,
            float[] mbAdvantages, float[] mbReturns, float[] mbValues)
        {
            for (int i = 0; i < mbSize; i++)
            {
                int idx = indices[indicesOffset + i];
                Buffer.BlockCopy(_obs, idx * _obsDim * sizeof(float),
                    mbObs, i * _obsDim * sizeof(float), _obsDim * sizeof(float));
                Buffer.BlockCopy(_actions, idx * _actDim * sizeof(float),
                    mbActions, i * _actDim * sizeof(float), _actDim * sizeof(float));
                mbLogProbs[i] = _logProbs[idx];
                mbAdvantages[i] = _advantages[idx];
                mbReturns[i] = _returns[idx];
                mbValues[i] = _values[idx];
            }
        }

        /// <summary>
        /// Get the last observation and done flag for bootstrapping value.
        /// </summary>
        public void GetLast(float[] lastObs, out float lastDone)
        {
            int lastIdx = _count - 1;
            Buffer.BlockCopy(_obs, lastIdx * _obsDim * sizeof(float),
                lastObs, 0, _obsDim * sizeof(float));
            lastDone = _dones[lastIdx];
        }

        public void Clear()
        {
            _count = 0;
        }
    }
}
