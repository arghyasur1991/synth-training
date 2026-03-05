using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using UnityEngine;
using Mujoco;
using Genesis.Sentience.Synth;
using Random = System.Random;
using Debug = UnityEngine.Debug;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Imitation learning skill using PPO + DeepMimic reward.
    /// Episodic: when the agent falls or deviates too much, it resets
    /// to a random reference pose (RSI). Supports multi-clip training
    /// with hard negative mining (failure-weighted clip sampling).
    /// </summary>
    public class ImitationLearningSkill : BaseTrainingSkill
    {
        [Header("PPO")]
        public PPOConfig ppoConfig = new PPOConfig();

        [Header("Reference Motion")]
        [Tooltip("Animation clips to imitate")]
        public AnimationClip[] referenceClips;

        [Tooltip("FPS for motion extraction")]
        public float extractionFps = 30f;

        [Tooltip("MuJoCo body names for key-position reward (head, hands, feet)")]
        public string[] keyBodyNames = { "Head", "lHand", "rHand", "lFoot", "rFoot" };

        [Header("DeepMimic Reward")]
        public DeepMimicConfig rewardConfig = DeepMimicConfig.Default;

        [Header("Termination")]
        [Tooltip("Min root Z as fraction of initial standing Z")]
        [Range(0.1f, 0.8f)]
        public float healthyZMinFraction = 0.3f;

        [Tooltip("Max root Z as fraction of initial standing Z")]
        [Range(1.0f, 3.0f)]
        public float healthyZMaxFraction = 2.0f;

        [Tooltip("Max body position error (m) before termination")]
        public float poseTerminationDist = 1.5f;

        [Tooltip("Max steps per episode (0 = clip duration)")]
        public int maxEpisodeSteps = 0;

        [Header("Hard Negative Mining")]
        [Tooltip("Weight clips by failure rate for focused practice")]
        public bool hardNegativeMining = true;

        [Tooltip("Steps between clip weight updates")]
        public int weightUpdateInterval = 5000;

        [Tooltip("Episode return above this = success")]
        public float successThreshold = 0.7f;

        [Tooltip("Minimum clip weight (prevents complete exclusion)")]
        public float minClipWeight = 0.1f;

        [Header("RSI Noise")]
        [Tooltip("Random noise added to qpos on reset")]
        [Range(0f, 0.05f)]
        public float resetNoise = 0.005f;

        // ── ISynthSkill ─────────────────────────────────────────────────

        public override string Name => "ImitationLearning";

        // ── Internal state ──────────────────────────────────────────────

        private MotionReferenceData[] _motionLibrary;
        private DeepMimicReward _reward;

        private int _activeClipIndex;
        private float _motionTime;
        private double[] _refQpos;
        private double[] _refQvel;
        private double[] _refBodyPos;
        private int[] _keyBodyIndices;

        private int _episodeStep;
        private float _episodeReturn;
        private float _standingZ;
        private int _nq, _nv, _nbody;

        // Hard negative mining state
        private float[] _clipSuccessRates;
        private int[] _clipAttempts;
        private int[] _clipSuccesses;
        private float[] _clipWeights;
        private float _clipWeightSum;
        private int _stepsSinceWeightUpdate;

        // Reference obs buffer
        private int _refObsDim;
        private float[] _refObsBuffer;
        private float[] _fullObsBuffer;

        // ── Diagnostics ─────────────────────────────────────────────────

        public int ActiveClipIndex => _activeClipIndex;
        public string ActiveClipName => referenceClips != null && _activeClipIndex < referenceClips.Length
            ? referenceClips[_activeClipIndex]?.name ?? "null" : "none";
        public int EpisodeStep => _episodeStep;
        public float EpisodeReturn => _episodeReturn;
        public ref readonly DeepMimicSnapshot LastRewardSnapshot =>
            ref _reward.LastSnapshot;

        // ── BaseTrainingSkill hooks ─────────────────────────────────────

        protected override ISkillTrainer CreateTrainer()
        {
            return new PPOSkillTrainer(ppoConfig, _isMobile);
        }

        protected override (int obsDim, int actDim) GetDimensions()
        {
            int physObs = _filter.physicsObsDim;
            int actDim = _filter.actDim;

            int filteredQpos = _filter.includedQposIdx?.Length ?? 0;
            int filteredQvel = _filter.includedQvelIdx?.Length ?? 0;
            _refObsDim = filteredQpos + filteredQvel + 1; // +1 for phase

            return (physObs + _refObsDim + actDim, actDim);
        }

        protected override unsafe void OnSkillInitialize()
        {
            var model = MjScene.Instance.Model;
            _nq = (int)model->nq;
            _nv = (int)model->nv;
            _nbody = (int)model->nbody;
            _standingZ = (float)MjScene.Instance.Data->qpos[2];

            _refQpos = new double[_nq];
            _refQvel = new double[_nv];
            _refBodyPos = new double[_nbody * 3];

            _refObsBuffer = new float[_refObsDim];
            var (obsDim, actDim) = GetDimensions();
            _fullObsBuffer = new float[obsDim];

            _reward = new DeepMimicReward(rewardConfig);

            ResolveKeyBodyIndices();
            ExtractMotionLibrary();

            if (_motionLibrary.Length > 0)
            {
                InitClipWeights();
                SelectNewClipAndReset();
            }

            if (maxEpisodeSteps <= 0 && _motionLibrary.Length > 0)
                maxEpisodeSteps = Mathf.RoundToInt(_motionLibrary[0].Duration * extractionFps) * 2;
        }

        protected override float[] BuildFullObs(float[] normalizedPhysicsObs)
        {
            int physLen = _physicsObsDim;
            Buffer.BlockCopy(normalizedPhysicsObs, 0, _fullObsBuffer, 0,
                physLen * sizeof(float));

            BuildReferenceObs();
            Buffer.BlockCopy(_refObsBuffer, 0, _fullObsBuffer,
                physLen * sizeof(float), _refObsDim * sizeof(float));

            int refEnd = physLen + _refObsDim;
            Buffer.BlockCopy(_smoothedAction, 0, _fullObsBuffer,
                refEnd * sizeof(float), _smoothedAction.Length * sizeof(float));

            return _fullObsBuffer;
        }

        protected override unsafe float ComputeReward()
        {
            return _reward.Compute(
                MjScene.Instance.Data, MjScene.Instance.Model,
                _refQpos, _refQvel, _refBodyPos,
                _keyBodyIndices, _filter);
        }

        protected override unsafe bool CheckTermination()
        {
            float rootZ = (float)MjScene.Instance.Data->qpos[2];
            float healthyMin = _standingZ * healthyZMinFraction;
            float healthyMax = _standingZ * healthyZMaxFraction;

            if (rootZ < healthyMin || rootZ > healthyMax)
                return true;

            if (_episodeStep >= maxEpisodeSteps && maxEpisodeSteps > 0)
                return true;

            return false;
        }

        protected override unsafe void OnTermination()
        {
            UpdateClipSuccess(_activeClipIndex, _episodeReturn);
            SelectNewClipAndReset();
        }

        protected override void OnTransitionStored(float reward, bool done)
        {
            _motionTime += Time.fixedDeltaTime * frameSkip;
            if (_motionLibrary != null && _motionLibrary.Length > 0)
            {
                _motionLibrary[_activeClipIndex].GetFrameAtTime(
                    _motionTime, _refQpos, _refQvel, _refBodyPos);
            }

            _episodeStep++;
            _episodeReturn += reward;

            _stepsSinceWeightUpdate++;
            if (hardNegativeMining && _stepsSinceWeightUpdate >= weightUpdateInterval)
            {
                UpdateClipWeights();
                _stepsSinceWeightUpdate = 0;
            }

            if (_metrics != null)
                _metrics.SampleGeneric(reward, _trainer?.StepsPerSecond ?? 0f,
                    _trainer?.ExperienceCount ?? 0, GetDiagnostics());
        }

        protected override void SaveExtraState(string directory)
        {
            if (_clipSuccessRates != null)
            {
                BaseTrainingSkill.WriteBinaryTmpStatic(
                    Path.Combine(directory, "imitation_state.bin"), bw =>
                    {
                        bw.Write(_activeClipIndex);
                        bw.Write(_motionTime);
                        bw.Write(_episodeStep);
                        bw.Write(_episodeReturn);
                        bw.Write(_clipSuccessRates.Length);
                        for (int i = 0; i < _clipSuccessRates.Length; i++)
                        {
                            bw.Write(_clipSuccessRates[i]);
                            bw.Write(_clipAttempts[i]);
                            bw.Write(_clipSuccesses[i]);
                        }
                    });
            }
        }

        protected override void LoadExtraState(string directory)
        {
            string path = Path.Combine(directory, "imitation_state.bin");
            if (!File.Exists(path)) return;

            try
            {
                using var br = new BinaryReader(File.OpenRead(path));
                _activeClipIndex = br.ReadInt32();
                _motionTime = br.ReadSingle();
                _episodeStep = br.ReadInt32();
                _episodeReturn = br.ReadSingle();

                int count = br.ReadInt32();
                if (count == (_clipSuccessRates?.Length ?? 0))
                {
                    for (int i = 0; i < count; i++)
                    {
                        _clipSuccessRates[i] = br.ReadSingle();
                        _clipAttempts[i] = br.ReadInt32();
                        _clipSuccesses[i] = br.ReadInt32();
                    }
                    UpdateClipWeights();
                }

                if (_motionLibrary != null && _motionLibrary.Length > 0)
                {
                    _activeClipIndex = Math.Clamp(_activeClipIndex, 0, _motionLibrary.Length - 1);
                    _motionLibrary[_activeClipIndex].GetFrameAtTime(
                        _motionTime, _refQpos, _refQvel, _refBodyPos);
                }

                Debug.Log($"ImitationLearning: Loaded state — clip={_activeClipIndex}, " +
                    $"ep_step={_episodeStep}");
            }
            catch (Exception e)
            {
                Debug.LogWarning($"ImitationLearning: Extra state load failed ({e.Message})");
            }
        }

        public override Dictionary<string, float> GetDiagnostics()
        {
            var d = new Dictionary<string, float>();
            d["activeClip"] = _activeClipIndex;
            d["episodeStep"] = _episodeStep;
            d["episodeReturn"] = _episodeReturn;
            d["motionPhase"] = _motionLibrary != null && _activeClipIndex < _motionLibrary.Length
                ? _motionLibrary[_activeClipIndex].GetPhase(_motionTime) : 0f;

            var snap = _reward?.LastSnapshot ?? default;
            d["dmPose"] = snap.Pose;
            d["dmVelocity"] = snap.Velocity;
            d["dmRootPose"] = snap.RootPose;
            d["dmRootVel"] = snap.RootVelocity;
            d["dmKeyPos"] = snap.KeyPosition;
            d["dmTotal"] = snap.Total;

            var ppoTrainer = _trainer as PPOSkillTrainer;
            if (ppoTrainer != null)
            {
                d["policyLoss"] = ppoTrainer.LastPolicyLoss;
                d["valueLoss"] = ppoTrainer.LastValueLoss;
                d["entropy"] = ppoTrainer.LastEntropy;
                d["approxKL"] = ppoTrainer.LastApproxKL;
            }

            return d;
        }

        protected override void OnSkillValidate()
        {
            if (ppoConfig == null) ppoConfig = new PPOConfig();
        }

        // ── Motion Library ──────────────────────────────────────────────

        private unsafe void ExtractMotionLibrary()
        {
            if (referenceClips == null || referenceClips.Length == 0)
            {
                _motionLibrary = Array.Empty<MotionReferenceData>();
                Debug.LogWarning("ImitationLearning: No reference clips assigned");
                return;
            }

            var humanoidRoot = _entity != null ? _entity.gameObject : gameObject;
            var extractor = new MotionClipExtractor();
            var results = new List<MotionReferenceData>();
            var sw = Stopwatch.StartNew();

            Debug.Log($"ImitationLearning: Extracting {referenceClips.Length} clips...");
            for (int i = 0; i < referenceClips.Length; i++)
            {
                if (referenceClips[i] == null) continue;
                var data = extractor.Extract(
                    referenceClips[i], humanoidRoot, MjScene.Instance.Model,
                    extractionFps, true, keyBodyNames);
                if (data != null)
                    results.Add(data);

                if ((i + 1) % 10 == 0 || i == referenceClips.Length - 1)
                    Debug.Log($"ImitationLearning: Extracted {i + 1}/{referenceClips.Length} " +
                        $"({sw.ElapsedMilliseconds}ms)");
            }

            _motionLibrary = results.ToArray();
            Debug.Log($"ImitationLearning: {_motionLibrary.Length} valid clips extracted");
        }

        private unsafe void ResolveKeyBodyIndices()
        {
            if (keyBodyNames == null || keyBodyNames.Length == 0)
            {
                _keyBodyIndices = Array.Empty<int>();
                return;
            }

            var model = MjScene.Instance.Model;
            var indices = new List<int>();
            foreach (var name in keyBodyNames)
            {
                int id = MujocoLib.mj_name2id(model, (int)MujocoLib.mjtObj.mjOBJ_BODY, name);
                if (id >= 0)
                    indices.Add(id);
            }
            _keyBodyIndices = indices.ToArray();
        }

        // ── RSI Reset ───────────────────────────────────────────────────

        private unsafe void SelectNewClipAndReset()
        {
            if (_motionLibrary.Length == 0) return;

            _activeClipIndex = SampleWeightedClip();
            _motionTime = _motionLibrary[_activeClipIndex].SampleRandomTime();

            _motionLibrary[_activeClipIndex].GetFrameAtTime(
                _motionTime, _refQpos, _refQvel, _refBodyPos);

            var data = MjScene.Instance.Data;
            var model = MjScene.Instance.Model;

            for (int i = 0; i < _nq; i++)
                data->qpos[i] = _refQpos[i] + (i >= 7 ? (_rng.NextDouble() * 2.0 - 1.0) * resetNoise : 0);

            // Re-normalize root quaternion
            double w = data->qpos[3], x = data->qpos[4], y = data->qpos[5], z = data->qpos[6];
            double norm = Math.Sqrt(w * w + x * x + y * y + z * z);
            if (norm > 1e-10)
            {
                double inv = 1.0 / norm;
                data->qpos[3] = w * inv;
                data->qpos[4] = x * inv;
                data->qpos[5] = y * inv;
                data->qpos[6] = z * inv;
            }

            for (int i = 0; i < _nv; i++)
                data->qvel[i] = _refQvel[i];

            MujocoLib.mj_forward(model, data);

            _episodeStep = 0;
            _episodeReturn = 0f;
            _hasPrevTransition = false;
        }

        // ── Hard Negative Mining ────────────────────────────────────────

        private void InitClipWeights()
        {
            int n = _motionLibrary.Length;
            _clipSuccessRates = new float[n];
            _clipAttempts = new int[n];
            _clipSuccesses = new int[n];
            _clipWeights = new float[n];

            for (int i = 0; i < n; i++)
                _clipWeights[i] = 1f;
            _clipWeightSum = n;
        }

        private int SampleWeightedClip()
        {
            if (!hardNegativeMining || _clipWeights == null || _motionLibrary.Length <= 1)
                return _rng.Next(0, _motionLibrary.Length);

            float r = (float)_rng.NextDouble() * _clipWeightSum;
            float cumulative = 0f;
            for (int i = 0; i < _clipWeights.Length; i++)
            {
                cumulative += _clipWeights[i];
                if (r <= cumulative) return i;
            }
            return _clipWeights.Length - 1;
        }

        private void UpdateClipSuccess(int clipIdx, float episodeReturn)
        {
            if (clipIdx < 0 || clipIdx >= (_clipAttempts?.Length ?? 0)) return;
            _clipAttempts[clipIdx]++;
            if (episodeReturn >= successThreshold)
                _clipSuccesses[clipIdx]++;

            _clipSuccessRates[clipIdx] = _clipAttempts[clipIdx] > 0
                ? (float)_clipSuccesses[clipIdx] / _clipAttempts[clipIdx]
                : 0f;
        }

        private void UpdateClipWeights()
        {
            if (_clipWeights == null) return;

            _clipWeightSum = 0f;
            for (int i = 0; i < _clipWeights.Length; i++)
            {
                float failRate = 1f - _clipSuccessRates[i];
                _clipWeights[i] = Math.Max(failRate, minClipWeight);
                _clipWeightSum += _clipWeights[i];
            }

            Debug.Log($"ImitationLearning: Updated clip weights — " +
                $"min={MinArray(_clipSuccessRates):F2}, max={MaxArray(_clipSuccessRates):F2}");
        }

        private unsafe void BuildReferenceObs()
        {
            var data = MjScene.Instance.Data;
            int idx = 0;

            // Relative qpos: (current - reference), naturally centered around 0
            var inclQpos = _filter.includedQposIdx;
            if (inclQpos != null)
            {
                for (int i = 0; i < inclQpos.Length; i++)
                {
                    int qi = inclQpos[i];
                    _refObsBuffer[idx++] = (float)(data->qpos[qi] - _refQpos[qi]);
                }
            }

            // Relative qvel: (current - reference)
            var inclQvel = _filter.includedQvelIdx;
            if (inclQvel != null)
            {
                for (int i = 0; i < inclQvel.Length; i++)
                {
                    int vi = inclQvel[i];
                    _refObsBuffer[idx++] = (float)(data->qvel[vi] - _refQvel[vi]);
                }
            }

            float phase = _motionLibrary != null && _activeClipIndex < _motionLibrary.Length
                ? _motionLibrary[_activeClipIndex].GetPhase(_motionTime) : 0f;
            _refObsBuffer[idx] = phase;
        }

        private static float MinArray(float[] arr)
        {
            if (arr == null || arr.Length == 0) return 0f;
            float min = arr[0];
            for (int i = 1; i < arr.Length; i++)
                if (arr[i] < min) min = arr[i];
            return min;
        }

        private static float MaxArray(float[] arr)
        {
            if (arr == null || arr.Length == 0) return 0f;
            float max = arr[0];
            for (int i = 1; i < arr.Length; i++)
                if (arr[i] > max) max = arr[i];
            return max;
        }
    }
}
