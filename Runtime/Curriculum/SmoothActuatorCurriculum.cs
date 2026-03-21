using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Mujoco;
using Genesis.Sentience.Synth;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Joint group classification for curriculum ramp rates.
    /// Locomotion joints ramp faster; fine-motor joints ramp slower.
    /// </summary>
    public enum JointGroup
    {
        Locomotion,
        FineMmotor,
        Auxiliary
    }

    /// <summary>
    /// Smooth actuator curriculum: replaces the 3-stage binary ActionCurriculum
    /// with continuous per-joint gain ramps.
    ///
    /// Each joint has a gain g_i ∈ [gMin, 1.0]. All start at gMin (passive).
    /// Gains ramp up monotonically based on a competency signal (EMA of centered
    /// reward). When competency improves, gains increase. When it drops, gains hold.
    ///
    /// Joint groups have different ramp rates:
    ///   - Locomotion (hips, legs, spine): 3x base rate
    ///   - Fine motor (hands, toes, head): 0.5x base rate
    ///   - Auxiliary (eyes, pectorals): 0.3x base rate
    ///
    /// Biologically motivated: infant actuator strength increases gradually
    /// as neural control improves (MIMo v2 growing body).
    /// </summary>
    [Serializable]
    public class SmoothActuatorCurriculum
    {
        private const float DEFAULT_G_MIN = 0.3f;
        private const float DEFAULT_G_MAX = 1.0f;
        private const float DEFAULT_RAMP_RATE = 1e-5f;
        private const float COMPETENCY_EMA_ALPHA = 0.001f;

        // Time-based minimum ramp: gains increase even without competency
        // improvement. At 1e-4/step and 30 SPS, gains increase by ~0.003/sec,
        // reaching 0.5 from 0.3 in ~67 seconds. Fast enough to prevent
        // stagnation, slow enough for the policy to adapt.
        private const float TIME_RAMP_PER_STEP = 1e-4f;

        private static readonly Dictionary<SynthBone, JointGroup> BoneGroups = new()
        {
            { SynthBone.Hips, JointGroup.Locomotion },
            { SynthBone.Spine, JointGroup.Locomotion },
            { SynthBone.Chest, JointGroup.Locomotion },
            { SynthBone.Neck, JointGroup.Locomotion },
            { SynthBone.LeftUpperLeg, JointGroup.Locomotion },
            { SynthBone.LeftLowerLeg, JointGroup.Locomotion },
            { SynthBone.LeftFoot, JointGroup.Locomotion },
            { SynthBone.RightUpperLeg, JointGroup.Locomotion },
            { SynthBone.RightLowerLeg, JointGroup.Locomotion },
            { SynthBone.RightFoot, JointGroup.Locomotion },
            { SynthBone.LeftShoulder, JointGroup.Locomotion },
            { SynthBone.LeftUpperArm, JointGroup.Locomotion },
            { SynthBone.LeftLowerArm, JointGroup.Locomotion },
            { SynthBone.RightShoulder, JointGroup.Locomotion },
            { SynthBone.RightUpperArm, JointGroup.Locomotion },
            { SynthBone.RightLowerArm, JointGroup.Locomotion },
            { SynthBone.UpperChest, JointGroup.Locomotion },
            { SynthBone.Head, JointGroup.FineMmotor },
            { SynthBone.Jaw, JointGroup.FineMmotor },
            { SynthBone.LeftHand, JointGroup.FineMmotor },
            { SynthBone.RightHand, JointGroup.FineMmotor },
            { SynthBone.LeftToes, JointGroup.FineMmotor },
            { SynthBone.RightToes, JointGroup.FineMmotor },
            { SynthBone.LeftEye, JointGroup.Auxiliary },
            { SynthBone.RightEye, JointGroup.Auxiliary },
            { SynthBone.LeftPectoral, JointGroup.Auxiliary },
            { SynthBone.RightPectoral, JointGroup.Auxiliary },
            { SynthBone.LeftGluteal, JointGroup.Auxiliary },
            { SynthBone.RightGluteal, JointGroup.Auxiliary },
        };

        private static readonly Dictionary<JointGroup, float> GroupMultiplier = new()
        {
            { JointGroup.Locomotion, 3.0f },
            { JointGroup.FineMmotor, 0.5f },
            { JointGroup.Auxiliary, 0.3f },
        };

        private int _totalActDim;
        private float[] _gains;
        private float[] _rampRates;
        private JointGroup[] _jointGroups;

        private float _competencyEMA;
        private float _prevCompetencyEMA;
        private bool _competencyInitialized;

        private float _gMin;
        private float _gMax;

        public float[] Gains => _gains;
        public int TotalActDim => _totalActDim;

        public float AverageGain
        {
            get
            {
                if (_gains == null || _gains.Length == 0) return 0f;
                float sum = 0f;
                for (int i = 0; i < _gains.Length; i++) sum += _gains[i];
                return sum / _gains.Length;
            }
        }

        public float LocomotionGain => GroupAverageGain(JointGroup.Locomotion);
        public float FineMotorGain => GroupAverageGain(JointGroup.FineMmotor);

        private float GroupAverageGain(JointGroup group)
        {
            if (_gains == null) return 0f;
            float sum = 0f;
            int count = 0;
            for (int i = 0; i < _gains.Length; i++)
            {
                if (_jointGroups[i] == group)
                {
                    sum += _gains[i];
                    count++;
                }
            }
            return count > 0 ? sum / count : 0f;
        }

        public unsafe void Initialize(
            MujocoLib.mjModel_* model,
            BoneFilterConfig filter,
            SynthBoneMapper boneMapper,
            float gMin = DEFAULT_G_MIN,
            float gMax = DEFAULT_G_MAX,
            float baseRampRate = DEFAULT_RAMP_RATE)
        {
            _totalActDim = filter.actDim;
            _gMin = gMin;
            _gMax = gMax;
            _gains = new float[_totalActDim];
            _rampRates = new float[_totalActDim];
            _jointGroups = new JointGroup[_totalActDim];

            for (int i = 0; i < _totalActDim; i++)
            {
                _gains[i] = gMin;
                _rampRates[i] = baseRampRate;
                _jointGroups[i] = JointGroup.Locomotion;
            }

            if (boneMapper == null)
            {
                Debug.LogWarning("SmoothActuatorCurriculum: No bone mapper — all joints at full gain");
                for (int i = 0; i < _totalActDim; i++) _gains[i] = gMax;
                return;
            }

            var globalToLocal = new Dictionary<int, int>();
            for (int i = 0; i < filter.includedActuatorIdx.Length; i++)
                globalToLocal[filter.includedActuatorIdx[i]] = i;

            var allActuators = UnityEngine.Object.FindObjectsByType<MjActuator>(
                FindObjectsInactive.Exclude, FindObjectsSortMode.None);

            int mapped = 0;
            foreach (SynthBone bone in Enum.GetValues(typeof(SynthBone)))
            {
                if (bone == SynthBone.Unknown) continue;
                var t = boneMapper.GetTransform(bone);
                if (t == null) continue;

                string boneName = t.name;
                JointGroup group = BoneGroups.TryGetValue(bone, out var g) ? g : JointGroup.Auxiliary;
                float groupMult = GroupMultiplier.TryGetValue(group, out var m) ? m : 1f;

                foreach (var act in allActuators)
                {
                    if (act.name.StartsWith(boneName) &&
                        act.name.Contains("Joint") &&
                        globalToLocal.TryGetValue(act.MujocoId, out int localIdx))
                    {
                        _jointGroups[localIdx] = group;
                        _rampRates[localIdx] = baseRampRate * groupMult;
                        mapped++;
                    }
                }
            }

            Debug.Log($"SmoothActuatorCurriculum: Mapped {mapped}/{_totalActDim} actuators, " +
                      $"gMin={gMin:F2}, gMax={gMax:F2}, baseRate={baseRampRate:E1}");
        }

        /// <summary>
        /// Step the curriculum. Call once per decision with the current centered reward.
        /// Gains ramp up monotonically when competency improves.
        /// </summary>
        public void Step(float centeredReward)
        {
            if (_gains == null) return;

            if (!_competencyInitialized)
            {
                _competencyEMA = centeredReward;
                _prevCompetencyEMA = centeredReward;
                _competencyInitialized = true;
            }
            else
            {
                _prevCompetencyEMA = _competencyEMA;
                _competencyEMA += COMPETENCY_EMA_ALPHA * (centeredReward - _competencyEMA);
            }

            float competencyDelta = Math.Max(0f, _competencyEMA - _prevCompetencyEMA);

            for (int i = 0; i < _totalActDim; i++)
            {
                // Competency-driven ramp (fast when learning)
                float competencyRamp = _rampRates[i] * competencyDelta;
                // Time-based minimum ramp (slow but unconditional — prevents deadlock)
                float timeRamp = TIME_RAMP_PER_STEP;
                _gains[i] += Math.Max(competencyRamp, timeRamp);
                if (_gains[i] > _gMax) _gains[i] = _gMax;
            }
        }

        /// <summary>
        /// Apply gain scaling to raw actions: action_i *= g_i.
        /// </summary>
        public void ApplyGains(float[] actions)
        {
            if (_gains == null) return;
            int n = Math.Min(actions.Length, _gains.Length);
            for (int i = 0; i < n; i++)
                actions[i] *= _gains[i];
        }

        public void Save(BinaryWriter bw)
        {
            bw.Write(_totalActDim);
            for (int i = 0; i < _totalActDim; i++)
                bw.Write(_gains[i]);
            bw.Write(_competencyEMA);
            bw.Write(_prevCompetencyEMA);
            bw.Write(_competencyInitialized);
        }

        public void Load(BinaryReader br)
        {
            int dim = br.ReadInt32();
            if (dim != _totalActDim)
            {
                Debug.LogWarning($"SmoothActuatorCurriculum: dimension mismatch " +
                    $"(saved={dim}, current={_totalActDim}), starting fresh");
                return;
            }
            for (int i = 0; i < dim; i++)
                _gains[i] = br.ReadSingle();
            _competencyEMA = br.ReadSingle();
            _prevCompetencyEMA = br.ReadSingle();
            _competencyInitialized = br.ReadBoolean();
        }
    }
}
