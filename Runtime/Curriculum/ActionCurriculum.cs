using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Mujoco;
using Genesis.Sentience.Synth;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Progressive Action Curriculum (PAC): unlocks joints in stages as the agent
    /// demonstrates competency. Inactive joints receive ctrl=0 (passive, spring-damped).
    ///
    /// Stages:
    ///   0 — Core locomotion: hips, upper legs, lower legs, spine, feet (~18 DOF)
    ///   1 — Upper body: + shoulders, upper/lower arms, chest, neck (~36 DOF)
    ///   2 — Fine motor: + hands, head, toes, upper chest (~60 DOF)
    ///   3 — Full body: all remaining joints (~90 DOF)
    ///
    /// The actor network keeps its full output dimension at all times — inactive
    /// actions are simply zeroed before applying to the simulation. This avoids
    /// network growing complexity while the replay buffer stays consistent.
    ///
    /// Stage advancement triggers when the standing-phase ratio exceeds the
    /// stage threshold over a sliding window. The agent must demonstrate sustained
    /// competency before new DOFs are unlocked.
    /// </summary>
    [Serializable]
    public class ActionCurriculum
    {
        private static readonly SynthBone[][] StageBones = new SynthBone[][]
        {
            // Stage 0: Core locomotion
            new[] {
                SynthBone.Hips, SynthBone.Spine,
                SynthBone.LeftUpperLeg, SynthBone.LeftLowerLeg, SynthBone.LeftFoot,
                SynthBone.RightUpperLeg, SynthBone.RightLowerLeg, SynthBone.RightFoot,
            },
            // Stage 1: Arms + upper body
            new[] {
                SynthBone.Chest, SynthBone.Neck,
                SynthBone.LeftShoulder, SynthBone.LeftUpperArm, SynthBone.LeftLowerArm,
                SynthBone.RightShoulder, SynthBone.RightUpperArm, SynthBone.RightLowerArm,
            },
            // Stage 2: Fine motor
            new[] {
                SynthBone.UpperChest, SynthBone.Head, SynthBone.Jaw,
                SynthBone.LeftHand, SynthBone.RightHand,
                SynthBone.LeftToes, SynthBone.RightToes,
            },
            // Stage 3: Auxiliary (everything else)
            new[] {
                SynthBone.LeftEye, SynthBone.RightEye,
                SynthBone.LeftPectoral, SynthBone.RightPectoral,
                SynthBone.LeftGluteal, SynthBone.RightGluteal,
            },
        };

        private static readonly float[] StageStandingThreshold = { 0.50f, 0.70f, 0.80f, 1f };
        private const int WINDOW_SIZE = 5000;
        private const int MIN_DECISIONS_PER_STAGE = 20000;

        private int _currentStage;
        private int _totalActDim;
        private bool[] _activeMask;
        private int _activeCount;
        private int _decisionsInStage;

        // Sliding window for phase tracking
        private int[] _phaseWindow;
        private int _windowIdx;
        private int _windowFilled;

        // Bone-to-actuator mapping (built at init)
        private Dictionary<SynthBone, List<int>> _boneActuatorMap;

        public int CurrentStage => _currentStage;
        public int TotalStages => StageBones.Length;
        public bool[] ActiveMask => _activeMask;
        public int ActiveActionDim => _activeCount;
        public int DecisionsInStage => _decisionsInStage;

        /// <summary>
        /// Initialize the curriculum from the MuJoCo model and bone mapper.
        /// Maps each bone to its actuator indices by matching MjActuator GameObjects
        /// to bone transforms in the hierarchy.
        /// </summary>
        public unsafe void Initialize(
            MujocoLib.mjModel_* model,
            BoneFilterConfig filter,
            SynthBoneMapper boneMapper)
        {
            _totalActDim = filter.actDim;
            _activeMask = new bool[_totalActDim];
            _phaseWindow = new int[WINDOW_SIZE];

            _boneActuatorMap = new Dictionary<SynthBone, List<int>>();

            if (boneMapper == null)
            {
                Debug.LogWarning("ActionCurriculum: No bone mapper — activating all joints immediately");
                SetStage(StageBones.Length - 1);
                return;
            }

            // Build bone → actuator index mapping.
            // Each bone's MjBody has child MjActuator components. Their MujocoId
            // is the global actuator index. We map that to the filter's included
            // actuator position (the action vector index).
            var globalToLocal = new Dictionary<int, int>();
            for (int i = 0; i < filter.includedActuatorIdx.Length; i++)
                globalToLocal[filter.includedActuatorIdx[i]] = i;

            var allActuators = UnityEngine.Object.FindObjectsByType<MjActuator>(FindObjectsSortMode.None);

            foreach (SynthBone bone in Enum.GetValues(typeof(SynthBone)))
            {
                if (bone == SynthBone.Unknown) continue;
                var t = boneMapper.GetTransform(bone);
                if (t == null) continue;

                var actuators = new List<int>();
                foreach (var act in allActuators)
                {
                    if (IsActuatorForBone(act.transform, t) &&
                        globalToLocal.TryGetValue(act.MujocoId, out int localIdx))
                    {
                        actuators.Add(localIdx);
                    }
                }

                if (actuators.Count > 0)
                    _boneActuatorMap[bone] = actuators;
            }

            int totalMapped = 0;
            foreach (var kv in _boneActuatorMap)
                totalMapped += kv.Value.Count;

            Debug.Log($"ActionCurriculum: Mapped {totalMapped}/{_totalActDim} actuators " +
                      $"to {_boneActuatorMap.Count} bones");

            SetStage(0);
        }

        /// <summary>
        /// Check if an MjActuator belongs to a bone by checking if its transform
        /// is a child (or the same as) the bone transform.
        /// </summary>
        private static bool IsActuatorForBone(Transform actuatorTransform, Transform boneTransform)
        {
            var t = actuatorTransform;
            while (t != null)
            {
                if (t == boneTransform) return true;
                t = t.parent;
            }
            return false;
        }

        /// <summary>
        /// Apply the curriculum mask to an action vector: zero out inactive actuators.
        /// </summary>
        public void MaskActions(float[] actions)
        {
            if (_activeMask == null || _currentStage >= StageBones.Length - 1) return;
            for (int i = 0; i < actions.Length && i < _activeMask.Length; i++)
            {
                if (!_activeMask[i])
                    actions[i] = 0f;
            }
        }

        /// <summary>
        /// Record a decision and check for stage advancement.
        /// Returns true if the stage changed.
        /// </summary>
        public bool Step(AgentPhase phase)
        {
            _decisionsInStage++;

            // Record phase in sliding window (Standing or Moving = success)
            _phaseWindow[_windowIdx] = (phase == AgentPhase.Standing || phase == AgentPhase.Moving) ? 1 : 0;
            _windowIdx = (_windowIdx + 1) % WINDOW_SIZE;
            if (_windowFilled < WINDOW_SIZE) _windowFilled++;

            if (_currentStage >= StageBones.Length - 1) return false;
            if (_decisionsInStage < MIN_DECISIONS_PER_STAGE) return false;
            if (_windowFilled < WINDOW_SIZE) return false;

            float standingRatio = 0f;
            for (int i = 0; i < _windowFilled; i++)
                standingRatio += _phaseWindow[i];
            standingRatio /= _windowFilled;

            if (standingRatio >= StageStandingThreshold[_currentStage])
            {
                int oldStage = _currentStage;
                SetStage(_currentStage + 1);
                Debug.Log($"ActionCurriculum: Stage {oldStage} → {_currentStage} " +
                          $"(standingRatio={standingRatio:F2} >= {StageStandingThreshold[oldStage]:F2}, " +
                          $"activeJoints={_activeCount}/{_totalActDim})");
                return true;
            }

            return false;
        }

        /// <summary>
        /// Adjusted target entropy based on active action dimensions.
        /// Only active joints contribute to the entropy target.
        /// </summary>
        public float AdjustedTargetEntropy(float entropyScale)
        {
            return -_activeCount * entropyScale;
        }

        private void SetStage(int stage)
        {
            _currentStage = Math.Min(stage, StageBones.Length - 1);
            _decisionsInStage = 0;
            _windowFilled = 0;
            _windowIdx = 0;

            // Activate all actuators for stages 0..currentStage
            Array.Clear(_activeMask, 0, _activeMask.Length);
            for (int s = 0; s <= _currentStage; s++)
            {
                foreach (var bone in StageBones[s])
                {
                    if (_boneActuatorMap.TryGetValue(bone, out var indices))
                    {
                        foreach (int idx in indices)
                        {
                            if (idx >= 0 && idx < _activeMask.Length)
                                _activeMask[idx] = true;
                        }
                    }
                }
            }

            // Final stage: activate everything (catch unmapped actuators)
            if (_currentStage >= StageBones.Length - 1)
            {
                for (int i = 0; i < _activeMask.Length; i++)
                    _activeMask[i] = true;
            }

            _activeCount = 0;
            for (int i = 0; i < _activeMask.Length; i++)
                if (_activeMask[i]) _activeCount++;

            Debug.Log($"ActionCurriculum: Stage {_currentStage} — " +
                      $"{_activeCount}/{_totalActDim} actuators active");
        }

        public void Save(BinaryWriter bw)
        {
            bw.Write(_currentStage);
            bw.Write(_decisionsInStage);
            bw.Write(_windowFilled);
            bw.Write(_windowIdx);
            for (int i = 0; i < WINDOW_SIZE; i++)
                bw.Write(_phaseWindow != null && i < _phaseWindow.Length ? _phaseWindow[i] : 0);
        }

        public void Load(BinaryReader br)
        {
            int stage = br.ReadInt32();
            _decisionsInStage = br.ReadInt32();
            _windowFilled = br.ReadInt32();
            _windowIdx = br.ReadInt32();
            if (_phaseWindow == null) _phaseWindow = new int[WINDOW_SIZE];
            for (int i = 0; i < WINDOW_SIZE; i++)
                _phaseWindow[i] = br.ReadInt32();

            if (_activeMask != null)
                SetStage(stage);
        }
    }
}
