using System;
using Mujoco;
using Genesis.Sentience.Synth;
using UnityEngine;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Configurable weights and scales for DeepMimic reward terms.
    /// Scales are tuned for mean-based (not sum-based) error computation.
    /// </summary>
    [Serializable]
    public struct DeepMimicConfig
    {
        public float PoseWeight;
        public float VelocityWeight;
        public float RootPoseWeight;
        public float RootVelocityWeight;
        public float KeyPositionWeight;

        public float PoseScale;
        public float VelocityScale;
        public float RootPoseScale;
        public float RootVelocityScale;
        public float KeyPositionScale;

        public static DeepMimicConfig Default => new DeepMimicConfig
        {
            PoseWeight = 0.50f,
            VelocityWeight = 0.10f,
            RootPoseWeight = 0.15f,
            RootVelocityWeight = 0.10f,
            KeyPositionWeight = 0.15f,
            PoseScale = 2.0f,
            VelocityScale = 0.1f,
            RootPoseScale = 5.0f,
            RootVelocityScale = 1.0f,
            KeyPositionScale = 10.0f
        };
    }

    /// <summary>Per-component snapshot from the last DeepMimic reward computation.</summary>
    public struct DeepMimicSnapshot
    {
        public float Pose, Velocity, RootPose, RootVelocity, KeyPosition;
        public float Total;
    }

    /// <summary>
    /// Standalone DeepMimic reward computation.
    /// Five exponential-based terms (all in [0,1]) compare current MuJoCo state
    /// against reference qpos/qvel/body positions from a motion clip.
    ///
    /// Extracted from SynthImitationEnv to be reusable by any skill.
    /// </summary>
    public class DeepMimicReward
    {
        private DeepMimicConfig _config;
        private DeepMimicSnapshot _lastSnapshot;

        public ref readonly DeepMimicSnapshot LastSnapshot => ref _lastSnapshot;

        public DeepMimicReward(DeepMimicConfig config)
        {
            _config = config;
        }

        /// <summary>
        /// Compute the full DeepMimic reward from current MuJoCo state vs reference.
        /// All arrays are indexed by the full model dimensions (not filtered).
        /// Only joints in includedQposIdx / includedQvelIdx are compared.
        /// </summary>
        public unsafe float Compute(
            MujocoLib.mjData_* data,
            MujocoLib.mjModel_* model,
            double[] refQpos,
            double[] refQvel,
            double[] refBodyPos,
            int[] keyBodyIndices,
            BoneFilterConfig filter)
        {
            float rPose = ComputePoseReward(data, refQpos, filter.includedQposIdx);
            float rVel = ComputeVelReward(data, refQvel, filter.includedQvelIdx);
            float rRootPose = ComputeRootPoseReward(data, refQpos);
            float rRootVel = ComputeRootVelReward(data, refQvel);
            float rKeyPos = ComputeKeyPosReward(data, model, refBodyPos, keyBodyIndices);

            float total = _config.PoseWeight * rPose
                        + _config.VelocityWeight * rVel
                        + _config.RootPoseWeight * rRootPose
                        + _config.RootVelocityWeight * rRootVel
                        + _config.KeyPositionWeight * rKeyPos;

            _lastSnapshot = new DeepMimicSnapshot
            {
                Pose = rPose,
                Velocity = rVel,
                RootPose = rRootPose,
                RootVelocity = rRootVel,
                KeyPosition = rKeyPos,
                Total = total
            };

            return total;
        }

        private unsafe float ComputePoseReward(MujocoLib.mjData_* data,
            double[] refQpos, int[] includedQposIdx)
        {
            if (includedQposIdx == null || includedQposIdx.Length == 0) return 1f;

            float sumSqErr = 0f;
            for (int i = 0; i < includedQposIdx.Length; i++)
            {
                int qi = includedQposIdx[i];
                float diff = (float)data->qpos[qi] - (float)refQpos[qi];
                sumSqErr += diff * diff;
            }

            float meanSqErr = sumSqErr / includedQposIdx.Length;
            return Mathf.Exp(-_config.PoseScale * meanSqErr);
        }

        private unsafe float ComputeVelReward(MujocoLib.mjData_* data,
            double[] refQvel, int[] includedQvelIdx)
        {
            if (includedQvelIdx == null || includedQvelIdx.Length == 0) return 1f;

            float sumSqErr = 0f;
            for (int i = 0; i < includedQvelIdx.Length; i++)
            {
                int vi = includedQvelIdx[i];
                float diff = (float)data->qvel[vi] - (float)refQvel[vi];
                sumSqErr += diff * diff;
            }

            float meanSqErr = sumSqErr / includedQvelIdx.Length;
            return Mathf.Exp(-_config.VelocityScale * meanSqErr);
        }

        private unsafe float ComputeRootPoseReward(MujocoLib.mjData_* data,
            double[] refQpos)
        {
            float zDiff = (float)data->qpos[2] - (float)refQpos[2];
            float rootPosErr = zDiff * zDiff;

            float rotErr = QuatAngleDiff(
                data->qpos[3], data->qpos[4], data->qpos[5], data->qpos[6],
                refQpos[3], refQpos[4], refQpos[5], refQpos[6]);
            float rootRotErr = rotErr * rotErr;

            return Mathf.Exp(-_config.RootPoseScale * (rootPosErr + rootRotErr));
        }

        private unsafe float ComputeRootVelReward(MujocoLib.mjData_* data,
            double[] refQvel)
        {
            float linErr = 0f;
            for (int i = 0; i < 3; i++)
            {
                float diff = (float)data->qvel[i] - (float)refQvel[i];
                linErr += diff * diff;
            }

            float angErr = 0f;
            for (int i = 3; i < 6; i++)
            {
                float diff = (float)data->qvel[i] - (float)refQvel[i];
                angErr += diff * diff;
            }

            return Mathf.Exp(-_config.RootVelocityScale * (linErr + angErr));
        }

        private unsafe float ComputeKeyPosReward(MujocoLib.mjData_* data,
            MujocoLib.mjModel_* model, double[] refBodyPos, int[] keyBodyIndices)
        {
            if (keyBodyIndices == null || keyBodyIndices.Length == 0) return 1f;

            int nbody = (int)model->nbody;

            float agentRootX = (float)data->xpos[0];
            float agentRootY = (float)data->xpos[1];
            float agentRootZ = (float)data->xpos[2];

            double refRootX = refBodyPos[0];
            double refRootY = refBodyPos[1];
            double refRootZ = refBodyPos[2];

            float sumSqErr = 0f;
            int validCount = 0;
            foreach (int bodyIdx in keyBodyIndices)
            {
                if (bodyIdx < 0 || bodyIdx >= nbody) continue;

                float ax = (float)data->xpos[bodyIdx * 3 + 0] - agentRootX;
                float ay = (float)data->xpos[bodyIdx * 3 + 1] - agentRootY;
                float az = (float)data->xpos[bodyIdx * 3 + 2] - agentRootZ;

                double rx = refBodyPos[bodyIdx * 3 + 0] - refRootX;
                double ry = refBodyPos[bodyIdx * 3 + 1] - refRootY;
                double rz = refBodyPos[bodyIdx * 3 + 2] - refRootZ;

                float dx = ax - (float)rx;
                float dy = ay - (float)ry;
                float dz = az - (float)rz;
                sumSqErr += dx * dx + dy * dy + dz * dz;
                validCount++;
            }

            if (validCount == 0) return 1f;
            float meanErr = sumSqErr / validCount;
            return Mathf.Exp(-_config.KeyPositionScale * meanErr);
        }

        private static float QuatAngleDiff(double w1, double x1, double y1, double z1,
                                            double w2, double x2, double y2, double z2)
        {
            double dot = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2;
            dot = Math.Abs(dot);
            dot = Math.Min(dot, 1.0);
            return (float)(2.0 * Math.Acos(dot));
        }
    }
}
