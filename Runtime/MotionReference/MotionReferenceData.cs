using System;
using System.Threading;
using UnityEngine;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Stores pre-extracted reference motion data from a Unity AnimationClip.
    /// 
    /// Design:
    ///   - Pure data container with no Unity runtime dependencies (usable by headless envs)
    ///   - Per-frame arrays of MuJoCo state: qpos, qvel, body positions (xipos)
    ///   - Time-indexed interpolated lookup for arbitrary time values
    ///   - Supports looping (wrap) and clamping modes
    ///   - Clip-agnostic: produced by MotionClipExtractor, consumed by SynthImitationEnv
    ///   - Extensible: any Mecanim humanoid animation can produce this data
    /// 
    /// Frame layout:
    ///   qposFrames[f][i]    — MuJoCo qpos at frame f (length nq)
    ///   qvelFrames[f][i]    — MuJoCo qvel at frame f (length nv)
    ///   bodyPosFrames[f][i] — MuJoCo xipos at frame f (length nbody*3, flattened x,y,z per body)
    /// </summary>
    public class MotionReferenceData
    {
        [ThreadStatic] private static System.Random _tlsRng;
        private static System.Random ThreadSafeRng =>
            _tlsRng ??= new System.Random(Environment.TickCount ^ Thread.CurrentThread.ManagedThreadId);

        // --- Dimensions (from MuJoCo model at extraction time) ---
        public int nq;     // qpos dimension
        public int nv;     // qvel dimension
        public int nbody;  // number of bodies

        // --- Frame data ---
        public double[][] qposFrames;     // [frameCount][nq]
        public double[][] qvelFrames;     // [frameCount][nv]
        public double[][] bodyPosFrames;  // [frameCount][nbody*3]

        // --- Timing ---
        public float fps;                 // frames per second used during extraction
        public int frameCount;            // total number of frames
        public bool isLooping;            // true = wrap around, false = clamp to last frame

        // --- Key body indices (for end-effector reward) ---
        // Indices into the body array (MuJoCo body IDs)
        public int[] keyBodyIndices;      // e.g., [head, lHand, rHand, lFoot, rFoot]

        /// <summary>
        /// Duration of the motion clip in seconds.
        /// </summary>
        public float Duration => frameCount > 1 ? (frameCount - 1) / fps : 0f;

        /// <summary>
        /// Get the frame duration (time between consecutive frames).
        /// </summary>
        public float FrameDuration => 1f / fps;

        /// <summary>
        /// Sample a random time uniformly from the motion clip.
        /// </summary>
        public float SampleRandomTime()
        {
            if (frameCount <= 1) return 0f;
            return (float)(ThreadSafeRng.NextDouble() * Duration);
        }

        /// <summary>
        /// Get interpolated motion state at arbitrary time t.
        /// Handles looping (wrap) and clamping automatically.
        /// Linear interpolation between adjacent frames.
        /// </summary>
        /// <param name="t">Time in seconds</param>
        /// <param name="outQpos">Pre-allocated array of length nq (will be filled)</param>
        /// <param name="outQvel">Pre-allocated array of length nv (will be filled)</param>
        /// <param name="outBodyPos">Pre-allocated array of length nbody*3 (will be filled), or null to skip</param>
        public void GetFrameAtTime(float t, double[] outQpos, double[] outQvel, double[] outBodyPos = null)
        {
            if (frameCount == 0) return;

            if (frameCount == 1)
            {
                // Single frame — no interpolation needed
                Array.Copy(qposFrames[0], outQpos, nq);
                Array.Copy(qvelFrames[0], outQvel, nv);
                if (outBodyPos != null && bodyPosFrames != null)
                    Array.Copy(bodyPosFrames[0], outBodyPos, nbody * 3);
                return;
            }

            // Normalize time
            float duration = Duration;
            if (isLooping)
            {
                t = t % duration;
                if (t < 0) t += duration;
            }
            else
            {
                t = Mathf.Clamp(t, 0f, duration);
            }

            // Find frame indices and interpolation factor
            float frameF = t * fps;
            int f0 = Mathf.FloorToInt(frameF);
            int f1 = f0 + 1;
            float alpha = frameF - f0;

            // Clamp/wrap frame indices
            if (isLooping)
            {
                f0 = f0 % frameCount;
                f1 = f1 % frameCount;
                if (f0 < 0) f0 += frameCount;
                if (f1 < 0) f1 += frameCount;
            }
            else
            {
                f0 = Mathf.Clamp(f0, 0, frameCount - 1);
                f1 = Mathf.Clamp(f1, 0, frameCount - 1);
            }

            // Linearly interpolate qpos
            var q0 = qposFrames[f0];
            var q1 = qposFrames[f1];
            for (int i = 0; i < nq; i++)
                outQpos[i] = q0[i] + alpha * (q1[i] - q0[i]);

            // Note: for root quaternion (qpos[3:7]), linear interpolation is approximate.
            // For small frame intervals this is acceptable. For large time gaps,
            // consider SLERP. Good enough for 30fps extraction.
            // Re-normalize the quaternion after interpolation
            NormalizeQuaternion(outQpos, 3);

            // Linearly interpolate qvel
            var v0 = qvelFrames[f0];
            var v1 = qvelFrames[f1];
            for (int i = 0; i < nv; i++)
                outQvel[i] = v0[i] + alpha * (v1[i] - v0[i]);

            // Linearly interpolate body positions (if requested)
            if (outBodyPos != null && bodyPosFrames != null)
            {
                var bp0 = bodyPosFrames[f0];
                var bp1 = bodyPosFrames[f1];
                int len = nbody * 3;
                for (int i = 0; i < len; i++)
                    outBodyPos[i] = bp0[i] + alpha * (bp1[i] - bp0[i]);
            }
        }

        /// <summary>
        /// Compute the motion phase at time t (0.0 = start, 1.0 = end/loop point).
        /// Useful for phase-based observations.
        /// </summary>
        public float GetPhase(float t)
        {
            float duration = Duration;
            if (duration <= 0f) return 0f;

            if (isLooping)
            {
                t = t % duration;
                if (t < 0) t += duration;
            }
            else
            {
                t = Mathf.Clamp(t, 0f, duration);
            }

            return t / duration;
        }

        /// <summary>
        /// Normalize a quaternion stored at qpos[offset:offset+4].
        /// </summary>
        private static void NormalizeQuaternion(double[] qpos, int offset)
        {
            double w = qpos[offset];
            double x = qpos[offset + 1];
            double y = qpos[offset + 2];
            double z = qpos[offset + 3];
            double norm = Math.Sqrt(w * w + x * x + y * y + z * z);
            if (norm > 1e-10)
            {
                double inv = 1.0 / norm;
                qpos[offset] = w * inv;
                qpos[offset + 1] = x * inv;
                qpos[offset + 2] = y * inv;
                qpos[offset + 3] = z * inv;
            }
        }
    }
}
