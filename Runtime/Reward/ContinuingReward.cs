using System;
using System.IO;
using System.Runtime.CompilerServices;
using UnityEngine;
using Mujoco;
using Genesis.Sentience.Synth;
using Random = System.Random;

namespace Genesis.Sentience.Learning
{
    public enum AgentPhase { Fallen, Recovering, Standing, Moving }

    /// <summary>Per-component weighted reward values from the last Compute() call.</summary>
    public struct RewardSnapshot
    {
        public float Alive, Height, Orientation, Energy, Recovery;
        public float Imitation, VelocityUp, Comfort;
        public float FootSupport, HandBrace, ActiveSupport;
        public float PhaseBonus, StandBlend, RootZ;
        public float RawReward, CenteredReward, RewardBar;
        public AgentPhase Phase;
    }

    /// <summary>
    /// Multi-phase continuing reward for non-episodic learning.
    ///
    /// Optimized hot path:
    ///   - Reference frames stored in a single contiguous double[] for cache locality
    ///   - Nearest-frame search uses unsafe fixed pointers, no bounds checks
    ///   - Early-exit in nearest-frame when running distance exceeds current best
    ///   - Amortized search: only recomputes every N steps, caches result between
    ///   - Downsampled matching: clips indexed at matchingFps (e.g. 5) not full extraction fps
    ///   - All Compute() work is pure arithmetic on raw pointers, zero allocations
    /// </summary>
    public class ContinuingReward
    {
        private const float FALLEN_Z = 0.15f;
        private const float STANDING_Z = 0.7f;
        private const float TILT_THRESHOLD = 0.707f;
        private const float MOVING_VEL = 0.1f;

        private const float W_ALIVE = 0.05f;
        private const float W_HEIGHT = 0.20f;
        private const float W_ORIENTATION = 0.15f;
        private const float W_ENERGY = 0.03f;
        private const float W_ENERGY_FALLEN = 0.01f;
        private const float W_RECOVERY = 0.10f;
        private const float W_IMITATION = 0.12f;
        private const float W_VELOCITY_UP = 0.10f;
        private const float W_COMFORT = 0.10f;
        private const float COMFORT_STRAIN_SCALE = 2f;

        private const float W_FOOT_SUPPORT = 0.15f;
        private const float W_HAND_BRACE = 0.08f;
        private const float W_ACTIVE_SUPPORT = 0.07f;
        private const float SUPPORT_FORCE_THRESHOLD_FRAC = 0.05f;
        private const float PROXIMITY_SCALE = 10f;
        private const float PROXIMITY_WEIGHT = 0.6f;
        private const float CONTACT_WEIGHT = 0.4f;
        private const float GROUND_PROXIMITY_THRESHOLD = 0.05f;

        private const float PHASE_BONUS_RECOVERING = 0.15f;
        private const float PHASE_BONUS_STANDING = 0.40f;
        private const float PHASE_BONUS_MOVING = 0.50f;

        private const float UPRIGHT_SCALE = 5f;
        private const float ENERGY_SCALE = 0.1f;
        private const float IMITATION_SCALE = 2f;
        private const float RECOVERY_TARGET_DELTA = 0.01f;
        private const float CENTERING_ETA = 0.005f;
        private const int CENTERING_WARMUP = 10000;
        private const float MIN_IMIT_BLEND = 0.15f;

        private readonly float _standingZ;
        private readonly int[] _includedQposIdx;
        private readonly int[] _includedQvelIdx;
        private readonly int _nbody;

        // Contiguous reference frame data: [numFrames * nq] doubles
        private double[] _refQposFlat;
        private int _numReferenceFrames;
        private int _refNq;

        // Amortized nearest-frame search
        private int _nearestFrameInterval = 1;
        private int _stepsSinceSearch;
        private float _cachedNearestDist;

        private float _rewardBar;
        private bool _centeringInitialized;
        private int _stepCount;
        private float _prevRootZ;
        private bool _prevRootZInitialized;

        // Diagnostics (written every Compute, read by Inspector / Dashboard)
        private AgentPhase _lastPhase;
        private float _lastRawReward;
        private float _lastCenteredReward;
        private float _lastNearestFrameDist;
        private RewardSnapshot _lastSnapshot;

        public AgentPhase LastPhase => _lastPhase;
        public float LastRawReward => _lastRawReward;
        public float LastCenteredReward => _lastCenteredReward;
        public float LastNearestFrameDistance => _lastNearestFrameDist;
        public float RewardBar => _rewardBar;
        public ref readonly RewardSnapshot LastSnapshot => ref _lastSnapshot;
        public bool HasReferenceFrames => _numReferenceFrames > 0;
        public int ReferenceNq => _refNq;
        public int NumReferenceFrames => _numReferenceFrames;

        public ContinuingReward(
            float standingZ,
            int[] includedQposIdx,
            int[] includedQvelIdx,
            int nbody)
        {
            _standingZ = standingZ;
            _includedQposIdx = includedQposIdx ?? Array.Empty<int>();
            _includedQvelIdx = includedQvelIdx ?? Array.Empty<int>();
            _nbody = nbody;
            _refQposFlat = Array.Empty<double>();
        }

        public void SetNearestFrameInterval(int interval)
        {
            _nearestFrameInterval = Math.Max(1, interval);
        }

        /// <summary>
        /// Index motion clips with optional downsampling for matching.
        /// matchingFps &lt;= 0 means no downsampling (use every frame).
        /// </summary>
        public void IndexMotionClips(MotionReferenceData[] clips, float matchingFps = 0f)
        {
            int totalFrames = 0;
            int nq = 0;
            for (int c = 0; c < clips.Length; c++)
            {
                var clip = clips[c];
                int step = (matchingFps > 0 && matchingFps < clip.fps)
                    ? Math.Max(1, (int)(clip.fps / matchingFps + 0.5f))
                    : 1;
                totalFrames += (clip.frameCount + step - 1) / step;
                nq = clip.nq;
            }

            _refQposFlat = new double[totalFrames * nq];
            _refNq = nq;
            int offset = 0;
            for (int c = 0; c < clips.Length; c++)
            {
                var clip = clips[c];
                int step = (matchingFps > 0 && matchingFps < clip.fps)
                    ? Math.Max(1, (int)(clip.fps / matchingFps + 0.5f))
                    : 1;
                for (int f = 0; f < clip.frameCount; f += step)
                {
                    Buffer.BlockCopy(clip.qposFrames[f], 0, _refQposFlat,
                        offset * sizeof(double), nq * sizeof(double));
                    offset += nq;
                }
            }
            _numReferenceFrames = totalFrames;
            _stepsSinceSearch = 0;

            int framesWithVariance = ValidateFrameVariance();
            Debug.Log($"ContinuingReward: Indexed {totalFrames} frames from {clips.Length} clips " +
                $"(matchingFps={matchingFps:F0}, interval={_nearestFrameInterval}, " +
                $"uniquePoses={framesWithVariance}/{totalFrames})");

            if (framesWithVariance < 2)
                Debug.LogWarning("ContinuingReward: *** REFERENCE FRAMES HAVE NO VARIANCE! " +
                    "All frames appear identical (likely extraction failure). " +
                    "Imitation reward will provide no learning signal. " +
                    "Check Humanoid Avatar setup on both the Synth and the animation clips.");
        }

        /// <summary>
        /// Load pre-built reference index directly (from disk cache).
        /// </summary>
        public void LoadReferenceIndex(double[] flatData, int numFrames, int nq)
        {
            _refQposFlat = flatData;
            _numReferenceFrames = numFrames;
            _refNq = nq;
            _stepsSinceSearch = 0;
        }

        public void SaveReferenceIndex(BinaryWriter bw)
        {
            bw.Write(_numReferenceFrames);
            bw.Write(_refNq);
            int len = _numReferenceFrames * _refNq;
            for (int i = 0; i < len; i++)
                bw.Write(_refQposFlat[i]);
        }

        /// <summary>
        /// Count how many frames differ from frame 0 by more than a tiny threshold.
        /// Returns 0 if no frames, 1 if all identical, N if N distinct poses.
        /// </summary>
        public int ValidateFrameVariance()
        {
            if (_numReferenceFrames < 2 || _refNq == 0) return _numReferenceFrames;

            int nq = _refNq;
            int distinct = 1;
            for (int f = 1; f < _numReferenceFrames; f++)
            {
                double sumSqDiff = 0;
                int baseOff = 0;
                int frameOff = f * nq;
                for (int i = 7; i < nq; i++)
                {
                    double d = _refQposFlat[baseOff + i] - _refQposFlat[frameOff + i];
                    sumSqDiff += d * d;
                }
                if (sumSqDiff > 1e-6)
                    distinct++;
            }
            return distinct;
        }

        public bool LoadReferenceIndex(BinaryReader br)
        {
            _numReferenceFrames = br.ReadInt32();
            _refNq = br.ReadInt32();
            int len = _numReferenceFrames * _refNq;
            _refQposFlat = new double[len];
            for (int i = 0; i < len; i++)
                _refQposFlat[i] = br.ReadDouble();
            _stepsSinceSearch = 0;
            return true;
        }

        /// <summary>
        /// Compute the continuing reward. Overload without strain (backward compat).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe float Compute(MujocoLib.mjData_* data, MujocoLib.mjModel_* model)
        {
            return Compute(data, model, 0f, null, 0f);
        }

        /// <summary>
        /// Compute the continuing reward with proprioceptive strain/comfort component.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe float Compute(MujocoLib.mjData_* data, MujocoLib.mjModel_* model, float meanStrain)
        {
            return Compute(data, model, meanStrain, null, 0f);
        }

        /// <summary>
        /// Full reward computation with contact-based micro-rewards for recovery learning.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe float Compute(MujocoLib.mjData_* data, MujocoLib.mjModel_* model,
            float meanStrain, SynthContact contact, float bodyWeight)
        {
            double* pQpos = data->qpos;
            double* pQvel = data->qvel;

            float rootZ = (float)pQpos[2];
            float qx = (float)pQpos[4];
            float qy = (float)pQpos[5];
            float bodyUpZ = 1f - 2f * (qx * qx + qy * qy);

            float vx = (float)pQvel[0], vy = (float)pQvel[1];
            float vz = (float)pQvel[2];
            float horizVel = vx * vx + vy * vy;

            AgentPhase phase;
            if (rootZ < FALLEN_Z) phase = AgentPhase.Fallen;
            else if (rootZ < STANDING_Z) phase = AgentPhase.Recovering;
            else if (bodyUpZ < TILT_THRESHOLD) phase = AgentPhase.Recovering;
            else if (horizVel > MOVING_VEL * MOVING_VEL) phase = AgentPhase.Moving;
            else phase = AgentPhase.Standing;
            _lastPhase = phase;

            float rAlive = 0.05f;

            // Height reward: linear below standing (gradient everywhere), gaussian near standing
            float heightFraction = Mathf.Clamp01(rootZ / _standingZ);
            float rHeight;
            if (rootZ < STANDING_Z)
                rHeight = heightFraction;
            else
            {
                float d = rootZ - _standingZ;
                rHeight = Mathf.Exp(-UPRIGHT_SCALE * d * d);
            }

            // Orientation: linear from face-down (-1) to upright (+1) → [0, 1]
            float rOrientation = (bodyUpZ + 1f) * 0.5f;

            // Upward velocity: reward pushing off ground (only below standing height)
            float rVelocityUp = heightFraction < 0.9f ? Mathf.Clamp01(vz * 2f) : 0f;

            // Energy: reduced when fallen to not punish exploration
            float energyWeight = phase == AgentPhase.Fallen ? W_ENERGY_FALLEN : W_ENERGY;
            float rEnergy;
            {
                double* pCtrl = data->ctrl;
                int n = (int)model->nu;
                if (pCtrl == null || n == 0)
                {
                    rEnergy = 1f;
                }
                else
                {
                    float sumSq = 0f;
                    for (int i = 0; i < n; i++)
                    {
                        float c = (float)pCtrl[i];
                        sumSq += c * c;
                    }
                    rEnergy = Mathf.Exp(-ENERGY_SCALE * sumSq / n);
                }
            }

            float standBlend = Mathf.Clamp01((rootZ - FALLEN_Z) / (STANDING_Z - FALLEN_Z));
            if (bodyUpZ < TILT_THRESHOLD)
                standBlend *= Mathf.Clamp01((bodyUpZ - 0.3f) / (TILT_THRESHOLD - 0.3f));

            float rRecovery = 0f;
            if (_prevRootZInitialized)
            {
                float deltaZ = rootZ - _prevRootZ;
                rRecovery = Mathf.Clamp01(deltaZ / RECOVERY_TARGET_DELTA);
            }
            _prevRootZ = rootZ;
            _prevRootZInitialized = true;

            float rImitation = 0f;
            if (_numReferenceFrames > 0)
            {
                if (_stepsSinceSearch >= _nearestFrameInterval)
                {
                    _cachedNearestDist = FindNearestFrameDistanceFast(pQpos);
                    _stepsSinceSearch = 0;
                }
                _stepsSinceSearch++;
                _lastNearestFrameDist = _cachedNearestDist;
                rImitation = Mathf.Exp(-IMITATION_SCALE * _cachedNearestDist);
            }

            // Never fully zero out imitation — keep minimum blend even when fallen
            float effectiveImitBlend = Mathf.Max(standBlend, MIN_IMIT_BLEND);
            float recoveryWeight = W_RECOVERY * (1f - standBlend);
            float imitationWeight = W_IMITATION * effectiveImitBlend;
            float velocityWeight = W_VELOCITY_UP * (1f - standBlend);

            // Orientation always partially active — being upright while sitting/crouching
            // is progress worth rewarding. Minimum 30% weight even when fallen.
            const float MIN_ORIENT_BLEND = 0.3f;
            float effectiveOrientBlend = Mathf.Max(standBlend, MIN_ORIENT_BLEND);
            float orientWeight = W_ORIENTATION * effectiveOrientBlend;
            float freedWeight = W_ORIENTATION * (1f - effectiveOrientBlend);
            float heightBoost = freedWeight * 0.6f;
            float velBoost = freedWeight * 0.4f;

            // Comfort reward: exponential decay of mean strain.
            float rComfort = Mathf.Exp(-COMFORT_STRAIN_SCALE * meanStrain);

            // Proximity + contact hybrid rewards: gradient exists even without contact.
            // Proximity pulls limbs toward the ground; contact force rewards weight-bearing.
            float rFootSupport = 0f;
            float rHandBrace = 0f;
            float rActiveSupport = 0f;

            if (contact != null && bodyWeight > 1e-3f)
            {
                // Foot: proximity (always provides gradient) + contact force bonus
                float footZ_L = contact.GetBodyWorldZ(SynthContact.SLOT_LEFT_FOOT, data);
                float footZ_R = contact.GetBodyWorldZ(SynthContact.SLOT_RIGHT_FOOT, data);
                float footProx = 0.5f * (Mathf.Exp(-PROXIMITY_SCALE * Mathf.Max(0f, footZ_L))
                                       + Mathf.Exp(-PROXIMITY_SCALE * Mathf.Max(0f, footZ_R)));

                float footDown = contact.GetSupportForce(SynthContact.SLOT_LEFT_FOOT)
                               + contact.GetSupportForce(SynthContact.SLOT_RIGHT_FOOT);
                float footContact = Mathf.Clamp01(footDown / bodyWeight);

                rFootSupport = PROXIMITY_WEIGHT * footProx + CONTACT_WEIGHT * footContact;

                // Hand: same proximity + contact structure
                float handZ_L = contact.GetBodyWorldZ(SynthContact.SLOT_LEFT_HAND, data);
                float handZ_R = contact.GetBodyWorldZ(SynthContact.SLOT_RIGHT_HAND, data);
                float handProx = 0.5f * (Mathf.Exp(-PROXIMITY_SCALE * Mathf.Max(0f, handZ_L))
                                       + Mathf.Exp(-PROXIMITY_SCALE * Mathf.Max(0f, handZ_R)));

                float handDown = contact.GetSupportForce(SynthContact.SLOT_LEFT_HAND)
                               + contact.GetSupportForce(SynthContact.SLOT_RIGHT_HAND);
                float handContact = Mathf.Clamp01(handDown / (bodyWeight * 0.3f));

                rHandBrace = PROXIMITY_WEIGHT * handProx + CONTACT_WEIGHT * handContact;

                // Active support: limbs near ground OR bearing weight
                float threshold = bodyWeight * SUPPORT_FORCE_THRESHOLD_FRAC;
                int supportCount = 0;
                if (footZ_L < GROUND_PROXIMITY_THRESHOLD || contact.GetSupportForce(SynthContact.SLOT_LEFT_FOOT) > threshold) supportCount++;
                if (footZ_R < GROUND_PROXIMITY_THRESHOLD || contact.GetSupportForce(SynthContact.SLOT_RIGHT_FOOT) > threshold) supportCount++;
                if (handZ_L < GROUND_PROXIMITY_THRESHOLD || contact.GetSupportForce(SynthContact.SLOT_LEFT_HAND) > threshold) supportCount++;
                if (handZ_R < GROUND_PROXIMITY_THRESHOLD || contact.GetSupportForce(SynthContact.SLOT_RIGHT_HAND) > threshold) supportCount++;

                float kneeZ_L = contact.GetBodyWorldZ(SynthContact.SLOT_LEFT_KNEE, data);
                float kneeZ_R = contact.GetBodyWorldZ(SynthContact.SLOT_RIGHT_KNEE, data);
                if (kneeZ_L < GROUND_PROXIMITY_THRESHOLD || contact.GetSupportForce(SynthContact.SLOT_LEFT_KNEE) > threshold) supportCount++;
                if (kneeZ_R < GROUND_PROXIMITY_THRESHOLD || contact.GetSupportForce(SynthContact.SLOT_RIGHT_KNEE) > threshold) supportCount++;

                rActiveSupport = Mathf.Clamp01(supportCount / 3f);
            }

            float footSupportWeight = W_FOOT_SUPPORT;
            float handBraceWeight = W_HAND_BRACE * (1f - standBlend);
            float activeSupportWeight = W_ACTIVE_SUPPORT * (1f - standBlend);

            float phaseBonus = phase switch
            {
                AgentPhase.Recovering => PHASE_BONUS_RECOVERING,
                AgentPhase.Standing   => PHASE_BONUS_STANDING,
                AgentPhase.Moving     => PHASE_BONUS_MOVING,
                _                     => 0f
            };

            float rawReward = W_ALIVE * rAlive
                            + (W_HEIGHT + heightBoost) * rHeight
                            + orientWeight * rOrientation
                            + energyWeight * rEnergy
                            + recoveryWeight * rRecovery
                            + imitationWeight * rImitation
                            + (velocityWeight + velBoost) * rVelocityUp
                            + W_COMFORT * rComfort
                            + footSupportWeight * rFootSupport
                            + handBraceWeight * rHandBrace
                            + activeSupportWeight * rActiveSupport
                            + phaseBonus;

            _lastRawReward = rawReward;

            _lastSnapshot.Alive = W_ALIVE * rAlive;
            _lastSnapshot.Height = (W_HEIGHT + heightBoost) * rHeight;
            _lastSnapshot.Orientation = orientWeight * rOrientation;
            _lastSnapshot.Energy = energyWeight * rEnergy;
            _lastSnapshot.Recovery = recoveryWeight * rRecovery;
            _lastSnapshot.Imitation = imitationWeight * rImitation;
            _lastSnapshot.VelocityUp = (velocityWeight + velBoost) * rVelocityUp;
            _lastSnapshot.Comfort = W_COMFORT * rComfort;
            _lastSnapshot.FootSupport = footSupportWeight * rFootSupport;
            _lastSnapshot.HandBrace = handBraceWeight * rHandBrace;
            _lastSnapshot.ActiveSupport = activeSupportWeight * rActiveSupport;
            _lastSnapshot.PhaseBonus = phaseBonus;
            _lastSnapshot.StandBlend = standBlend;
            _lastSnapshot.RootZ = rootZ;
            _lastSnapshot.Phase = phase;

            _stepCount++;

            if (_stepCount <= CENTERING_WARMUP)
            {
                _rewardBar = rawReward;
                _lastCenteredReward = rawReward;
                _lastSnapshot.RawReward = rawReward;
                _lastSnapshot.CenteredReward = rawReward;
                _lastSnapshot.RewardBar = _rewardBar;
                return rawReward;
            }

            if (!_centeringInitialized)
            {
                _rewardBar = rawReward;
                _centeringInitialized = true;
            }
            else
            {
                _rewardBar += CENTERING_ETA * (rawReward - _rewardBar);
            }

            float centeredReward = rawReward - _rewardBar;
            _lastCenteredReward = centeredReward;
            _lastSnapshot.RawReward = rawReward;
            _lastSnapshot.CenteredReward = centeredReward;
            _lastSnapshot.RewardBar = _rewardBar;
            return centeredReward;
        }

        /// <summary>
        /// Nearest-frame search over contiguous flat array with early exit.
        /// Avoids jagged array indirection and bounds checks via fixed pointers.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private unsafe float FindNearestFrameDistanceFast(double* currentQpos)
        {
            int numFrames = _numReferenceFrames;
            int nq = _refNq;
            var inclIdx = _includedQposIdx;
            int numJoints = inclIdx.Length;
            if (numFrames == 0 || numJoints == 0) return 0f;

            float bestDist = float.MaxValue;

            fixed (double* pRef = _refQposFlat)
            fixed (int* pIdx = inclIdx)
            {
                double* framePtr = pRef;
                for (int f = 0; f < numFrames; f++)
                {
                    float dist = 0f;
                    for (int j = 0; j < numJoints; j++)
                    {
                        int qi = pIdx[j];
                        float diff = (float)(currentQpos[qi] - framePtr[qi]);
                        dist += diff * diff;
                        if (dist >= bestDist) goto nextFrame;
                    }

                    bestDist = dist;
                    nextFrame:
                    framePtr += nq;
                }
            }

            return bestDist < float.MaxValue ? Mathf.Sqrt(bestDist / numJoints) : 0f;
        }

        /// <summary>
        /// Copy a random reference frame's qpos into the output buffer.
        /// Returns false if no reference frames are indexed.
        /// </summary>
        public bool GetRandomReferenceQpos(Random rng, double[] outQpos)
        {
            if (_numReferenceFrames == 0 || _refNq == 0) return false;
            int frame = rng.Next(_numReferenceFrames);
            Buffer.BlockCopy(_refQposFlat, frame * _refNq * sizeof(double),
                outQpos, 0, _refNq * sizeof(double));
            return true;
        }

        public void Save(BinaryWriter bw)
        {
            bw.Write(_rewardBar);
            bw.Write(_centeringInitialized);
            bw.Write(_prevRootZ);
            bw.Write(_prevRootZInitialized);
            bw.Write(_stepCount);
        }

        public void Load(BinaryReader br)
        {
            _rewardBar = br.ReadSingle();
            _centeringInitialized = br.ReadBoolean();
            _prevRootZ = br.ReadSingle();
            _prevRootZInitialized = br.ReadBoolean();
            if (br.BaseStream.Position < br.BaseStream.Length)
                _stepCount = br.ReadInt32();
        }
    }
}
