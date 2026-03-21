using System;
using System.IO;
using System.Runtime.CompilerServices;
using UnityEngine;
using Mujoco;
using Genesis.Sentience.Synth;
using Random = System.Random;

namespace Genesis.Sentience.Learning
{
    /// <summary>Per-component weighted reward values from the last V2 Compute() call.</summary>
    public struct RewardSnapshotV2
    {
        public float Height, Orientation, Contact, Energy, Imitation;
        public float Progress, RootZ;
        public float RawReward, CenteredReward, RewardBar;
    }

    /// <summary>
    /// PBT-configurable reward weights for V2. All fields are mutable
    /// so PopulationTrainer can read/write them during evolution.
    /// </summary>
    [Serializable]
    public struct RewardWeightsV2
    {
        public float Height;
        public float Orientation;
        public float Contact;
        public float Energy;
        public float Imitation;

        public static RewardWeightsV2 Default => new RewardWeightsV2
        {
            Height = 0.30f,
            Orientation = 0.20f,
            Contact = 0.25f,
            Energy = 0.05f,
            Imitation = 0.20f,
        };
    }

    /// <summary>
    /// V2 continuing reward: 5 terms, no discrete phases, continuous modulation
    /// via a learned progress signal from the StateEncoder.
    ///
    /// Compared to V1 (ContinuingReward):
    ///   - 5 terms instead of 12 (height, orientation, contact, energy, imitation)
    ///   - No AgentPhase enum, no discrete phase bonuses
    ///   - No proximity reward (encoder handles spatial awareness)
    ///   - Weight modulation via continuous progress ∈ [0,1] from encoder
    ///   - All base weights are PBT-configurable via RewardWeightsV2
    ///
    /// Keeps: reward centering, reference frame indexing, amortized nearest-frame search.
    /// </summary>
    public class ContinuingRewardV2
    {
        private const float ALIVE_BONUS = 0.02f;
        private const float UPRIGHT_SCALE = 5f;
        private const float ENERGY_SCALE = 0.1f;
        private const float IMITATION_SCALE = 2f;
        private const float CENTERING_ETA = 0.005f;
        private const int CENTERING_WARMUP = 10000;

        private readonly float _standingZ;
        private readonly int[] _includedQposIdx;

        // Reference frame data (same format as V1)
        private double[] _refQposFlat;
        private int _numReferenceFrames;
        private int _refNq;

        private int _nearestFrameInterval = 1;
        private int _stepsSinceSearch;
        private float _cachedNearestDist;

        private float _rewardBar;
        private bool _centeringInitialized;
        private int _stepCount;

        // Diagnostics
        private float _lastRawReward;
        private float _lastCenteredReward;
        private float _lastNearestFrameDist;
        private RewardSnapshotV2 _lastSnapshot;

        public float LastRawReward => _lastRawReward;
        public float LastCenteredReward => _lastCenteredReward;
        public float LastNearestFrameDistance => _lastNearestFrameDist;
        public float RewardBar => _rewardBar;
        public ref readonly RewardSnapshotV2 LastSnapshot => ref _lastSnapshot;
        public bool HasReferenceFrames => _numReferenceFrames > 0;
        public int NumReferenceFrames => _numReferenceFrames;
        public int ReferenceNq => _refNq;

        public ContinuingRewardV2(float standingZ, int[] includedQposIdx)
        {
            _standingZ = standingZ;
            _includedQposIdx = includedQposIdx ?? Array.Empty<int>();
            _refQposFlat = Array.Empty<double>();
        }

        public void SetNearestFrameInterval(int interval)
        {
            _nearestFrameInterval = Math.Max(1, interval);
        }

        /// <summary>
        /// Compute the V2 continuing reward.
        /// progress: learned continuous signal from StateEncoder, ∈ [0,1].
        /// weights: PBT-configurable base weights.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe float Compute(
            MujocoLib.mjData_* data,
            MujocoLib.mjModel_* model,
            float progress,
            in RewardWeightsV2 weights,
            SynthContact contact,
            float bodyWeight)
        {
            double* pQpos = data->qpos;
            double* pQvel = data->qvel;

            float rootZ = (float)pQpos[2];
            float qx = (float)pQpos[4];
            float qy = (float)pQpos[5];
            float bodyUpZ = 1f - 2f * (qx * qx + qy * qy);
            float vz = (float)pQvel[2];

            // --- Term 1: Height (merges old height + recovery + velocity_up) ---
            // Below standing: linear ramp + velocity bonus gives gradient to push up.
            // At/above standing: gaussian peak rewards maintaining height.
            float heightFraction = Mathf.Clamp01(rootZ / _standingZ);
            float rHeight;
            if (rootZ < _standingZ)
            {
                float velBonus = Mathf.Clamp01(vz * 2f) * (1f - heightFraction);
                rHeight = heightFraction + 0.3f * velBonus;
            }
            else
            {
                float d = rootZ - _standingZ;
                rHeight = Mathf.Exp(-UPRIGHT_SCALE * d * d);
            }

            // --- Term 2: Orientation ---
            float rOrientation = (bodyUpZ + 1f) * 0.5f;

            // --- Term 3: Contact force (unified, no proximity) ---
            float rContact = 0f;
            if (contact != null && bodyWeight > 1e-3f)
            {
                float footDown = contact.GetSupportForce(SynthContact.SLOT_LEFT_FOOT)
                               + contact.GetSupportForce(SynthContact.SLOT_RIGHT_FOOT);
                float handDown = contact.GetSupportForce(SynthContact.SLOT_LEFT_HAND)
                               + contact.GetSupportForce(SynthContact.SLOT_RIGHT_HAND);
                float kneeDown = contact.GetSupportForce(SynthContact.SLOT_LEFT_KNEE)
                               + contact.GetSupportForce(SynthContact.SLOT_RIGHT_KNEE);

                float footFrac = Mathf.Clamp01(footDown / bodyWeight);
                float handFrac = Mathf.Clamp01(handDown / (bodyWeight * 0.3f));
                float kneeFrac = Mathf.Clamp01(kneeDown / (bodyWeight * 0.5f));

                // Weight by progress: when low progress (fallen), hands/knees matter more;
                // when high progress (standing), feet matter more.
                float footW = Mathf.Lerp(0.3f, 0.7f, progress);
                float handW = Mathf.Lerp(0.4f, 0.1f, progress);
                float kneeW = Mathf.Lerp(0.3f, 0.2f, progress);

                rContact = footW * footFrac + handW * handFrac + kneeW * kneeFrac;
            }

            // --- Term 4: Energy ---
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

            // --- Term 5: Imitation ---
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

            // --- Continuous weight modulation via progress ---
            // Progress-based emphasis: when progress is low (fallen), amplify
            // height and contact (the getting-up signals). When high, amplify
            // orientation, imitation, energy (the refinement signals).
            float fallenEmphasis = 1f - progress;
            float wHeight = weights.Height * (1f + 0.8f * fallenEmphasis);
            float wOrientation = weights.Orientation * (0.3f + 0.7f * progress);
            float wContact = weights.Contact * (1f + 0.5f * fallenEmphasis);
            float wEnergy = weights.Energy * (0.3f + 0.7f * progress);
            float wImitation = weights.Imitation * Mathf.Max(0.15f, progress);

            float rawReward = ALIVE_BONUS
                            + wHeight * rHeight
                            + wOrientation * rOrientation
                            + wContact * rContact
                            + wEnergy * rEnergy
                            + wImitation * rImitation;

            _lastRawReward = rawReward;
            _lastSnapshot = new RewardSnapshotV2
            {
                Height = wHeight * rHeight,
                Orientation = wOrientation * rOrientation,
                Contact = wContact * rContact,
                Energy = wEnergy * rEnergy,
                Imitation = wImitation * rImitation,
                Progress = progress,
                RootZ = rootZ,
                RawReward = rawReward,
            };

            // --- Reward centering (same as V1) ---
            _stepCount++;
            if (_stepCount <= CENTERING_WARMUP)
            {
                _rewardBar = rawReward;
                _lastCenteredReward = rawReward;
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
            _lastSnapshot.CenteredReward = centeredReward;
            _lastSnapshot.RewardBar = _rewardBar;
            return centeredReward;
        }

        // ── Reference motion indexing (reused from V1) ──────────────────

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
            Debug.Log($"ContinuingRewardV2: Indexed {totalFrames} frames from {clips.Length} clips");
        }

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

        public bool GetRandomReferenceQpos(Random rng, double[] outQpos)
        {
            if (_numReferenceFrames == 0 || _refNq == 0) return false;
            int frame = rng.Next(_numReferenceFrames);
            Buffer.BlockCopy(_refQposFlat, frame * _refNq * sizeof(double),
                outQpos, 0, _refNq * sizeof(double));
            return true;
        }

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

        public void Save(BinaryWriter bw)
        {
            bw.Write(_rewardBar);
            bw.Write(_centeringInitialized);
            bw.Write(_stepCount);
        }

        public void Load(BinaryReader br)
        {
            _rewardBar = br.ReadSingle();
            _centeringInitialized = br.ReadBoolean();
            if (br.BaseStream.Position < br.BaseStream.Length)
                _stepCount = br.ReadInt32();
        }
    }
}
