using System;
using System.Linq;
using System.Collections.Generic;
using UnityEngine;
using Mujoco;
using Genesis.Sentience.Synth;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Standalone test bench for verifying the motion extraction pipeline.
    ///
    /// Supports two modes:
    ///   Single-clip mode: test one clip at a time
    ///   Multi-clip mode:  cycle through all clips from ContinuousLearningSkill
    ///                     to validate extraction quality en masse
    ///
    /// Extracts AnimationClips into MuJoCo joint angles (qpos) via MotionClipExtractor,
    /// then plays them back through MuJoCo forward kinematics + SyncUnityToMjState.
    ///
    /// No physics simulation runs — MjScene is paused immediately.
    /// Pipeline: qpos → mj_forward → SyncUnityToMjState → bone world transforms
    /// </summary>
    public class MotionExtractionTestBench : MonoBehaviour
    {
        [Header("Mode")]
        [Tooltip("Enable multi-clip mode: extracts and cycles through all clips from ContinuousLearningSkill.")]
        public bool multiClipMode = false;

        [Header("Single-Clip (when multiClipMode=false)")]
        [Tooltip("The animation clip to extract and play back through MuJoCo.")]
        public AnimationClip referenceClip;

        [Header("Extraction")]
        [Tooltip("Sampling rate for motion extraction (frames per second).")]
        public float extractionFps = 30f;

        [Tooltip("Whether clips loop.")]
        public bool clipIsLooping = true;

        [Tooltip("Key body names for end-effector tracking.")]
        public string[] keyBodyNames = new string[] { "head", "lHand", "rHand", "lFoot", "rFoot" };

        [Header("Playback")]
        [Tooltip("Playback speed multiplier (1 = realtime, 0 = paused). Ignored in scrub mode.")]
        [Range(0f, 2f)]
        public float playbackSpeed = 1f;

        [Header("Frame Scrubber")]
        [Tooltip("Enable scrub mode: manually control the frame via scrubFrame slider.")]
        public bool scrubMode = false;

        [Tooltip("Current frame index (drag to scrub).")]
        [Range(0, 9999)]
        public int scrubFrame = 0;

        [Header("Multi-Clip Controls")]
        [Tooltip("Current clip index in multi-clip mode.")]
        public int currentClipIndex = 0;

        [Tooltip("Auto-advance to next clip after playback duration.")]
        public bool autoAdvance = true;

        [Tooltip("Seconds to play each clip before auto-advancing.")]
        [Range(1f, 30f)]
        public float clipHoldTime = 4f;

        [Header("Debug Overrides")]
        [Tooltip("Force all hinge joint qpos to 0 (shows MuJoCo default T-pose).")]
        public bool zeroAllJoints = false;

        [Tooltip("Dump full qpos array to console for current frame.")]
        public bool dumpQpos = false;

        [Tooltip("Log bone comparison for the current frame.")]
        public bool logNextFrame = false;

        [Header("References")]
        [Tooltip("The Synth character (auto-found if null).")]
        public SynthEntity synthEntity;

        [Tooltip("Reference animation player for side-by-side (auto-found if null).")]
        public ReferenceAnimationPlayer referencePlayer;

        [Header("Diagnostics")]
        [Tooltip("Only log bones with angular error above this threshold (degrees).")]
        public float angleThreshold = 1.0f;

        [Tooltip("Log lightweight per-frame info.")]
        public bool logFrameInfo = false;

        // --- Runtime: current playback ---
        private MotionReferenceData extractedMotion;
        private double[] qposBuffer;
        private double[] qvelBuffer;
        private double[] bodyPosBuffer;
        private float currentTime;
        private int frameCounter;
        private bool isReady;
        private int lastAppliedScrubFrame = -1;
        private bool lastZeroAllJoints = false;
        private List<MjBody> mjBodies;

        // --- Runtime: multi-clip ---
        private AnimationClip[] allClips;
        private MotionReferenceData[] allExtracted;
        private ClipExtractionResult[] clipResults;
        private float clipPlayTimer;
        private int lastAppliedClipIndex = -1;

        private struct ClipExtractionResult
        {
            public string name;
            public int frameCount;
            public int jointsWithMotion;
            public int totalHingeJoints;
            public float maxAbsQpos;
            public bool passed;
        }

        public string CurrentClipName => multiClipMode && allClips != null && currentClipIndex < allClips.Length
            ? allClips[currentClipIndex]?.name ?? "null" : referenceClip?.name ?? "null";
        public int ClipCount => allClips?.Length ?? (referenceClip != null ? 1 : 0);
        public int PassedClips => clipResults?.Count(r => r.passed) ?? 0;
        public int FailedClips => clipResults?.Count(r => !r.passed) ?? 0;

        void Start()
        {
            if (MjScene.InstanceExists)
                MjScene.Instance.PauseSimulation = true;

            Initialize();
        }

        private unsafe void Initialize()
        {
            if (!MjScene.InstanceExists || MjScene.Instance.Model == null)
            {
                Debug.Log("TestBench: Waiting for MjScene...");
                Invoke(nameof(Initialize), 0.5f);
                return;
            }

            MjScene.Instance.PauseSimulation = true;

            if (synthEntity == null)
                synthEntity = GetComponent<SynthEntity>();
            if (synthEntity == null)
                synthEntity = FindObjectsByType<SynthEntity>(FindObjectsSortMode.None).FirstOrDefault();
            if (synthEntity == null)
            {
                Debug.LogError("TestBench: No SynthEntity found!");
                return;
            }

            var humanoidRoot = synthEntity.gameObject;
            var model = MjScene.Instance.Model;

            mjBodies = new List<MjBody>();
            var queue = new Queue<Transform>();
            queue.Enqueue(humanoidRoot.transform);
            while (queue.Count > 0)
            {
                var current = queue.Dequeue();
                var body = current.GetComponent<MjBody>();
                if (body != null) mjBodies.Add(body);
                for (int i = 0; i < current.childCount; i++)
                    queue.Enqueue(current.GetChild(i));
            }

            int nq = (int)model->nq;
            int nv = (int)model->nv;
            int nbody = (int)model->nbody;
            qposBuffer = new double[nq];
            qvelBuffer = new double[nv];
            bodyPosBuffer = new double[nbody * 3];

            if (multiClipMode)
                InitializeMultiClip(humanoidRoot, model);
            else
                InitializeSingleClip(humanoidRoot, model);
        }

        private unsafe void InitializeSingleClip(GameObject humanoidRoot, MujocoLib.mjModel_* model)
        {
            if (referenceClip == null)
            {
                Debug.LogError("TestBench: No referenceClip assigned!");
                return;
            }

            var extractor = new MotionClipExtractor();
            extractedMotion = extractor.Extract(
                referenceClip, humanoidRoot, model,
                extractionFps, clipIsLooping, keyBodyNames);

            SetupReferencePlayer();
            LogExtractionSummary(extractedMotion, referenceClip.name);

            currentTime = 0f;
            frameCounter = 0;
            scrubFrame = 0;
            lastAppliedScrubFrame = -1;
            isReady = true;

            Debug.Log($"TestBench: Ready (single) — {extractedMotion.frameCount} frames, " +
                $"{extractedMotion.Duration:F2}s, '{referenceClip.name}'");
            ApplyFrame(0f);
        }

        private unsafe void InitializeMultiClip(GameObject humanoidRoot, MujocoLib.mjModel_* model)
        {
            var skill = synthEntity.GetComponent<ContinuousLearningSkill>();
            if (skill == null)
                skill = synthEntity.GetComponentInChildren<ContinuousLearningSkill>(true);

            if (skill == null || skill.referenceClips == null || skill.referenceClips.Length == 0)
            {
                Debug.LogError("TestBench: Multi-clip mode requires ContinuousLearningSkill " +
                    "with referenceClips assigned on the Synth.");
                return;
            }

            allClips = skill.referenceClips.Where(c => c != null).ToArray();
            allExtracted = new MotionReferenceData[allClips.Length];
            clipResults = new ClipExtractionResult[allClips.Length];

            Debug.Log($"TestBench: Extracting {allClips.Length} clips...");

            var extractor = new MotionClipExtractor();
            int passCount = 0, failCount = 0;

            for (int i = 0; i < allClips.Length; i++)
            {
                allExtracted[i] = extractor.Extract(
                    allClips[i], humanoidRoot, model,
                    extractionFps, clipIsLooping, keyBodyNames);

                clipResults[i] = AnalyzeExtraction(allExtracted[i], allClips[i].name);

                if (clipResults[i].passed) passCount++; else failCount++;
            }

            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"TestBench: === MULTI-CLIP EXTRACTION SUMMARY ===");
            sb.AppendLine($"  Total: {allClips.Length}  |  PASS: {passCount}  |  FAIL: {failCount}");
            sb.AppendLine($"  {"#",4} {"Status",6} {"Joints",7} {"MaxRad",8} {"Frames",7} {"Name"}");
            sb.AppendLine($"  {"----",4} {"------",6} {"------",7} {"------",8} {"------",7} {"----"}");
            for (int i = 0; i < clipResults.Length; i++)
            {
                var r = clipResults[i];
                string status = r.passed ? "PASS" : "FAIL";
                sb.AppendLine($"  {i,4} {status,6} {r.jointsWithMotion,4}/{r.totalHingeJoints,-3}" +
                    $" {r.maxAbsQpos,8:F4} {r.frameCount,7} {r.name}");
            }
            Debug.Log(sb.ToString());

            if (failCount > 0)
            {
                sb.Clear();
                sb.AppendLine($"TestBench: FAILED CLIPS ({failCount}):");
                for (int i = 0; i < clipResults.Length; i++)
                    if (!clipResults[i].passed)
                        sb.AppendLine($"  [{i}] {clipResults[i].name}");
                Debug.LogWarning(sb.ToString());
            }

            SetupReferencePlayer();

            currentClipIndex = 0;
            lastAppliedClipIndex = -1;
            clipPlayTimer = 0f;
            SwitchToClip(0);

            isReady = true;
            Debug.Log($"TestBench: Ready (multi) — {allClips.Length} clips, " +
                $"{passCount} passed, {failCount} failed. " +
                $"autoAdvance={autoAdvance}, holdTime={clipHoldTime}s");
        }

        private ClipExtractionResult AnalyzeExtraction(MotionReferenceData data, string clipName)
        {
            int nq = data.nq;
            int hingeStart = 7;
            int totalHinges = nq - hingeStart;

            int jointsWithMotion = 0;
            float maxAbs = 0f;
            for (int j = hingeStart; j < nq; j++)
            {
                double minVal = double.MaxValue, maxVal = double.MinValue;
                for (int f = 0; f < data.frameCount; f++)
                {
                    double v = data.qposFrames[f][j];
                    if (v < minVal) minVal = v;
                    if (v > maxVal) maxVal = v;
                    float abs = Mathf.Abs((float)v);
                    if (abs > maxAbs) maxAbs = abs;
                }
                if (maxVal - minVal > 0.001) jointsWithMotion++;
            }

            return new ClipExtractionResult
            {
                name = clipName,
                frameCount = data.frameCount,
                jointsWithMotion = jointsWithMotion,
                totalHingeJoints = totalHinges,
                maxAbsQpos = maxAbs,
                passed = jointsWithMotion > 0
            };
        }

        private void SwitchToClip(int index)
        {
            if (allExtracted == null || index < 0 || index >= allExtracted.Length) return;

            currentClipIndex = index;
            extractedMotion = allExtracted[index];
            currentTime = 0f;
            frameCounter = 0;
            scrubFrame = 0;
            lastAppliedScrubFrame = -1;
            clipPlayTimer = 0f;

            if (referencePlayer != null && referencePlayer.IsReady && allClips != null)
            {
                referencePlayer.referenceClip = allClips[index];
                referencePlayer.Init();
            }

            var r = clipResults[index];
            string status = r.passed ? "PASS" : "FAIL";
            Debug.Log($"TestBench: [{index}/{allClips.Length}] '{r.name}' — " +
                $"{status}, {r.frameCount} frames, {r.jointsWithMotion}/{r.totalHingeJoints} joints");

            ApplyFrame(0f);
            lastAppliedClipIndex = index;
        }

        private void SetupReferencePlayer()
        {
            if (referencePlayer == null)
                referencePlayer = FindObjectsByType<ReferenceAnimationPlayer>(FindObjectsSortMode.None).FirstOrDefault();
            if (referencePlayer != null)
            {
                if (referencePlayer.referenceClip == null && referenceClip != null)
                    referencePlayer.referenceClip = referenceClip;
                if (!referencePlayer.IsReady)
                    referencePlayer.Init();
                Debug.Log($"TestBench: ReferenceAnimationPlayer on '{referencePlayer.gameObject.name}'");
            }
        }

        unsafe void FixedUpdate()
        {
            if (!isReady || extractedMotion == null) return;

            var scene = MjScene.Instance;
            if (scene == null || scene.Model == null || scene.Data == null) return;

            if (multiClipMode && allExtracted != null)
            {
                if (currentClipIndex != lastAppliedClipIndex)
                {
                    currentClipIndex = Mathf.Clamp(currentClipIndex, 0, allExtracted.Length - 1);
                    SwitchToClip(currentClipIndex);
                    return;
                }

                if (autoAdvance && !scrubMode)
                {
                    clipPlayTimer += Time.fixedDeltaTime;
                    if (clipPlayTimer >= clipHoldTime)
                    {
                        int next = (currentClipIndex + 1) % allExtracted.Length;
                        SwitchToClip(next);
                        return;
                    }
                }
            }

            if (scrubMode)
            {
                scrubFrame = Mathf.Clamp(scrubFrame, 0, extractedMotion.frameCount - 1);

                bool needsUpdate = scrubFrame != lastAppliedScrubFrame
                    || zeroAllJoints != lastZeroAllJoints
                    || dumpQpos || logNextFrame;

                if (needsUpdate)
                {
                    float time = scrubFrame / extractedMotion.fps;
                    ApplyFrame(time);
                    lastAppliedScrubFrame = scrubFrame;
                    lastZeroAllJoints = zeroAllJoints;
                }
            }
            else
            {
                frameCounter++;
                currentTime += Time.fixedDeltaTime * playbackSpeed;

                if (extractedMotion.isLooping && extractedMotion.Duration > 0)
                {
                    currentTime %= extractedMotion.Duration;
                    if (currentTime < 0) currentTime += extractedMotion.Duration;
                }
                else
                {
                    currentTime = Mathf.Clamp(currentTime, 0f, extractedMotion.Duration);
                }

                ApplyFrame(currentTime);
            }
        }

        private unsafe void ApplyFrame(float time)
        {
            var scene = MjScene.Instance;
            var model = scene.Model;
            var data = scene.Data;
            int nq = (int)model->nq;
            int nv = (int)model->nv;

            extractedMotion.GetFrameAtTime(time, qposBuffer, qvelBuffer, bodyPosBuffer);

            if (zeroAllJoints)
            {
                for (int i = 7; i < qposBuffer.Length; i++)
                    qposBuffer[i] = 0.0;
            }

            for (int i = 0; i < nq && i < qposBuffer.Length; i++)
                data->qpos[i] = qposBuffer[i];
            for (int i = 0; i < nv && i < qvelBuffer.Length; i++)
                data->qvel[i] = qvelBuffer[i];

            MujocoLib.mj_forward(model, data);
            scene.SyncUnityToMjState();

            if (referencePlayer != null && referencePlayer.IsReady)
                referencePlayer.SetTimeAndEvaluate(time);

            if (dumpQpos)
            {
                DumpQposToLog(time);
                dumpQpos = false;
            }

            if (logNextFrame)
            {
                LogBoneComparison(time);
                logNextFrame = false;
            }

            if (logFrameInfo)
            {
                int frameIdx = Mathf.FloorToInt(time * extractedMotion.fps);
                float phase = extractedMotion.GetPhase(time);
                string jointSample = "";
                int hingeStart = 7;
                int numSample = Mathf.Min(6, qposBuffer.Length - hingeStart);
                for (int i = 0; i < numSample; i++)
                    jointSample += $"{qposBuffer[hingeStart + i]:F3} ";
                Debug.Log($"[TestBench] t={time:F3}s frame={frameIdx}/{extractedMotion.frameCount} " +
                          $"phase={phase:F2} joints[7:13]=[{jointSample.TrimEnd()}]");
            }
        }

        private void LogExtractionSummary(MotionReferenceData data, string clipName)
        {
            if (data == null || data.frameCount == 0) return;

            int nq = data.nq;
            int hingeStart = 7;
            var frame0 = data.qposFrames[0];
            int nonZeroCount = 0;
            float maxAbsVal = 0f;
            int maxAbsIdx = 0;
            for (int i = hingeStart; i < nq; i++)
            {
                float absVal = Mathf.Abs((float)frame0[i]);
                if (absVal > 0.001f) nonZeroCount++;
                if (absVal > maxAbsVal) { maxAbsVal = absVal; maxAbsIdx = i; }
            }

            int jointsWithMotion = 0;
            for (int j = hingeStart; j < nq; j++)
            {
                double minVal = double.MaxValue, maxVal = double.MinValue;
                for (int f = 0; f < data.frameCount; f++)
                {
                    double v = data.qposFrames[f][j];
                    if (v < minVal) minVal = v;
                    if (v > maxVal) maxVal = v;
                }
                if (maxVal - minVal > 0.001) jointsWithMotion++;
            }

            Debug.Log($"TestBench: === EXTRACTION SUMMARY for '{clipName}' ===\n" +
                      $"  qpos dim: {nq} (root: 7, hinges: {nq - 7})\n" +
                      $"  Frame 0: {nonZeroCount}/{nq - 7} non-zero, max |qpos| = {maxAbsVal:F4} rad at [{maxAbsIdx}]\n" +
                      $"  All {data.frameCount} frames: {jointsWithMotion}/{nq - 7} joints with motion\n" +
                      $"  Root pos: ({frame0[0]:F3}, {frame0[1]:F3}, {frame0[2]:F3})");
        }

        private void DumpQposToLog(float time)
        {
            int frameIdx = Mathf.FloorToInt(time * extractedMotion.fps);
            var sb = new System.Text.StringBuilder();
            sb.AppendLine($"=== QPOS DUMP at t={time:F4}s (frame {frameIdx}) ===");
            sb.AppendLine($"  Root pos   [0:2]:  ({qposBuffer[0]:F4}, {qposBuffer[1]:F4}, {qposBuffer[2]:F4})");
            sb.AppendLine($"  Root quat  [3:6]:  ({qposBuffer[3]:F4}, {qposBuffer[4]:F4}, {qposBuffer[5]:F4}, {qposBuffer[6]:F4})");

            int idx = 7;
            int boneIdx = 0;
            while (idx < qposBuffer.Length)
            {
                if (idx + 2 < qposBuffer.Length)
                {
                    float xDeg = (float)qposBuffer[idx] * Mathf.Rad2Deg;
                    float yDeg = (float)qposBuffer[idx + 1] * Mathf.Rad2Deg;
                    float zDeg = (float)qposBuffer[idx + 2] * Mathf.Rad2Deg;
                    bool allZero = Mathf.Abs(xDeg) < 0.01f && Mathf.Abs(yDeg) < 0.01f && Mathf.Abs(zDeg) < 0.01f;
                    string marker = allZero ? " [ZERO]" : "";
                    sb.AppendLine($"  Joint {boneIdx,2} [{idx,3}:{idx + 2,3}]: " +
                                  $"({qposBuffer[idx],8:F4}, {qposBuffer[idx + 1],8:F4}, {qposBuffer[idx + 2],8:F4}) rad  " +
                                  $"= ({xDeg,7:F2}°, {yDeg,7:F2}°, {zDeg,7:F2}°){marker}");
                    idx += 3;
                    boneIdx++;
                }
                else
                {
                    for (int i = idx; i < qposBuffer.Length; i++)
                        sb.AppendLine($"  qpos[{i}] = {qposBuffer[i]:F6}");
                    break;
                }
            }
            Debug.Log(sb.ToString());
        }

        private void LogBoneComparison(float time)
        {
            var refRots = referencePlayer?.BoneWorldRotations;
            var refPoss = referencePlayer?.BoneWorldPositions;
            bool hasRef = refRots != null && refRots.Count > 0;

            Debug.Log($"╔══════════════════════════════════════════════════════════════════╗");
            Debug.Log($"║ BONE COMPARISON at t={time:F4}s  frame#{frameCounter}  hasRef={hasRef}");
            Debug.Log($"╠══════════════════════════════════════════════════════════════════╣");

            if (hasRef)
                Debug.Log($"  {"Bone",-28} {"SynthEuler",30} {"RefEuler",30} {"Err°",7}");
            else
                Debug.Log($"  {"Bone",-28} {"SynthEuler",30}");

            float maxErr = 0f;
            string worstBone = "";
            int errCount = 0;
            int totalBones = 0;

            foreach (var body in mjBodies)
            {
                var t = body.transform;
                totalBones++;

                Quaternion synthRot = t.rotation;
                Vector3 synthEuler = synthRot.eulerAngles;

                if (hasRef && refRots.TryGetValue(t.name, out Quaternion refRot))
                {
                    Vector3 refEuler = refRot.eulerAngles;
                    float err = Quaternion.Angle(synthRot, refRot);

                    if (err > maxErr) { maxErr = err; worstBone = t.name; }
                    if (err >= angleThreshold) errCount++;

                    if (err >= angleThreshold || angleThreshold <= 0.5f)
                    {
                        Debug.Log($"  {t.name,-28} " +
                                  $"({synthEuler.x,7:F1},{synthEuler.y,7:F1},{synthEuler.z,7:F1}) " +
                                  $"({refEuler.x,7:F1},{refEuler.y,7:F1},{refEuler.z,7:F1}) " +
                                  $"{err,7:F2}");
                    }
                }
                else
                {
                    Debug.Log($"  {t.name,-28} " +
                              $"({synthEuler.x,7:F1},{synthEuler.y,7:F1},{synthEuler.z,7:F1})" +
                              (hasRef ? "  [NO REF MATCH]" : ""));
                }
            }

            if (hasRef && refPoss != null)
            {
                int posErrCount = 0;
                float maxPosErr = 0f;
                string worstPosBone = "";
                foreach (var body in mjBodies)
                {
                    var t = body.transform;
                    if (refPoss.TryGetValue(t.name, out Vector3 refPos))
                    {
                        float dist = Vector3.Distance(t.position, refPos);
                        if (dist > maxPosErr) { maxPosErr = dist; worstPosBone = t.name; }
                        if (dist > 0.01f) posErrCount++;
                    }
                }
                Debug.Log($"  [POS] {posErrCount}/{totalBones} bones >1cm off. " +
                          $"Worst: '{worstPosBone}' {maxPosErr:F4}m");
            }

            Debug.Log($"╠══════════════════════════════════════════════════════════════════╣");
            if (hasRef)
                Debug.Log($"║ SUMMARY: {errCount}/{totalBones} bones >= {angleThreshold}° error. " +
                          $"Max: {maxErr:F2}° on '{worstBone}'");
            else
                Debug.Log($"║ SUMMARY: {totalBones} MjBody bones (no reference for comparison)");
            Debug.Log($"╚══════════════════════════════════════════════════════════════════╝");
        }

        [ContextMenu("Next Clip")]
        public void NextClip()
        {
            if (!multiClipMode || allExtracted == null) return;
            int next = (currentClipIndex + 1) % allExtracted.Length;
            SwitchToClip(next);
        }

        [ContextMenu("Previous Clip")]
        public void PreviousClip()
        {
            if (!multiClipMode || allExtracted == null) return;
            int prev = (currentClipIndex - 1 + allExtracted.Length) % allExtracted.Length;
            SwitchToClip(prev);
        }

        [ContextMenu("Stop Test")]
        public void StopTest()
        {
            isReady = false;
            if (MjScene.InstanceExists)
                MjScene.Instance.PauseSimulation = false;
            Debug.Log("TestBench: Stopped.");
        }

        [ContextMenu("Restart Test")]
        public void RestartTest()
        {
            isReady = false;
            currentTime = 0f;
            frameCounter = 0;
            scrubFrame = 0;
            lastAppliedScrubFrame = -1;
            lastAppliedClipIndex = -1;
            extractedMotion = null;
            allExtracted = null;
            clipResults = null;
            Initialize();
        }

        [ContextMenu("Dump Qpos")]
        public void DumpCurrentQpos()
        {
            dumpQpos = true;
        }

        [ContextMenu("Log Bone Comparison")]
        public void LogCurrentBoneComparison()
        {
            logNextFrame = true;
        }

        [ContextMenu("Log Multi-Clip Summary")]
        public void LogMultiClipSummary()
        {
            if (clipResults == null) { Debug.Log("TestBench: No multi-clip data."); return; }

            var sb = new System.Text.StringBuilder();
            int pass = 0, fail = 0;
            for (int i = 0; i < clipResults.Length; i++)
                if (clipResults[i].passed) pass++; else fail++;

            sb.AppendLine($"TestBench: {pass} PASS / {fail} FAIL out of {clipResults.Length} clips");
            if (fail > 0)
            {
                sb.AppendLine("Failed:");
                for (int i = 0; i < clipResults.Length; i++)
                    if (!clipResults[i].passed)
                        sb.AppendLine($"  [{i}] {clipResults[i].name}");
            }
            Debug.Log(sb.ToString());
        }
    }
}
