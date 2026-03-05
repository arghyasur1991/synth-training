using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Animations;
using UnityEngine.Playables;
using Mujoco;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Extracts MotionReferenceData from any Unity Mecanim AnimationClip.
    /// 
    /// For cross-Avatar clips (e.g. Mixamo → Synth), uses a three-stage pipeline:
    ///   1. Evaluate the clip on the SOURCE FBX model's own Animator (guaranteed to work)
    ///   2. Transfer the pose to a clean Synth prefab via HumanPoseHandler (muscle space)
    ///   3. Read bone local rotations from the Synth and decompose to MuJoCo qpos
    /// 
    /// This avoids all cross-Avatar Playable API retargeting issues.
    /// Bone-to-qpos mappings and default rotations always come from the live MuJoCo hierarchy.
    /// </summary>
    public class MotionClipExtractor
    {
        private struct BoneJointMapping
        {
            public Transform bone;
            public Quaternion defaultLocalRotation;
            public int qposIndexX;
            public int qposIndexY;
            public int qposIndexZ;
        }

        private const string DEFAULT_SYNTH_PREFAB_PATH = "Assets/Sentience/Prefabs/Synth.prefab";

        /// <summary>
        /// Extract motion reference data from an AnimationClip.
        /// </summary>
        /// <param name="synthPrefabPath">
        /// Path to the Synth prefab for HumanPoseHandler retargeting. If null,
        /// attempts to find a SynthEntity in the scene and derive the prefab path
        /// from its Animator avatar. Falls back to the default path.
        /// </param>
        public unsafe MotionReferenceData Extract(
            AnimationClip clip,
            GameObject humanoidRoot,
            MujocoLib.mjModel_* sharedModel,
            float fps = 30f,
            bool isLooping = true,
            string[] keyBodyNames = null,
            string synthPrefabPath = null)
        {
            if (clip == null) throw new ArgumentNullException(nameof(clip));
            if (humanoidRoot == null) throw new ArgumentNullException(nameof(humanoidRoot));

            int nq = (int)sharedModel->nq;
            int nv = (int)sharedModel->nv;
            int nbody = (int)sharedModel->nbody;

            if (keyBodyNames == null)
                keyBodyNames = new string[] { "head", "lHand", "rHand", "lFoot", "rFoot" };

            // --- Step 1: Build bone-to-qpos mapping from MuJoCo hierarchy ---
            var mappings = BuildBoneJointMappings(humanoidRoot);
            var rootBone = FindRootBone(humanoidRoot);

            Debug.Log($"MotionClipExtractor: {mappings.Count} bones, root={rootBone?.name ?? "null"}, " +
                      $"clip='{clip.name}' (humanMotion={clip.humanMotion}), " +
                      $"duration={clip.length:F2}s, fps={fps}");

            // --- Step 2: Capture default pose from MuJoCo's qpos=0 ---
            var savedTransforms = SaveTransforms(humanoidRoot);

            var tempData = MujocoLib.mj_makeData(sharedModel);
            MujocoLib.mj_resetData(sharedModel, tempData);
            MujocoLib.mj_forward(sharedModel, tempData);
            double[] defaultQpos = new double[nq];
            for (int i = 0; i < nq; i++)
                defaultQpos[i] = tempData->qpos[i];

            if (MjScene.InstanceExists && MjScene.Instance.Data != null)
            {
                var sceneData = MjScene.Instance.Data;
                for (int i = 0; i < nq; i++)
                    sceneData->qpos[i] = defaultQpos[i];
                for (int i = 0; i < nv; i++)
                    sceneData->qvel[i] = 0.0;
                MujocoLib.mj_forward(MjScene.Instance.Model, sceneData);
                MjScene.Instance.SyncUnityToMjState();
            }

            mappings = BuildBoneJointMappings(humanoidRoot);

            // --- Step 3: Set up evaluation pipeline ---
            GameObject sourceInstance = null;
            GameObject synthInstance = null;
            Animator graphAnimator = null;
            HumanPoseHandler sourcePoseHandler = null;
            HumanPoseHandler synthPoseHandler = null;
            HumanPose humanPose = new HumanPose();
            bool useHumanPose = false;
            bool mjAnimatorWasEnabled = false;
            Dictionary<string, Transform> synthBoneLookup = null;

            #if UNITY_EDITOR
            // Strategy: evaluate clip on its SOURCE FBX model, transfer to Synth via HumanPoseHandler
            string clipAssetPath = AssetDatabase.GetAssetPath(clip);
            GameObject sourceModelAsset = null;
            if (!string.IsNullOrEmpty(clipAssetPath))
                sourceModelAsset = AssetDatabase.LoadAssetAtPath<GameObject>(clipAssetPath);

            string resolvedSynthPath = ResolveSynthPrefabPath(synthPrefabPath, humanoidRoot);
            var synthPrefab = AssetDatabase.LoadAssetAtPath<GameObject>(resolvedSynthPath);
            GameObject synthModelAsset = null;
            bool synthFromFBX = false;
            if (synthPrefab != null)
            {
                var prefabAnimator = synthPrefab.GetComponent<Animator>()
                    ?? synthPrefab.GetComponentInChildren<Animator>(true);
                if (prefabAnimator?.avatar != null)
                {
                    string avatarPath = AssetDatabase.GetAssetPath(prefabAnimator.avatar);
                    if (!string.IsNullOrEmpty(avatarPath))
                    {
                        synthModelAsset = AssetDatabase.LoadAssetAtPath<GameObject>(avatarPath);
                        if (synthModelAsset != null)
                        {
                            synthFromFBX = true;
                            Debug.Log($"MotionClipExtractor: Synth FBX from avatar: '{avatarPath}'");
                        }
                    }
                }
                if (synthModelAsset == null)
                {
                    synthModelAsset = synthPrefab;
                    Debug.Log("MotionClipExtractor: Synth FBX not found, using full prefab");
                }
            }

            if (sourceModelAsset != null && synthModelAsset != null)
            {
                sourceInstance = UnityEngine.Object.Instantiate(sourceModelAsset, Vector3.zero, Quaternion.identity);
                sourceInstance.name = "__ExtractorSource__";

                synthInstance = UnityEngine.Object.Instantiate(synthModelAsset, Vector3.zero, Quaternion.identity);
                synthInstance.name = "__ExtractorSynth__";

                foreach (var r in sourceInstance.GetComponentsInChildren<Renderer>())
                    r.enabled = false;
                foreach (var r in synthInstance.GetComponentsInChildren<Renderer>())
                    r.enabled = false;

                var srcAnimator = sourceInstance.GetComponent<Animator>()
                    ?? sourceInstance.GetComponentInChildren<Animator>(true);
                var snthAnimator = synthInstance.GetComponent<Animator>()
                    ?? synthInstance.GetComponentInChildren<Animator>(true);

                if (srcAnimator != null && srcAnimator.isHuman &&
                    snthAnimator != null && snthAnimator.isHuman)
                {
                    // Match ReferenceAnimationPlayer: prevent culling and auto-play interference
                    srcAnimator.cullingMode = AnimatorCullingMode.AlwaysAnimate;
                    srcAnimator.runtimeAnimatorController = null;
                    srcAnimator.speed = 0f;
                    srcAnimator.enabled = true;
                    srcAnimator.Rebind();

                    snthAnimator.cullingMode = AnimatorCullingMode.AlwaysAnimate;
                    snthAnimator.runtimeAnimatorController = null;
                    snthAnimator.speed = 0f;
                    snthAnimator.enabled = true;
                    snthAnimator.Rebind();

                    sourcePoseHandler = new HumanPoseHandler(srcAnimator.avatar, srcAnimator.transform);
                    synthPoseHandler = new HumanPoseHandler(snthAnimator.avatar, snthAnimator.transform);
                    graphAnimator = srcAnimator;
                    useHumanPose = true;

                    synthBoneLookup = new Dictionary<string, Transform>();
                    foreach (var t in synthInstance.GetComponentsInChildren<Transform>())
                        if (!synthBoneLookup.ContainsKey(t.name))
                            synthBoneLookup[t.name] = t;

                    Debug.Log($"MotionClipExtractor: HumanPoseHandler pipeline — " +
                        $"source='{srcAnimator.avatar?.name}' (isHuman={srcAnimator.isHuman}), " +
                        $"synth='{snthAnimator.avatar?.name}' (isHuman={snthAnimator.isHuman}), " +
                        $"synthBones={synthBoneLookup.Count}, synthFromFBX={synthFromFBX}");
                }
                else
                {
                    Debug.LogWarning($"MotionClipExtractor: Source or Synth not Humanoid — " +
                        $"src={srcAnimator?.isHuman}, synth={snthAnimator?.isHuman}. " +
                        $"Falling back to MuJoCo Animator.");
                    if (sourceInstance != null) UnityEngine.Object.DestroyImmediate(sourceInstance);
                    if (synthInstance != null) UnityEngine.Object.DestroyImmediate(synthInstance);
                    sourceInstance = null;
                    synthInstance = null;
                }
            }
            else
            {
                Debug.LogWarning($"MotionClipExtractor: Could not load source model " +
                    $"(path='{clipAssetPath ?? "null"}') or Synth model. Falling back.");
            }
            #endif

            // Fallback: MuJoCo Animator (works for same-Avatar / Generic clips)
            if (graphAnimator == null)
            {
                var animator = humanoidRoot.GetComponent<Animator>();
                if (animator == null)
                    animator = humanoidRoot.GetComponentInChildren<Animator>(true);
                if (animator == null)
                    animator = humanoidRoot.GetComponentInParent<Animator>(true);
                if (animator == null)
                {
                    var all = GameObject.FindObjectsByType<Animator>(
                        FindObjectsInactive.Include, FindObjectsSortMode.None);
                    if (all.Length > 0) animator = all[0];
                }
                if (animator == null)
                    throw new InvalidOperationException("MotionClipExtractor: No Animator found.");

                mjAnimatorWasEnabled = animator.enabled;
                animator.enabled = true;
                animator.Rebind();
                graphAnimator = animator;

                Debug.Log($"MotionClipExtractor: Fallback MuJoCo Animator '{animator.name}' " +
                    $"(isHuman={animator.isHuman})");
            }

            // Build eval bone array: synth instance bones matched by name to MuJoCo mappings
            Transform[] evalBones = new Transform[mappings.Count];
            int matchedBones = 0;
            for (int i = 0; i < mappings.Count; i++)
            {
                evalBones[i] = mappings[i].bone; // default: MuJoCo bone
                if (synthBoneLookup != null &&
                    synthBoneLookup.TryGetValue(mappings[i].bone.name, out var sb))
                {
                    evalBones[i] = sb;
                    matchedBones++;
                }
            }

            Transform evalRootBone = rootBone;
            if (synthBoneLookup != null && rootBone != null &&
                synthBoneLookup.TryGetValue(rootBone.name, out var synthRootT))
                evalRootBone = synthRootT;

            if (synthBoneLookup != null)
                Debug.Log($"MotionClipExtractor: Matched {matchedBones}/{mappings.Count} bones, " +
                    $"evalRoot={evalRootBone?.name ?? "null"}");

            // Playable graph on the graph animator (source model if HumanPose, MuJoCo otherwise)
            PlayableGraph graph = PlayableGraph.Create("MotionClipExtractor");
            var clipPlayable = AnimationClipPlayable.Create(graph, clip);
            var output = AnimationPlayableOutput.Create(graph, "ExtractorOutput", graphAnimator);
            output.SetSourcePlayable(clipPlayable);
            graph.SetTimeUpdateMode(DirectorUpdateMode.Manual);
            graph.Play();

            // --- Pre-loop diagnostic: verify animation evaluation works on source ---
            if (useHumanPose && sourceInstance != null)
            {
                var srcBones = sourceInstance.GetComponentsInChildren<Transform>();
                var srcDefaults = new Quaternion[srcBones.Length];
                for (int i = 0; i < srcBones.Length; i++)
                    srcDefaults[i] = srcBones[i].localRotation;

                float testTime = Mathf.Min(clip.length * 0.5f, 1f);
                clipPlayable.SetTime(testTime);
                graph.Evaluate(0f);

                int srcMoved = 0;
                for (int i = 0; i < srcBones.Length; i++)
                    if (Quaternion.Angle(srcDefaults[i], srcBones[i].localRotation) > 0.01f) srcMoved++;

                sourcePoseHandler.GetHumanPose(ref humanPose);
                int nonZeroMuscles = 0;
                float maxMuscle = 0f;
                for (int m = 0; m < humanPose.muscles.Length; m++)
                {
                    float abs = Mathf.Abs(humanPose.muscles[m]);
                    if (abs > 0.001f) nonZeroMuscles++;
                    if (abs > maxMuscle) maxMuscle = abs;
                }

                Debug.Log($"MotionClipExtractor: PRE-DIAG at t={testTime:F2}s — " +
                    $"srcBonesMoved={srcMoved}/{srcBones.Length}, " +
                    $"nonZeroMuscles={nonZeroMuscles}/{humanPose.muscles.Length}, " +
                    $"maxMuscle={maxMuscle:F4}, " +
                    $"bodyPos={humanPose.bodyPosition}, bodyRot={humanPose.bodyRotation.eulerAngles}");

                if (srcMoved == 0)
                {
                    Debug.LogWarning("MotionClipExtractor: Source bones didn't move after graph.Evaluate()! " +
                        "Trying Animator.Update(0) warm-up...");
                    graphAnimator.Update(0);
                    clipPlayable.SetTime(testTime);
                    graph.Evaluate(0f);

                    srcMoved = 0;
                    for (int i = 0; i < srcBones.Length; i++)
                        if (Quaternion.Angle(srcDefaults[i], srcBones[i].localRotation) > 0.01f) srcMoved++;
                    Debug.Log($"MotionClipExtractor: After warm-up retry — srcBonesMoved={srcMoved}/{srcBones.Length}");
                }

                // Transfer test and check Synth side (apply same Y-flip correction)
                Quaternion diagYFlip = Quaternion.Euler(0f, 180f, 0f);
                humanPose.bodyRotation = diagYFlip * humanPose.bodyRotation;
                humanPose.bodyPosition = diagYFlip * humanPose.bodyPosition;
                synthPoseHandler.SetHumanPose(ref humanPose);
                int synthMoved = 0;
                for (int bi = 0; bi < mappings.Count; bi++)
                {
                    if (Quaternion.Angle(mappings[bi].defaultLocalRotation, evalBones[bi].localRotation) > 0.01f)
                        synthMoved++;
                }
                Debug.Log($"MotionClipExtractor: PRE-DIAG synth — {synthMoved}/{mappings.Count} eval bones differ from default");

                // Reset graph for main loop
                clipPlayable.SetTime(0f);
                graph.Evaluate(0f);
            }

            // --- Step 4: Sample frames ---
            int frameCount = Mathf.Max(1, Mathf.CeilToInt(clip.length * fps) + 1);
            float frameDuration = 1f / fps;

            var qposFrames = new double[frameCount][];
            var qvelFrames = new double[frameCount][];
            var bodyPosFrames = new double[frameCount][];

            for (int f = 0; f < frameCount; f++)
            {
                float time = f * frameDuration;
                time = Mathf.Min(time, clip.length);

                // Evaluate animation on graph animator
                clipPlayable.SetTime(time);
                graph.Evaluate(0f);

                // Transfer pose to Synth via muscle space
                if (useHumanPose)
                {
                    sourcePoseHandler.GetHumanPose(ref humanPose);
                    // Correct 180° Y offset between Mixamo and Synth rest orientations
                    Quaternion yFlip = Quaternion.Euler(0f, 180f, 0f);
                    humanPose.bodyRotation = yFlip * humanPose.bodyRotation;
                    humanPose.bodyPosition = yFlip * humanPose.bodyPosition;
                    synthPoseHandler.SetHumanPose(ref humanPose);
                }

                qposFrames[f] = new double[nq];
                Array.Copy(defaultQpos, qposFrames[f], nq);
                bodyPosFrames[f] = new double[nbody * 3];

                if (evalRootBone != null)
                    ExtractRootQpos(evalRootBone, qposFrames[f]);

                for (int mi = 0; mi < mappings.Count; mi++)
                {
                    var mapping = mappings[mi];
                    Quaternion animLocalRot = evalBones[mi].localRotation;
                    Quaternion relRot = Quaternion.Inverse(mapping.defaultLocalRotation) * animLocalRot;
                    DecomposeXYZIntrinsic(relRot, out float qX, out float qY, out float qZ);

                    qposFrames[f][mapping.qposIndexX] = -qX;
                    qposFrames[f][mapping.qposIndexY] = -qY;
                    qposFrames[f][mapping.qposIndexZ] = -qZ;
                }

                if (f == 0 || f == frameCount - 1)
                {
                    int nonZero = 0;
                    for (int i = 7; i < nq; i++)
                        if (Math.Abs(qposFrames[f][i]) > 1e-6) nonZero++;
                    Debug.Log($"MotionClipExtractor: Frame {f}/{frameCount} t={time:F3}s " +
                              $"rootPos=({qposFrames[f][0]:F3},{qposFrames[f][1]:F3},{qposFrames[f][2]:F3}) " +
                              $"nonZeroJoints={nonZero}/{nq - 7}");
                }

                for (int i = 0; i < nq; i++)
                    tempData->qpos[i] = qposFrames[f][i];
                for (int i = 0; i < nv; i++)
                    tempData->qvel[i] = 0.0;

                MujocoLib.mj_forward(sharedModel, tempData);

                for (int b = 0; b < nbody; b++)
                {
                    bodyPosFrames[f][b * 3 + 0] = tempData->xpos[b * 3 + 0];
                    bodyPosFrames[f][b * 3 + 1] = tempData->xpos[b * 3 + 1];
                    bodyPosFrames[f][b * 3 + 2] = tempData->xpos[b * 3 + 2];
                }
            }

            // --- Cleanup ---
            graph.Destroy();
            sourcePoseHandler?.Dispose();
            synthPoseHandler?.Dispose();
            if (sourceInstance != null) UnityEngine.Object.DestroyImmediate(sourceInstance);
            if (synthInstance != null) UnityEngine.Object.DestroyImmediate(synthInstance);
            if (!useHumanPose) graphAnimator.enabled = mjAnimatorWasEnabled;

            // --- Step 5: Compute qvel via finite differences ---
            for (int f = 0; f < frameCount; f++)
                qvelFrames[f] = new double[nv];

            if (frameCount > 1)
            {
                float dt = frameDuration;
                for (int f = 0; f < frameCount; f++)
                {
                    int fPrev = isLooping ? (f - 1 + frameCount) % frameCount : Math.Max(0, f - 1);
                    int fNext = isLooping ? (f + 1) % frameCount : Math.Min(frameCount - 1, f + 1);

                    double[] qPrev = qposFrames[fPrev];
                    double[] qNext = qposFrames[fNext];
                    float dtDiff = (fNext != fPrev) ? (fNext - fPrev) * dt : dt;

                    if (nv >= 6)
                    {
                        for (int i = 0; i < 3; i++)
                            qvelFrames[f][i] = (qNext[i] - qPrev[i]) / dtDiff;
                        qvelFrames[f][3] = 0;
                        qvelFrames[f][4] = 0;
                        qvelFrames[f][5] = 0;
                    }

                    int hingeQposStart = 7;
                    int hingeQvelStart = 6;
                    int numHinges = nq - hingeQposStart;
                    for (int h = 0; h < numHinges; h++)
                    {
                        int qi = hingeQposStart + h;
                        int vi = hingeQvelStart + h;
                        qvelFrames[f][vi] = (qNext[qi] - qPrev[qi]) / dtDiff;
                    }
                }
            }

            // --- Step 6: Find key body indices ---
            int[] keyBodyIndices = FindKeyBodyIndices(humanoidRoot, keyBodyNames);

            // --- Step 7: Restore MuJoCo state ---
            MujocoLib.mj_deleteData(tempData);
            RestoreTransforms(humanoidRoot, savedTransforms);

            if (MjScene.InstanceExists && MjScene.Instance.Data != null)
            {
                var sceneData = MjScene.Instance.Data;
                MujocoLib.mj_resetData(MjScene.Instance.Model, sceneData);
                MujocoLib.mj_forward(MjScene.Instance.Model, sceneData);
                MjScene.Instance.SyncUnityToMjState();
            }

            var result = new MotionReferenceData
            {
                nq = nq, nv = nv, nbody = nbody, fps = fps,
                frameCount = frameCount, isLooping = isLooping,
                qposFrames = qposFrames, qvelFrames = qvelFrames,
                bodyPosFrames = bodyPosFrames, keyBodyIndices = keyBodyIndices
            };

            Debug.Log($"MotionClipExtractor: Done — {frameCount} frames at {fps}fps " +
                      $"({result.Duration:F2}s), nq={nq}, nv={nv}");

            return result;
        }

        #region Private helpers

        private static string ResolveSynthPrefabPath(string explicitPath, GameObject humanoidRoot)
        {
            if (!string.IsNullOrEmpty(explicitPath))
                return explicitPath;

            #if UNITY_EDITOR
            // Try to derive path from SynthEntity in scene
            var entity = humanoidRoot.GetComponent<Genesis.Sentience.Synth.SynthEntity>();
            if (entity == null)
                entity = humanoidRoot.GetComponentInParent<Genesis.Sentience.Synth.SynthEntity>();
            if (entity != null)
            {
                var animator = entity.GetComponent<Animator>()
                    ?? entity.GetComponentInChildren<Animator>(true);
                if (animator?.avatar != null)
                {
                    string prefabPath = UnityEditor.PrefabUtility.GetPrefabAssetPathOfNearestInstanceRoot(entity.gameObject);
                    if (!string.IsNullOrEmpty(prefabPath))
                    {
                        Debug.Log($"MotionClipExtractor: Resolved synth prefab from scene entity: {prefabPath}");
                        return prefabPath;
                    }
                }
            }
            #endif

            return DEFAULT_SYNTH_PREFAB_PATH;
        }

        private List<BoneJointMapping> BuildBoneJointMappings(GameObject root)
        {
            var mappings = new List<BoneJointMapping>();
            var allBones = root.GetComponentsInChildren<MjBody>();
            foreach (var mjBody in allBones)
            {
                var bone = mjBody.transform;
                string baseName = bone.name;
                var jointX = FindHingeJoint(bone, baseName + "JointX");
                var jointY = FindHingeJoint(bone, baseName + "JointY");
                var jointZ = FindHingeJoint(bone, baseName + "JointZ");

                if (jointX != null && jointY != null && jointZ != null)
                {
                    mappings.Add(new BoneJointMapping
                    {
                        bone = bone,
                        defaultLocalRotation = bone.localRotation,
                        qposIndexX = jointX.QposAddress,
                        qposIndexY = jointY.QposAddress,
                        qposIndexZ = jointZ.QposAddress
                    });
                }
            }
            return mappings;
        }

        private Transform FindRootBone(GameObject root)
        {
            var freeJoint = root.GetComponentInChildren<MjFreeJoint>();
            return freeJoint != null ? freeJoint.transform.parent : null;
        }

        private MjHingeJoint FindHingeJoint(Transform bone, string jointName)
        {
            for (int i = 0; i < bone.childCount; i++)
            {
                var child = bone.GetChild(i);
                if (child.name == jointName)
                    return child.GetComponent<MjHingeJoint>();
            }
            return null;
        }

        private void ExtractRootQpos(Transform rootBone, double[] qpos)
        {
            Vector3 pos = rootBone.position;
            qpos[0] = pos.x;
            qpos[1] = pos.z;
            qpos[2] = pos.y;

            Quaternion rot = rootBone.rotation;
            qpos[3] = -rot.w;
            qpos[4] = rot.x;
            qpos[5] = rot.z;
            qpos[6] = rot.y;
        }

        private static void DecomposeXYZIntrinsic(Quaternion q, out float qX, out float qY, out float qZ)
        {
            Matrix4x4 m = Matrix4x4.Rotate(q);
            float sinY = Mathf.Clamp(m.m02, -1f, 1f);
            qY = Mathf.Asin(sinY);
            float cosY = Mathf.Cos(qY);
            if (cosY > 1e-6f)
            {
                qX = Mathf.Atan2(-m.m12, m.m22);
                qZ = Mathf.Atan2(-m.m01, m.m00);
            }
            else
            {
                qX = Mathf.Atan2(m.m10, m.m11);
                qZ = 0f;
            }
        }

        private int[] FindKeyBodyIndices(GameObject root, string[] bodyNames)
        {
            var indices = new List<int>();
            var allBodies = root.GetComponentsInChildren<MjBody>();
            foreach (var name in bodyNames)
            {
                bool found = false;
                foreach (var body in allBodies)
                {
                    if (body.transform.name.Equals(name, StringComparison.OrdinalIgnoreCase) ||
                        body.transform.name.Contains(name))
                    {
                        indices.Add(body.MujocoId);
                        found = true;
                        break;
                    }
                }
                if (!found)
                    Debug.LogWarning($"MotionClipExtractor: Key body '{name}' not found");
            }
            return indices.ToArray();
        }

        private Dictionary<Transform, (Vector3 pos, Quaternion rot)> SaveTransforms(GameObject root)
        {
            var saved = new Dictionary<Transform, (Vector3, Quaternion)>();
            foreach (var t in root.GetComponentsInChildren<Transform>())
                saved[t] = (t.localPosition, t.localRotation);
            return saved;
        }

        private void RestoreTransforms(GameObject root, Dictionary<Transform, (Vector3 pos, Quaternion rot)> saved)
        {
            foreach (var kvp in saved)
            {
                if (kvp.Key != null)
                {
                    kvp.Key.localPosition = kvp.Value.pos;
                    kvp.Key.localRotation = kvp.Value.rot;
                }
            }
        }

        #endregion
    }
}
