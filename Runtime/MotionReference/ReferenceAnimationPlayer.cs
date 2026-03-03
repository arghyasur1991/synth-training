using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Animations;
using UnityEngine.Playables;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Drives a reference character's animation using the Playable API.
    ///
    /// Attach this to the REFERENCE character (the non-MuJoCo copy). It will:
    ///   1. Create a PlayableGraph for the given AnimationClip
    ///   2. Sample the clip at any requested time via SetTimeAndEvaluate()
    ///   3. Store bone world rotations/positions by name for side-by-side comparison
    ///   4. Log per-bone diagnostics when requested
    ///
    /// Driven externally by MotionExtractionTestBench, trainers, or any
    /// controller that calls SetTimeAndEvaluate(time) each frame.
    /// </summary>
    public class ReferenceAnimationPlayer : MonoBehaviour
    {
        [Header("Animation")]
        [Tooltip("The animation clip to play on the reference character.")]
        public AnimationClip referenceClip;

        [Tooltip("Whether the clip loops")]
        public bool clipIsLooping = true;

        [Header("Diagnostics")]
        [Tooltip("Log all bone rotations each frame (very verbose)")]
        public bool logEveryFrame = false;

        [Tooltip("Log bone rotations once at the next frame, then auto-disable")]
        public bool logNextFrame = false;

        public float CurrentTime { get; private set; }
        public bool IsReady { get; private set; }
        public Dictionary<string, Quaternion> BoneWorldRotations => boneWorldRotations;
        public Dictionary<string, Vector3> BoneWorldPositions => boneWorldPositions;
        public Dictionary<string, Transform> BoneTransforms => boneTransformsByName;

        private PlayableGraph graph;
        private AnimationClipPlayable clipPlayable;
        private Animator animator;
        private Dictionary<string, Quaternion> boneWorldRotations = new Dictionary<string, Quaternion>();
        private Dictionary<string, Vector3> boneWorldPositions = new Dictionary<string, Vector3>();
        private Dictionary<string, Transform> boneTransformsByName = new Dictionary<string, Transform>();
        private Transform[] allBones;

        private Dictionary<string, Quaternion> defaultWorldRotations = new Dictionary<string, Quaternion>();
        private Dictionary<string, Vector3> defaultWorldPositions = new Dictionary<string, Vector3>();

        /// <summary>
        /// Initialize the Playable API graph and capture default bone transforms.
        /// Disables the Animator's own controller so only the manual PlayableGraph
        /// drives the bones. The character will freeze until SetTimeAndEvaluate() is called.
        /// </summary>
        public void Init()
        {
            if (referenceClip == null)
            {
                Debug.LogError("ReferenceAnimationPlayer: No referenceClip assigned!");
                return;
            }

            animator = GetComponent<Animator>();
            if (animator == null)
                animator = GetComponentInChildren<Animator>(true);
            if (animator == null)
            {
                Debug.LogError("ReferenceAnimationPlayer: No Animator found on reference character!");
                return;
            }

            animator.enabled = true;
            animator.Rebind();
            animator.runtimeAnimatorController = null;
            animator.speed = 0f;

            allBones = GetComponentsInChildren<Transform>(true);
            boneTransformsByName.Clear();
            foreach (var bone in allBones)
            {
                if (!boneTransformsByName.ContainsKey(bone.name))
                    boneTransformsByName[bone.name] = bone;
            }

            defaultWorldRotations.Clear();
            defaultWorldPositions.Clear();
            foreach (var bone in allBones)
            {
                defaultWorldRotations[bone.name] = bone.rotation;
                defaultWorldPositions[bone.name] = bone.position;
            }

            if (graph.IsValid()) graph.Destroy();
            graph = PlayableGraph.Create("ReferenceAnimationPlayer");
            clipPlayable = AnimationClipPlayable.Create(graph, referenceClip);
            var output = AnimationPlayableOutput.Create(graph, "RefOutput", animator);
            output.SetSourcePlayable(clipPlayable);
            graph.SetTimeUpdateMode(DirectorUpdateMode.Manual);

            IsReady = true;
            Debug.Log($"ReferenceAnimationPlayer: Initialized on '{gameObject.name}' " +
                      $"with clip '{referenceClip.name}', {allBones.Length} bones, " +
                      $"animator='{animator.name}', isHuman={animator.isHuman}");
        }

        /// <summary>
        /// Set the animation time and evaluate. Called by the driving controller each frame.
        /// </summary>
        public void SetTimeAndEvaluate(float time)
        {
            if (!IsReady) return;

            CurrentTime = time;

            float duration = referenceClip.length;
            if (clipIsLooping && duration > 0)
            {
                time %= duration;
                if (time < 0) time += duration;
            }
            else
            {
                time = Mathf.Clamp(time, 0f, duration);
            }

            clipPlayable.SetTime(time);
            graph.Evaluate();

            UpdateBoneDictionaries();

            if (logEveryFrame || logNextFrame)
            {
                LogAllBones(time);
                logNextFrame = false;
            }
        }

        private void UpdateBoneDictionaries()
        {
            boneWorldRotations.Clear();
            boneWorldPositions.Clear();
            foreach (var bone in allBones)
            {
                boneWorldRotations[bone.name] = bone.rotation;
                boneWorldPositions[bone.name] = bone.position;
            }
        }

        public void LogAllBones(float time)
        {
            Debug.Log($"=== ReferenceAnimationPlayer bones at t={time:F4}s ===");
            foreach (var bone in allBones)
            {
                Vector3 euler = bone.rotation.eulerAngles;
                Quaternion defRot = defaultWorldRotations.ContainsKey(bone.name)
                    ? defaultWorldRotations[bone.name] : Quaternion.identity;
                float angleDiff = Quaternion.Angle(defRot, bone.rotation);

                if (angleDiff > 0.01f)
                {
                    Debug.Log($"  [REF] {bone.name,-30} euler=({euler.x,7:F2}, {euler.y,7:F2}, {euler.z,7:F2}) " +
                              $"diffFromDefault={angleDiff,6:F2}° pos=({bone.position.x:F3},{bone.position.y:F3},{bone.position.z:F3})");
                }
            }
        }

        public float GetAngleFromDefault(string boneName)
        {
            if (!boneWorldRotations.ContainsKey(boneName) || !defaultWorldRotations.ContainsKey(boneName))
                return -1f;
            return Quaternion.Angle(defaultWorldRotations[boneName], boneWorldRotations[boneName]);
        }

        void OnDestroy()
        {
            if (graph.IsValid()) graph.Destroy();
        }
    }
}
