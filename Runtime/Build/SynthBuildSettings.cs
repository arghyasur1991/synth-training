using UnityEngine;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Project-wide settings for packaging trained Synth models into builds.
    /// Create via Assets > Create > Synth > Build Settings, or auto-created
    /// by the build hook when none exists.
    /// </summary>
    [CreateAssetMenu(fileName = "SynthBuildSettings", menuName = "Synth/Build Settings")]
    public class SynthBuildSettings : ScriptableObject
    {
        [Tooltip("Include trained model checkpoints in builds (copies from " +
                 "persistentDataPath to StreamingAssets before build, cleans up after)")]
        public bool includeModelsInBuild = true;

        [Tooltip("Source subdirectories under Application.persistentDataPath " +
                 "(must match BaseTrainingSkill.saveSubdirectory). " +
                 "Empty = auto-discover all subdirectories containing models.")]
        public string[] sourceSubdirectories = { };

        [Tooltip("Only include models for these synth names. " +
                 "Empty = include all found models.")]
        public string[] synthFilter = { };

        [Tooltip("Remove copied models from StreamingAssets after build completes. " +
                 "Disable to keep them for inspection or manual deployment.")]
        public bool cleanUpAfterBuild = true;

        [Tooltip("Log detailed file operations during build")]
        public bool verboseLogging = true;

        public const string STREAMING_ASSETS_SUBDIR = "SynthModels";
        public const string SETTINGS_RESOURCE_PATH = "SynthBuildSettings";

        public static SynthBuildSettings Load()
        {
            var settings = Resources.Load<SynthBuildSettings>(SETTINGS_RESOURCE_PATH);
#if UNITY_EDITOR
            if (settings == null)
                settings = FindAnyInProject();
#endif
            return settings;
        }

#if UNITY_EDITOR
        private static SynthBuildSettings FindAnyInProject()
        {
            string[] guids = UnityEditor.AssetDatabase.FindAssets("t:SynthBuildSettings");
            if (guids.Length > 0)
            {
                string path = UnityEditor.AssetDatabase.GUIDToAssetPath(guids[0]);
                return UnityEditor.AssetDatabase.LoadAssetAtPath<SynthBuildSettings>(path);
            }
            return null;
        }
#endif
    }
}
