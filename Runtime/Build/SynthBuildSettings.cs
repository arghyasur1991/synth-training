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

        [Tooltip("Source subdirectory under Application.persistentDataPath " +
                 "(must match BaseTrainingSkill.saveSubdirectory)")]
        public string sourceSubdirectory = "Training";

        [Tooltip("Only include models for these synth names. " +
                 "Empty = include all found models.")]
        public string[] synthFilter = { };

        [Tooltip("Log detailed file operations during build")]
        public bool verboseLogging = true;

        internal const string STREAMING_ASSETS_SUBDIR = "SynthModels";
        internal const string SETTINGS_RESOURCE_PATH = "SynthBuildSettings";

        public static SynthBuildSettings Load()
        {
            return Resources.Load<SynthBuildSettings>(SETTINGS_RESOURCE_PATH);
        }
    }
}
