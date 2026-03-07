using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Pre-build: copies trained model checkpoints from persistentDataPath
    /// into StreamingAssets so they ship with the build.
    /// Post-build: removes the temporary StreamingAssets copy to keep the
    /// working tree clean.
    /// </summary>
    public class SynthModelBuildProcessor : IPreprocessBuildWithReport, IPostprocessBuildWithReport
    {
        public int callbackOrder => 0;

        private static string StreamingDest =>
            Path.Combine(Application.streamingAssetsPath, SynthBuildSettings.STREAMING_ASSETS_SUBDIR);

        public void OnPreprocessBuild(BuildReport report)
        {
            var settings = SynthBuildSettings.Load();
            if (settings == null)
            {
                settings = EnsureSettingsAsset();
                if (settings == null) return;
            }
            if (!settings.includeModelsInBuild)
                return;

            string[] sourceSubdirs = DiscoverSourceDirectories(settings);
            if (sourceSubdirs.Length == 0)
            {
                Debug.LogWarning("[SynthBuild] No trained model directories found " +
                    "— building without models.");
                return;
            }

            var filter = settings.synthFilter;
            bool hasFilter = filter != null && filter.Length > 0 &&
                             filter.Any(f => !string.IsNullOrWhiteSpace(f));

            string dest = StreamingDest;
            if (Directory.Exists(dest))
                Directory.Delete(dest, true);
            Directory.CreateDirectory(dest);

            int copiedSynths = 0;
            int copiedFiles = 0;

            foreach (string sourceRoot in sourceSubdirs)
            {
                string subdirName = Path.GetFileName(sourceRoot);
                string[] synthDirs = Directory.GetDirectories(sourceRoot);

                foreach (string synthDir in synthDirs)
                {
                    string synthName = Path.GetFileName(synthDir);

                    if (hasFilter && !filter.Any(f =>
                        string.Equals(f.Trim(), synthName, StringComparison.OrdinalIgnoreCase)))
                    {
                        if (settings.verboseLogging)
                            Debug.Log($"[SynthBuild] Skipping '{synthName}' (not in filter)");
                        continue;
                    }

                    string destSynthDir = Path.Combine(dest, subdirName, synthName);
                    Directory.CreateDirectory(destSynthDir);

                    foreach (string file in Directory.GetFiles(synthDir))
                    {
                        string ext = Path.GetExtension(file).ToLower();
                        if (ext == ".tmp") continue;

                        string destFile = Path.Combine(destSynthDir, Path.GetFileName(file));
                        File.Copy(file, destFile, overwrite: true);
                        copiedFiles++;

                        if (settings.verboseLogging)
                            Debug.Log($"[SynthBuild] Copied {subdirName}/{synthName}/{Path.GetFileName(file)}");
                    }

                    copiedSynths++;
                }
            }

            if (copiedSynths == 0)
            {
                Debug.LogWarning("[SynthBuild] No matching synth models found — " +
                    "building without models.");
                Cleanup();
                return;
            }

            // Force synchronous import so the build pipeline sees the new files
            // before it collects StreamingAssets for the APK.
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);

            // Verify the files are on disk
            int verifyCount = Directory.GetFiles(dest, "*", SearchOption.AllDirectories).Length;
            Debug.Log($"[SynthBuild] Packaged {copiedFiles} files from " +
                $"{copiedSynths} synth(s) into StreamingAssets " +
                $"(verified {verifyCount} files on disk at {dest}).");
        }

        public void OnPostprocessBuild(BuildReport report)
        {
            var settings = SynthBuildSettings.Load();
            if (settings == null || settings.cleanUpAfterBuild)
                Cleanup();
        }

        /// <summary>
        /// If sourceSubdirectories is populated, use those.
        /// Otherwise auto-discover: scan persistentDataPath for any subdirectory
        /// that contains at least one child folder with model files.
        /// </summary>
        private static string[] DiscoverSourceDirectories(SynthBuildSettings settings)
        {
            string root = Application.persistentDataPath;
            var explicit_ = settings.sourceSubdirectories;
            bool hasExplicit = explicit_ != null && explicit_.Length > 0 &&
                               explicit_.Any(s => !string.IsNullOrWhiteSpace(s));

            if (hasExplicit)
            {
                return explicit_
                    .Where(s => !string.IsNullOrWhiteSpace(s))
                    .Select(s => Path.Combine(root, s.Trim()))
                    .Where(Directory.Exists)
                    .ToArray();
            }

            if (!Directory.Exists(root))
                return Array.Empty<string>();

            return Directory.GetDirectories(root)
                .Where(d => Directory.GetDirectories(d)
                    .Any(sub => Directory.GetFiles(sub).Length > 0))
                .ToArray();
        }

        private static SynthBuildSettings EnsureSettingsAsset()
        {
            string dir = "Assets/Resources";
            if (!Directory.Exists(dir))
                Directory.CreateDirectory(dir);

            string path = $"{dir}/{SynthBuildSettings.SETTINGS_RESOURCE_PATH}.asset";
            if (File.Exists(path))
                return AssetDatabase.LoadAssetAtPath<SynthBuildSettings>(path);

            var asset = ScriptableObject.CreateInstance<SynthBuildSettings>();
            AssetDatabase.CreateAsset(asset, path);
            AssetDatabase.SaveAssets();
            Debug.Log($"[SynthBuild] Created default SynthBuildSettings at {path}");
            return asset;
        }

        private static void Cleanup()
        {
            string dest = StreamingDest;
            if (Directory.Exists(dest))
            {
                Directory.Delete(dest, true);
                string meta = dest + ".meta";
                if (File.Exists(meta))
                    File.Delete(meta);

                AssetDatabase.Refresh();
                Debug.Log("[SynthBuild] Cleaned up StreamingAssets/SynthModels.");
            }
        }
    }
}
