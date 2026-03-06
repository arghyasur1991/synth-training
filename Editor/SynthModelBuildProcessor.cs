using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace Genesis.Sentience.Learning.Editor
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
            if (settings == null || !settings.includeModelsInBuild)
                return;

            string sourceRoot = Path.Combine(
                Application.persistentDataPath, settings.sourceSubdirectory);

            if (!Directory.Exists(sourceRoot))
            {
                Debug.LogWarning("[SynthBuild] No trained models found at " +
                    $"{sourceRoot} — building without models.");
                return;
            }

            string[] synthDirs = Directory.GetDirectories(sourceRoot);
            if (synthDirs.Length == 0)
            {
                Debug.LogWarning("[SynthBuild] No synth model directories in " +
                    $"{sourceRoot} — building without models.");
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

                string destSynthDir = Path.Combine(dest, synthName);
                Directory.CreateDirectory(destSynthDir);

                foreach (string file in Directory.GetFiles(synthDir))
                {
                    string ext = Path.GetExtension(file).ToLower();
                    if (ext == ".tmp") continue;

                    string destFile = Path.Combine(destSynthDir, Path.GetFileName(file));
                    File.Copy(file, destFile, overwrite: true);
                    copiedFiles++;

                    if (settings.verboseLogging)
                        Debug.Log($"[SynthBuild] Copied {synthName}/{Path.GetFileName(file)}");
                }

                copiedSynths++;
            }

            if (copiedSynths == 0)
            {
                Debug.LogWarning("[SynthBuild] No matching synth models found — " +
                    "building without models.");
                Cleanup();
                return;
            }

            AssetDatabase.Refresh();
            Debug.Log($"[SynthBuild] Packaged {copiedFiles} files from " +
                $"{copiedSynths} synth(s) into StreamingAssets.");
        }

        public void OnPostprocessBuild(BuildReport report)
        {
            Cleanup();
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
