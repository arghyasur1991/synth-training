using System;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;

#if UNITY_ANDROID && !UNITY_EDITOR
using System.IO.Compression;
#endif

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Extracts bundled model checkpoints from StreamingAssets to
    /// persistentDataPath on first launch.
    ///
    /// Runs on a background thread, triggered automatically before
    /// scene load via RuntimeInitializeOnLoadMethod. On Android, reads
    /// directly from the APK (zip) — no UnityWebRequest / main thread needed.
    ///
    /// Skills check <see cref="IsComplete"/> during init; SynthBrain's
    /// retry loop naturally waits until extraction finishes.
    /// </summary>
    public static class ModelBootstrap
    {
        private const string MARKER_FILE = ".synth_extracted";

        private static volatile bool _complete;
        private static volatile bool _started;
        private static string _dataPath;
        private static string _persistentDataPath;
        private static string _streamingAssetsPath;

        /// <summary>True once background extraction has finished (or was skipped).</summary>
        public static bool IsComplete => _complete;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        private static void AutoStart()
        {
            _complete = false;
            _started = false;
            Start();
        }

        /// <summary>
        /// Kick off background extraction if not already running.
        /// Safe to call multiple times (idempotent).
        /// </summary>
        public static void Start()
        {
            if (_started) return;
            _started = true;

            // Cache Unity API values on the main thread before going to bg
            _dataPath = Application.dataPath;
            _persistentDataPath = Application.persistentDataPath;
            _streamingAssetsPath = Application.streamingAssetsPath;

            Task.Run(ExtractAll);
        }

        private static void ExtractAll()
        {
            try
            {
#if UNITY_ANDROID && !UNITY_EDITOR
                ExtractFromApk();
#else
                ExtractFromFileSystem();
#endif
            }
            catch (Exception e)
            {
                Debug.LogError($"[ModelBootstrap] Extraction failed: {e.Message}\n{e.StackTrace}");
            }
            finally
            {
                _complete = true;
            }
        }

#if UNITY_ANDROID && !UNITY_EDITOR
        private static void ExtractFromApk()
        {
            string prefix = "assets/" + SynthBuildSettings.STREAMING_ASSETS_SUBDIR + "/";
            int extracted = 0;

            using (var zip = ZipFile.OpenRead(_dataPath))
            {
                foreach (var entry in zip.Entries)
                {
                    if (!entry.FullName.StartsWith(prefix, StringComparison.Ordinal))
                        continue;
                    if (entry.FullName.EndsWith("/", StringComparison.Ordinal))
                        continue; // skip directory entries
                    if (entry.Length == 0)
                        continue;

                    // "assets/SynthModels/ImitationLearning/GirlSynth/meta.json"
                    //  → "ImitationLearning/GirlSynth/meta.json"
                    string relativePath = entry.FullName.Substring(prefix.Length);
                    string destPath = Path.Combine(_persistentDataPath, relativePath);

                    if (File.Exists(destPath))
                        continue;

                    Directory.CreateDirectory(Path.GetDirectoryName(destPath));
                    using (var src = entry.Open())
                    using (var dst = File.Create(destPath))
                        src.CopyTo(dst);
                    extracted++;
                }
            }

            // Write marker files per synth directory
            if (extracted > 0)
                WriteMarkers(_persistentDataPath, prefix.Length);

            Debug.Log($"[ModelBootstrap] APK extraction complete — {extracted} files");
        }
#endif

        private static void ExtractFromFileSystem()
        {
            string srcRoot = Path.Combine(
                _streamingAssetsPath, SynthBuildSettings.STREAMING_ASSETS_SUBDIR);

            if (!Directory.Exists(srcRoot))
            {
                Debug.Log("[ModelBootstrap] No bundled models in StreamingAssets.");
                return;
            }

            int extracted = 0;

            foreach (string file in Directory.GetFiles(srcRoot, "*", SearchOption.AllDirectories))
            {
                string relativePath = file.Substring(srcRoot.Length + 1);
                string destPath = Path.Combine(_persistentDataPath, relativePath);

                if (File.Exists(destPath))
                    continue;

                Directory.CreateDirectory(Path.GetDirectoryName(destPath));
                File.Copy(file, destPath);
                extracted++;
            }

            if (extracted > 0)
                WriteMarkers(_persistentDataPath, 0);

            Debug.Log($"[ModelBootstrap] FileSystem extraction complete — {extracted} files");
        }

        /// <summary>
        /// Write a marker file in each synth leaf directory to skip future extractions.
        /// Scans persistentDataPath for directories that contain meta.json.
        /// </summary>
        private static void WriteMarkers(string root, int unused)
        {
            try
            {
                foreach (string metaFile in Directory.GetFiles(root, "meta.json", SearchOption.AllDirectories))
                {
                    string dir = Path.GetDirectoryName(metaFile);
                    string marker = Path.Combine(dir, MARKER_FILE);
                    if (!File.Exists(marker))
                        File.WriteAllText(marker, DateTime.UtcNow.ToString("o"));
                }
            }
            catch { /* best effort */ }
        }
    }
}
