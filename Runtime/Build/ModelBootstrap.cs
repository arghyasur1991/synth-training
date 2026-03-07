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
        private static string _apkPath;
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

            _persistentDataPath = Application.persistentDataPath;
            _streamingAssetsPath = Application.streamingAssetsPath;

            // On Android, derive APK path from streamingAssetsPath which is
            // "jar:file:///data/app/.../base.apk!/assets" — more reliable
            // than Application.dataPath which may not include the full path.
            _apkPath = Application.dataPath;
#if UNITY_ANDROID && !UNITY_EDITOR
            _apkPath = ResolveApkPath(_streamingAssetsPath, _apkPath);
#endif

            Debug.Log($"[ModelBootstrap] Starting background extraction " +
                      $"(apk={_apkPath}, persist={_persistentDataPath})");

            Task.Run(ExtractAll);
        }

        private static string ResolveApkPath(string streamingAssetsPath, string fallback)
        {
            // streamingAssetsPath = "jar:file:///data/app/.../base.apk!/assets"
            const string jarPrefix = "jar:file://";
            if (streamingAssetsPath.StartsWith(jarPrefix))
            {
                string path = streamingAssetsPath.Substring(jarPrefix.Length);
                int bang = path.IndexOf('!');
                if (bang >= 0)
                    return path.Substring(0, bang);
            }
            return fallback;
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
            int totalEntries = 0;
            string firstEntry = null;

            using (var zip = ZipFile.OpenRead(_apkPath))
            {
                foreach (var entry in zip.Entries)
                {
                    totalEntries++;
                    if (firstEntry == null && entry.FullName.StartsWith("assets/"))
                        firstEntry = entry.FullName;

                    if (!entry.FullName.StartsWith(prefix, StringComparison.Ordinal))
                        continue;
                    if (entry.FullName.EndsWith("/", StringComparison.Ordinal))
                        continue;
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

            if (extracted > 0)
                WriteMarkers(_persistentDataPath);

            Debug.Log($"[ModelBootstrap] APK extraction: {extracted} files extracted " +
                      $"(zip has {totalEntries} entries, prefix='{prefix}', " +
                      $"firstAsset='{firstEntry ?? "none"}')");
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
                WriteMarkers(_persistentDataPath);

            Debug.Log($"[ModelBootstrap] FileSystem extraction complete — {extracted} files");
        }

        private static void WriteMarkers(string root)
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
