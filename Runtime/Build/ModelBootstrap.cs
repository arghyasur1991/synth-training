using System;
using System.IO;
using UnityEngine;
using UnityEngine.Networking;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Extracts bundled model checkpoints from StreamingAssets to
    /// persistentDataPath on first launch. On Android, StreamingAssets
    /// are inside the APK and must be read via UnityWebRequest.
    /// </summary>
    public static class ModelBootstrap
    {
        private const string MARKER_FILE = ".synth_extracted";

        /// <summary>
        /// If persistentDataPath/{subdir}/{synthName} is empty but
        /// StreamingAssets/SynthModels/{synthName} has bundled models,
        /// extract them. Returns true if extraction happened.
        /// </summary>
        public static bool ExtractIfNeeded(string saveSubdirectory, string synthName)
        {
            string destDir = Path.Combine(
                Application.persistentDataPath, saveSubdirectory, synthName);

            if (Directory.Exists(destDir) && File.Exists(Path.Combine(destDir, MARKER_FILE)))
                return false;

            string streamingDir = Path.Combine(
                SynthBuildSettings.STREAMING_ASSETS_SUBDIR, saveSubdirectory, synthName);

            string manifestPath = Path.Combine(
                Application.streamingAssetsPath, streamingDir, "meta.json");

            if (!StreamingFileExists(manifestPath))
                return false;

            Directory.CreateDirectory(destDir);

            string[] knownFiles = {
                "meta.json", "normalizer.bin", "physics_state.bin",
                "ppo_actor.pt", "ppo_critic.pt", "ppo_state.bin",
                "sac_agent.pt", "sac_state.bin",
                "imitation_state.bin",
                "reward_bar.bin", "curriculum.bin"
            };

            int extracted = 0;
            foreach (string fileName in knownFiles)
            {
                string srcPath = Path.Combine(
                    Application.streamingAssetsPath, streamingDir, fileName);
                string dstPath = Path.Combine(destDir, fileName);

                if (File.Exists(dstPath)) continue;

                byte[] data = ReadStreamingAsset(srcPath);
                if (data != null && data.Length > 0)
                {
                    File.WriteAllBytes(dstPath, data);
                    extracted++;
                }
            }

            if (extracted > 0)
            {
                File.WriteAllText(Path.Combine(destDir, MARKER_FILE),
                    DateTime.UtcNow.ToString("o"));
                Debug.Log($"[ModelBootstrap] Extracted {extracted} files for '{synthName}'");
            }

            return extracted > 0;
        }

        private static bool StreamingFileExists(string path)
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            return ReadStreamingAsset(path) != null;
#else
            return File.Exists(path);
#endif
        }

        private static byte[] ReadStreamingAsset(string path)
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            try
            {
                using var request = UnityWebRequest.Get(path);
                var op = request.SendWebRequest();
                while (!op.isDone) { }
                if (request.result != UnityWebRequest.Result.Success)
                    return null;
                return request.downloadHandler.data;
            }
            catch
            {
                return null;
            }
#else
            try
            {
                if (!File.Exists(path)) return null;
                return File.ReadAllBytes(path);
            }
            catch
            {
                return null;
            }
#endif
        }
    }
}
