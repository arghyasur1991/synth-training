using System;
using System.IO;
using System.Runtime.InteropServices;
using UnityEngine;

namespace Genesis.Sentience.Learning
{
    /// <summary>
    /// Pre-loads TorchSharp native libraries before any managed TorchSharp code runs.
    ///
    /// Why this is needed:
    ///   TorchSharp's built-in discovery (LoadNativeBackend) expects a NuGet package
    ///   directory layout.  Unity puts native plugins in Assets/Plugins/{arch}/ which
    ///   doesn't match that layout, so discovery fails on macOS.
    ///
    /// How it works:
    ///   1. [RuntimeInitializeOnLoadMethod(BeforeSceneLoad)] fires before any
    ///      MonoBehaviour, guaranteeing the libs are in memory before TorchSharp's
    ///      static constructor runs.
    ///   2. dlopen with RTLD_NOW | RTLD_GLOBAL loads each library and exposes its
    ///      symbols process-wide so that dependent libraries can resolve them.
    ///   3. TORCHSHARP_NATIVE_PRELOADED=1 tells the patched TorchSharp submodule
    ///      to skip its own discovery entirely (see Torch.cs NativeBackendPreloaded).
    ///
    /// Dependency chain (load order matters):
    ///   libomp  →  libc10  →  libtorch_cpu  →  libtorch  →  libLibTorchSharp
    ///
    /// Adding new native dependencies:
    ///   If a future PyTorch/TorchSharp upgrade introduces new shared libraries,
    ///   add them to the RequiredLibs or OptionalLibs arrays in the correct
    ///   dependency order.  Use OptionalLibs for libs that may not exist on all
    ///   platforms (loader skips missing optional libs silently).
    ///
    /// Platform support:
    ///   macOS arm64   — tested and working
    ///   macOS x86_64  — should work (same dlopen mechanism)
    ///   Windows        — uses LoadLibrary (untested, needs validation)
    ///   Linux          — uses dlopen (untested, needs validation)
    ///   Android ARM64  — single-SO build; Unity/DllImport handles loading automatically
    /// </summary>
    public static class TorchSharpLoader
    {
        /// <summary>Whether native libraries have been successfully loaded.</summary>
        public static bool IsLoaded { get; private set; }

        /// <summary>Human-readable error if loading failed, null otherwise.</summary>
        public static string LoadError { get; private set; }

        // ── Native loading functions ──────────────────────────────────────

#if UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX || UNITY_EDITOR_LINUX || UNITY_STANDALONE_LINUX
        [DllImport("libdl")]
        private static extern IntPtr dlopen(string filename, int flags);

        [DllImport("libdl")]
        private static extern IntPtr dlerror();

        private const int RTLD_NOW    = 0x2;
        private const int RTLD_GLOBAL = 0x8;

        private static IntPtr PlatformLoad(string path)
        {
            dlerror();
            IntPtr handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
            if (handle == IntPtr.Zero)
            {
                IntPtr errPtr = dlerror();
                string err = errPtr != IntPtr.Zero
                    ? Marshal.PtrToStringAnsi(errPtr)
                    : "unknown error";
                throw new DllNotFoundException($"dlopen failed for {Path.GetFileName(path)}: {err}");
            }
            return handle;
        }

        private static string LibName(string baseName) => baseName + ".dylib";

#elif UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern IntPtr LoadLibraryEx(string lpFileName, IntPtr hReservedNull, int dwFlags);

        private const int LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008;

        private static IntPtr PlatformLoad(string path)
        {
            IntPtr handle = LoadLibraryEx(path, IntPtr.Zero, LOAD_WITH_ALTERED_SEARCH_PATH);
            if (handle == IntPtr.Zero)
            {
                int err = Marshal.GetLastWin32Error();
                throw new DllNotFoundException(
                    $"LoadLibraryEx failed for {Path.GetFileName(path)}: Win32 error {err}");
            }
            return handle;
        }

        private static string LibName(string baseName) => baseName + ".dll";
#else
        private static IntPtr PlatformLoad(string path)
        {
            throw new PlatformNotSupportedException("TorchSharpLoader: unsupported platform");
        }

        private static string LibName(string baseName) => baseName + ".so";
#endif

        // ── Library lists ─────────────────────────────────────────────────

        private static readonly string[] RequiredLibs = new[]
        {
            "libc10",
            "libtorch_cpu",
            "libtorch",
            "libLibTorchSharp",
        };

        private static readonly string[] OptionalLibs = new[]
        {
            "libomp",
            "libshm",
            "libtorch_global_deps",
        };

        // ── Plugin directory resolution ───────────────────────────────────

        private static string GetPluginDirectory()
        {
#if UNITY_EDITOR_OSX || UNITY_STANDALONE_OSX
            return Path.Combine(Application.dataPath, "Plugins", "arm64");
#elif UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN
            return Path.Combine(Application.dataPath, "Plugins", "x86_64");
#elif UNITY_EDITOR_LINUX || UNITY_STANDALONE_LINUX
            return Path.Combine(Application.dataPath, "Plugins", "x86_64");
#else
            return null;
#endif
        }

        // ── Entry point ───────────────────────────────────────────────────

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
        private static void Initialize()
        {
            if (IsLoaded) return;

#if UNITY_ANDROID && !UNITY_EDITOR
            // Android single-SO build: LibTorch is statically linked into
            // libLibTorchSharp.so.  Unity extracts it from the APK and
            // Android's linker resolves DllImport("LibTorchSharp") automatically.
            Environment.SetEnvironmentVariable("TORCHSHARP_NATIVE_PRELOADED", "1");
            IsLoaded = true;
            Debug.Log("TorchSharpLoader: Android — single-SO mode, DllImport handles loading.");
            return;
#endif

            string pluginDir = GetPluginDirectory();
            if (pluginDir == null || !Directory.Exists(pluginDir))
            {
                LoadError = $"Plugin directory not found: {pluginDir ?? "null"}";
                Debug.LogWarning($"TorchSharpLoader: {LoadError}");
                return;
            }

            foreach (string lib in OptionalLibs)
            {
                string path = Path.Combine(pluginDir, LibName(lib));
                if (!File.Exists(path)) continue;

                try { PlatformLoad(path); }
                catch (Exception e)
                {
                    Debug.LogWarning($"TorchSharpLoader: Optional lib {lib} failed: {e.Message}");
                }
            }

            foreach (string lib in RequiredLibs)
            {
                string path = Path.Combine(pluginDir, LibName(lib));
                if (!File.Exists(path))
                {
                    LoadError = $"Required library not found: {path}";
                    Debug.LogError($"TorchSharpLoader: {LoadError}");
                    return;
                }

                try
                {
                    PlatformLoad(path);
                }
                catch (Exception e)
                {
                    LoadError = $"Failed to load {lib}: {e.Message}";
                    Debug.LogError($"TorchSharpLoader: {LoadError}");
                    return;
                }
            }

            Environment.SetEnvironmentVariable("TORCHSHARP_NATIVE_PRELOADED", "1");

            IsLoaded = true;
            Debug.Log($"TorchSharpLoader: Native libraries loaded successfully from {pluginDir}");
        }
    }
}
