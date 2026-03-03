#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNTH_TRAINING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_ROOT="${BUILD_DIR:-$SYNTH_TRAINING_ROOT/build~}"

TORCHSHARP_SRC="${TORCHSHARP_SRC:-$(cd "$SYNTH_TRAINING_ROOT/.." && pwd)/TorchSharp}"
PYTORCH_SRC="${PYTORCH_SRC:-$(cd "$SYNTH_TRAINING_ROOT/.." && pwd)/pytorch}"

ANDROID_ABI="${ANDROID_ABI:-arm64-v8a}"
ANDROID_PLATFORM="${ANDROID_PLATFORM:-android-32}"

DEPLOY_DIR="${1:-}"
if [ -z "$DEPLOY_DIR" ]; then
    echo "Usage: $0 <unity-project-path>"
    echo ""
    echo "  Cross-compiles LibTorch + LibTorchSharp for Android arm64-v8a"
    echo "  and deploys to a Unity project."
    echo ""
    echo "  Expects:"
    echo "    PyTorch source:    $PYTORCH_SRC"
    echo "    TorchSharp source: $TORCHSHARP_SRC"
    echo ""
    echo "  Set PYTORCH_SRC and TORCHSHARP_SRC to override."
    exit 1
fi

PLUGINS_ANDROID="$DEPLOY_DIR/Assets/Plugins/Android/$ANDROID_ABI"

find_ndk() {
    if [ -n "${ANDROID_NDK_HOME:-}" ] && [ -d "$ANDROID_NDK_HOME" ]; then
        echo "$ANDROID_NDK_HOME"
        return
    fi
    for editor_dir in /Applications/Unity/Hub/Editor/*/; do
        ndk="$editor_dir/PlaybackEngines/AndroidPlayer/NDK"
        if [ -d "$ndk" ]; then
            echo "$ndk"
            return
        fi
    done
    echo ""
}

NDK_HOME="${ANDROID_NDK_HOME:-$(find_ndk)}"
if [ -z "$NDK_HOME" ] || [ ! -d "$NDK_HOME" ]; then
    echo "ERROR: Android NDK not found."
    echo "  Set ANDROID_NDK_HOME or install Android Build Support via Unity Hub."
    exit 1
fi

NDK_TOOLCHAIN="$NDK_HOME/build/cmake/android.toolchain.cmake"

echo "=== Building TorchSharp for Android $ANDROID_ABI ==="
echo "  NDK:        $NDK_HOME"
echo "  PyTorch:    $PYTORCH_SRC"
echo "  TorchSharp: $TORCHSHARP_SRC"
echo "  Deploy:     $DEPLOY_DIR"

if [ ! -f "$PYTORCH_SRC/CMakeLists.txt" ]; then
    echo "ERROR: PyTorch source not found at $PYTORCH_SRC"
    echo "  Clone with: git clone --depth=1 https://github.com/pytorch/pytorch.git"
    echo "  Then: cd pytorch && git submodule update --init --recursive --depth=1"
    exit 1
fi

echo ""
echo "--- Step 1: Build LibTorch (static, CPU, with autograd) ---"
PYTORCH_BUILD="$BUILD_ROOT/pytorch_android_arm64"
mkdir -p "$PYTORCH_BUILD"

export ANDROID_NDK="$NDK_HOME"
export ANDROID_ABI="$ANDROID_ABI"
export BUILD_LITE_INTERPRETER=0

LIBTORCH_INSTALL="$PYTORCH_BUILD/install"

if [ ! -f "$LIBTORCH_INSTALL/share/cmake/Torch/TorchConfig.cmake" ]; then
    # PyTorch forces NO_API=ON for Android (INTERN_BUILD_MOBILE), which disables
    # the C++ frontend (torch::nn, torch::optim). We need the C++ API for
    # TorchSharp, so temporarily patch CMakeLists.txt.
    PYTORCH_CMAKE="$PYTORCH_SRC/CMakeLists.txt"
    PATCHED=0
    if grep -q "set(NO_API ON)" "$PYTORCH_CMAKE"; then
        sed -i.bak \
            -e 's/set(NO_API ON)/# set(NO_API ON)  # Patched for C++ API support/' \
            -e 's/set(INTERN_DISABLE_AUTOGRAD ON)/set(INTERN_DISABLE_AUTOGRAD OFF)  # Patched for autograd/' \
            "$PYTORCH_CMAKE"
        PATCHED=1
        echo "  Applied temporary CMakeLists.txt patches (NO_API, autograd)"
    fi

    bash "$PYTORCH_SRC/scripts/build_android.sh" \
        -DUSE_NNPACK=OFF -DUSE_MKLDNN=OFF -DUSE_QNNPACK=OFF \
        -DUSE_XNNPACK=OFF -DUSE_PYTORCH_QNNPACK=OFF \
        -DUSE_DISTRIBUTED=OFF -DUSE_OBSERVERS=OFF -DUSE_KINETO=OFF \
        -DBUILD_CAFFE2_OPS=OFF -DUSE_FBGEMM=OFF -DUSE_METAL=OFF \
        -DUSE_VULKAN=OFF -DANDROID_NATIVE_API_LEVEL=32
    BUILD_RC=$?

    if [ "$PATCHED" = "1" ]; then
        cd "$PYTORCH_SRC" && git checkout CMakeLists.txt 2>/dev/null
        rm -f "${PYTORCH_CMAKE}.bak"
        echo "  Reverted CMakeLists.txt patches"
    fi

    [ $BUILD_RC -ne 0 ] && { echo "ERROR: LibTorch build failed."; exit 1; }

    if [ ! -f "$LIBTORCH_INSTALL/share/cmake/Torch/TorchConfig.cmake" ]; then
        ALT="$PYTORCH_SRC/build_android/install"
        if [ -f "$ALT/share/cmake/Torch/TorchConfig.cmake" ]; then
            echo "  Linking to default build output: $ALT"
            ln -sf "$ALT" "$LIBTORCH_INSTALL"
        else
            echo "ERROR: LibTorch build failed."
            exit 1
        fi
    fi
else
    echo "  LibTorch already built, skipping. Remove $LIBTORCH_INSTALL to rebuild."
fi

echo ""
echo "--- Step 2: Build LibTorchSharp.so ---"
TORCHSHARP_BUILD="$BUILD_ROOT/torchsharp_android_arm64"
TORCHSHARP_NATIVE_SRC="$TORCHSHARP_SRC/src/Native"

# Look for the CMake wrapper in synth-training/tools or create inline
TORCHSHARP_CMAKE="$SCRIPT_DIR/../tools~/torchsharp_android"
if [ ! -f "$TORCHSHARP_CMAKE/CMakeLists.txt" ]; then
    echo "  WARNING: No CMake wrapper found at $TORCHSHARP_CMAKE"
    echo "  Skipping LibTorchSharp build. Create the CMake wrapper first."
    exit 1
fi

mkdir -p "$TORCHSHARP_BUILD"
cmake \
    -DCMAKE_TOOLCHAIN_FILE="$NDK_TOOLCHAIN" \
    -DANDROID_ABI="$ANDROID_ABI" \
    -DANDROID_PLATFORM="$ANDROID_PLATFORM" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLIBTORCH_PATH="$LIBTORCH_INSTALL" \
    -DTORCHSHARP_NATIVE_SRC="$TORCHSHARP_NATIVE_SRC" \
    -DANDROID_CPP_FEATURES="rtti exceptions" \
    -S "$TORCHSHARP_CMAKE" \
    -B "$TORCHSHARP_BUILD"

CPU_COUNT=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
cmake --build "$TORCHSHARP_BUILD" -j"$CPU_COUNT"

SO_PATH=$(find "$TORCHSHARP_BUILD" -name "libLibTorchSharp.so" | head -1)
if [ -z "$SO_PATH" ]; then
    echo "ERROR: libLibTorchSharp.so not found after build."
    exit 1
fi

STRIP=$(find "$NDK_HOME" -name "llvm-strip" -path "*/bin/*" | head -1)
if [ -n "$STRIP" ]; then
    "$STRIP" "$SO_PATH"
fi

echo ""
echo "--- Step 3: Deploy ---"
mkdir -p "$PLUGINS_ANDROID"
cp -v "$SO_PATH" "$PLUGINS_ANDROID/libLibTorchSharp.so"

echo ""
echo "=== TorchSharp Android deployment complete ==="
