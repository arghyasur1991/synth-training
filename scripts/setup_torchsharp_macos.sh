#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNTH_TRAINING_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default: TorchSharp source as sibling directory to synth-training
TORCHSHARP_SRC="${TORCHSHARP_SRC:-$(cd "$SYNTH_TRAINING_ROOT/.." && pwd)/TorchSharp}"
TORCHSHARP_PROJ="$TORCHSHARP_SRC/src/TorchSharp"

DEPLOY_DIR="${1:-}"
if [ -z "$DEPLOY_DIR" ]; then
    echo "Usage: $0 <unity-project-path>"
    echo ""
    echo "  Builds TorchSharp from source and deploys to a Unity project."
    echo "  Expects TorchSharp source at: $TORCHSHARP_SRC"
    echo "  Set TORCHSHARP_SRC to override."
    echo ""
    echo "Example:"
    echo "  $0 /path/to/MyUnityProject"
    exit 1
fi

PLUGINS_ARM64="$DEPLOY_DIR/Assets/Plugins/arm64"
PACKAGES_TORCH="$DEPLOY_DIR/Assets/Packages/TorchSharp"

echo "=== Building TorchSharp for macOS ==="
echo "  Source:  $TORCHSHARP_SRC"
echo "  Deploy:  $DEPLOY_DIR"

if [ ! -f "$TORCHSHARP_PROJ/TorchSharp.csproj" ]; then
    echo "ERROR: TorchSharp project not found at $TORCHSHARP_PROJ"
    echo "  Clone with: git clone https://github.com/arghyasur1991/TorchSharp.git -b unity-il2cpp-support"
    exit 1
fi

echo ""
echo "--- Step 1: Build TorchSharp ---"
dotnet build -c Release "$TORCHSHARP_PROJ"

echo ""
echo "--- Step 2: Deploy native libraries ---"
NATIVE_DIR="$TORCHSHARP_SRC/bin/arm64.Release/Native"
mkdir -p "$PLUGINS_ARM64"
if [ -d "$NATIVE_DIR" ]; then
    cp -v "$NATIVE_DIR"/*.dylib "$PLUGINS_ARM64/" 2>/dev/null || echo "  No .dylib files in $NATIVE_DIR"
else
    echo "  WARNING: Native build dir not found: $NATIVE_DIR"
fi

echo ""
echo "--- Step 3: Deploy managed DLLs ---"
MANAGED_DIR="$TORCHSHARP_SRC/bin/AnyCPU.Release/TorchSharp/netstandard2.0"
mkdir -p "$PACKAGES_TORCH"
if [ -d "$MANAGED_DIR" ]; then
    cp -v "$MANAGED_DIR"/*.dll "$PACKAGES_TORCH/" 2>/dev/null || echo "  No .dll files in $MANAGED_DIR"
else
    echo "  WARNING: Managed build dir not found: $MANAGED_DIR"
fi

echo ""
echo "--- Step 4: Deploy NuGet dependencies ---"
NUGET_CACHE="$HOME/.nuget/packages"
deploy_nuget() {
    local pkg="$1" rel="$2" dest="$3"
    local src=$(find "$NUGET_CACHE/$pkg" -path "*/$rel" 2>/dev/null | sort | tail -1)
    if [ -n "$src" ] && [ -f "$src" ]; then
        cp -v "$src" "$dest/"
    else
        echo "  WARNING: $pkg/$rel not in NuGet cache"
    fi
}

deploy_nuget "skiasharp" "lib/netstandard2.0/SkiaSharp.dll" "$PACKAGES_TORCH"
deploy_nuget "google.protobuf" "lib/netstandard2.0/Google.Protobuf.dll" "$PACKAGES_TORCH"
deploy_nuget "sharpziplib" "lib/netstandard2.0/ICSharpCode.SharpZipLib.dll" "$PACKAGES_TORCH"
deploy_nuget "system.memory" "lib/netstandard2.0/System.Memory.dll" "$PACKAGES_TORCH"
deploy_nuget "skiasharp.nativeassets.macos" "runtimes/osx/native/libSkiaSharp.dylib" "$PLUGINS_ARM64"

echo ""
echo "=== TorchSharp macOS deployment complete ==="
