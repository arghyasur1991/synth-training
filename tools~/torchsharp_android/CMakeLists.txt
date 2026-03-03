# CMake wrapper for cross-compiling LibTorchSharp as a single shared library
# for Android ARM64, statically linking all of LibTorch inside it.
#
# Invoked by build.py --torchsharp-android after LibTorch static build completes.
#
# Usage:
#   cmake \
#     -DCMAKE_TOOLCHAIN_FILE=<ndk>/build/cmake/android.toolchain.cmake \
#     -DANDROID_ABI=arm64-v8a \
#     -DANDROID_PLATFORM=android-32 \
#     -DLIBTORCH_PATH=<pytorch>/build_android/install \
#     -DTORCHSHARP_NATIVE_SRC=<TorchSharp>/src/Native \
#     -S tools/build_torchsharp_android.cmake \
#     -B build/torchsharp_android_arm64

cmake_minimum_required(VERSION 3.18)
project(LibTorchSharpAndroid LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Required cache variables (set by build.py) ---
set(LIBTORCH_PATH "" CACHE PATH "LibTorch install prefix from build_android.sh")
set(TORCHSHARP_NATIVE_SRC "" CACHE PATH "TorchSharp/src/Native directory")

if(NOT EXISTS "${LIBTORCH_PATH}/share/cmake/Torch/TorchConfig.cmake")
    message(FATAL_ERROR
        "LIBTORCH_PATH does not contain a valid Torch install: ${LIBTORCH_PATH}\n"
        "Run the LibTorch Android build first (build.py handles this).")
endif()

if(NOT EXISTS "${TORCHSHARP_NATIVE_SRC}/LibTorchSharp/CMakeLists.txt")
    message(FATAL_ERROR
        "TORCHSHARP_NATIVE_SRC is invalid: ${TORCHSHARP_NATIVE_SRC}\n"
        "Should point to TorchSharp/src/Native")
endif()

# --- Find static LibTorch ---
# Android NDK toolchain restricts find_library() to sysroot only.
# Add LibTorch install to the root search path so TorchConfig.cmake
# can locate libtorch.a, libc10.a, etc.
list(APPEND CMAKE_FIND_ROOT_PATH "${LIBTORCH_PATH}")
set(CMAKE_PREFIX_PATH "${LIBTORCH_PATH}")
set(Torch_DIR "${LIBTORCH_PATH}/share/cmake/Torch")
set(Caffe2_DIR "${LIBTORCH_PATH}/share/cmake/Caffe2")
find_package(Torch REQUIRED)
message(STATUS "Found Torch: ${TORCH_LIBRARIES}")
message(STATUS "Torch include: ${TORCH_INCLUDE_DIRS}")

# --- Collect LibTorchSharp sources ---
set(LTS_SRC_DIR "${TORCHSHARP_NATIVE_SRC}/LibTorchSharp")
file(GLOB LTS_CPP_SOURCES "${LTS_SRC_DIR}/*.cpp")
file(GLOB LTS_C_SOURCES   "${LTS_SRC_DIR}/*.c")
set(LTS_SOURCES ${LTS_CPP_SOURCES} ${LTS_C_SOURCES})

# Exclude JIT module — JIT serialization (export.cpp) is not compiled in mobile builds.
# TorchSharp's THSJIT.cpp references Module::save() and ExportModule() which are unavailable.
# We don't need JIT on Android (RL training uses the C++ API directly).
list(FILTER LTS_SOURCES EXCLUDE REGEX "THSJIT\\.cpp$")

# Add stub implementations for JIT serialization symbols missing in mobile builds
list(APPEND LTS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/android_stubs.cpp")

list(LENGTH LTS_SOURCES _src_count)
message(STATUS "LibTorchSharp sources: ${_src_count} files from ${LTS_SRC_DIR} (THSJIT excluded, stubs added)")

# --- Build shared library ---
add_library(LibTorchSharp SHARED ${LTS_SOURCES})

target_include_directories(LibTorchSharp PRIVATE
    ${TORCH_INCLUDE_DIRS}
    "${LTS_SRC_DIR}"
)

target_compile_definitions(LibTorchSharp PRIVATE
    -D_GLIBCXX_USE_CXX11_ABI=1
)

# Suppress PyTorch header warnings that TorchSharp triggers
target_compile_options(LibTorchSharp PRIVATE
    -Wno-unused-local-typedef
    -fPIC
    -fvisibility=hidden
)

# --- Link static LibTorch ---
# TorchConfig.cmake already wraps libtorch.a and libtorch_cpu.a in
# --whole-archive to preserve operator registrations. Use TORCH_LIBRARIES
# directly — it contains the correct linker flags.
target_link_libraries(LibTorchSharp ${TORCH_LIBRARIES} log)

# LTO for dead code elimination — reduces binary size significantly
include(CheckIPOSupported)
check_ipo_supported(RESULT _lto_supported OUTPUT _lto_error)
if(_lto_supported)
    set_target_properties(LibTorchSharp PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION TRUE)
    message(STATUS "LTO enabled for LibTorchSharp")
else()
    message(STATUS "LTO not supported: ${_lto_error}")
endif()

# RPATH for Android (not strictly needed but good practice)
set_target_properties(LibTorchSharp PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# --- Install ---
install(TARGETS LibTorchSharp
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
