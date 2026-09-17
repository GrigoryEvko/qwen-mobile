#!/bin/bash
# Build libqwenmobile.so in the Snapdragon container against the shared llama.cpp
# build, and collect every library the app ships in jniLibs/arm64-v8a.
#
#   LLAMA_CPP_DIR=~/Downloads/llama.cpp android/snapdragon/build.sh
#
# The llama.cpp build must exist: cmake --preset arm64-android-snapdragon-release
# -B build-snapdragon in the same container (refer to the memory notes).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
LLAMA=${LLAMA_CPP_DIR:-$HOME/Downloads/llama.cpp}
IMAGE=ghcr.io/snapdragon-toolchain/arm64-android:v0.7
DSP=v79

podman run --rm --userns=keep-id --security-opt label=disable \
    -v "$LLAMA:/llama" -v "$HERE/..:/android" -w /android/snapdragon "$IMAGE" bash -c '
set -e
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-34 \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
    -DLLAMA_CPP_DIR=/llama \
    -DLLAMA_BUILD_DIR=/llama/build-snapdragon
cmake --build build -j
'

OUT="$HERE/jniLibs/arm64-v8a"
rm -rf "$OUT"
mkdir -p "$OUT"
cp "$HERE/build/libqwenmobile.so" "$OUT/"
for lib in llama ggml ggml-base ggml-cpu ggml-opencl ggml-hexagon mtmd llama-common; do
    cp "$LLAMA/build-snapdragon/bin/lib$lib.so" "$OUT/"
done
cp "$LLAMA/build-snapdragon/ggml/src/ggml-hexagon/libggml-htp-$DSP.so" "$OUT/"
ls -la "$OUT"
