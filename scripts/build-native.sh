#!/usr/bin/env bash
# Build the native libraries of the app in the Snapdragon container, and
# collect them into android/snapdragon/jniLibs/arm64-v8a.
#
#   scripts/build-native.sh
#
# Environment:
#   KEEP_BUILD=1     Keep build/native from the previous run (an incremental build).
#                    Without it, the script removes build/native first.
#   NATIVE_TARGETS   The llama.cpp targets, space separated. The default is the
#                    set of libraries that the app ships plus the v79 DSP
#                    library. "all" builds every target, with the tools and
#                    the tests of llama.cpp.
#
# The steps:
#   1. scripts/apply-patches.sh puts the series on the submodule.
#   2. The tracked presets file android/snapdragon/CMakeUserPresets.json goes
#      to the root of the submodule. llama.cpp does not track that file.
#   3. In the container, CMake configures llama.cpp with the preset
#      arm64-android-snapdragon-release into build/native/llama, then the JNI
#      library from android/snapdragon/CMakeLists.txt into build/native/jni,
#      with the flags of android/snapdragon/build.sh.
#   4. The script copies the libraries into android/snapdragon/jniLibs and
#      writes build/hashes-native.txt with the SHA-256 of each library.
#
# Determinism:
#   - The image is pinned by digest, and the repository is always at /workspace.
#   - SOURCE_DATE_EPOCH is the date of the last commit of this repository.
#   - -ffile-prefix-map and -fdebug-prefix-map replace /workspace with "." in
#     the objects. The preset gives the compiler flags, and the script adds the
#     reproducibility flags to them, because a -D on the command line replaces
#     the flags of the preset.
#   - -Werror=date-time stops the build if a source uses __DATE__ or __TIME__.
#   - The DSP library has its own CMake project (an ExternalProject), and its
#     toolchain file sets no CMAKE_C_FLAGS. Thus the CFLAGS and CXXFLAGS
#     environment variables carry the reproducibility flags into it.
#   - LLAMA_BUILD_NUMBER and LLAMA_BUILD_COMMIT come from scripts/lib.sh,
#     because a shallow clone counts one commit. The ggml commit string comes
#     from git in the submodule: "c6824a9-dirty". The applied series makes the
#     tree dirty, thus that string is constant.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

readonly REPRO_FLAGS="-ffile-prefix-map=/workspace=. -fdebug-prefix-map=/workspace=. -Werror=date-time"
readonly DEFAULT_TARGETS="$LLAMA_LIBS htp-$HTP_DSP"
NATIVE_TARGETS=${NATIVE_TARGETS:-$DEFAULT_TARGETS}

cd "$REPO_ROOT"

# 1. The patches.
scripts/apply-patches.sh

# 2. The presets file. The exclude entry keeps the submodule status clean.
cp android/snapdragon/CMakeUserPresets.json "$LLAMA_SUBMODULE/CMakeUserPresets.json"
exclude=$(git -C "$LLAMA_SUBMODULE" rev-parse --git-path info/exclude)
grep -q -x 'CMakeUserPresets.json' "$exclude" 2> /dev/null || echo 'CMakeUserPresets.json' >> "$exclude"

# 3. The build.
if [[ "${KEEP_BUILD:-0}" != 1 ]]; then
    rm -rf build/native
fi
mkdir -p build/native

SOURCE_DATE_EPOCH=$(source_date_epoch)
export SOURCE_DATE_EPOCH
echo "native: SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH targets=[$NATIVE_TARGETS]"

container_run \
    -e SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
    -e LLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER" \
    -e LLAMA_BUILD_COMMIT_SHORT="${LLAMA_COMMIT:0:7}" \
    -e NATIVE_TARGETS="$NATIVE_TARGETS" \
    -e REPRO_FLAGS="$REPRO_FLAGS" \
    "$SNAPDRAGON_IMAGE" bash -euo pipefail -c '
presets=third_party/llama.cpp/CMakeUserPresets.json
preset_flag() {
    python3 -c "
import json, sys
presets = json.load(open(sys.argv[1]))[\"configurePresets\"]
preset = [p for p in presets if p[\"name\"] == \"arm64-android-snapdragon\"][0]
print(preset[\"cacheVariables\"][sys.argv[2]])
" "$presets" "$1"
}
c_flags="$(preset_flag CMAKE_C_FLAGS) $REPRO_FLAGS"
cxx_flags="$(preset_flag CMAKE_CXX_FLAGS) $REPRO_FLAGS"
export CFLAGS="$REPRO_FLAGS" CXXFLAGS="$REPRO_FLAGS"

cmake -S third_party/llama.cpp --preset arm64-android-snapdragon-release -B build/native/llama \
    -DLLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER" \
    -DLLAMA_BUILD_COMMIT="$LLAMA_BUILD_COMMIT_SHORT" \
    -DCMAKE_C_FLAGS="$c_flags" \
    -DCMAKE_CXX_FLAGS="$cxx_flags"
# shellcheck disable=SC2086
cmake --build build/native/llama -j"$(nproc)" --target $NATIVE_TARGETS

cmake -S android/snapdragon -B build/native/jni -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-34 \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
    -DLLAMA_CPP_DIR=/workspace/third_party/llama.cpp \
    -DLLAMA_BUILD_DIR=/workspace/build/native/llama \
    -DCMAKE_C_FLAGS="$REPRO_FLAGS" \
    -DCMAKE_CXX_FLAGS="$REPRO_FLAGS"
cmake --build build/native/jni -j"$(nproc)"
'

# 4. The libraries and their hashes.
out="android/snapdragon/jniLibs/arm64-v8a"
rm -rf "$out"
mkdir -p "$out"
cp build/native/jni/libqwenmobile.so "$out/"
for lib in $LLAMA_LIBS; do
    cp "build/native/llama/bin/lib$lib.so" "$out/"
done
cp "build/native/llama/ggml/src/ggml-hexagon/libggml-htp-$HTP_DSP.so" "$out/"

sha256_table "$out"/*.so > build/hashes-native.txt
echo "native: build/hashes-native.txt"
cat build/hashes-native.txt

# A library must not contain the container path. The prefix map removes it
# from the objects, and this line shows a leak if one occurs.
for lib in "$out"/*.so; do
    count=$(grep -a -c '/workspace' "$lib" || true)
    [[ "$count" == 0 ]] || echo "native: note: $(basename "$lib") contains /workspace on $count lines"
done
