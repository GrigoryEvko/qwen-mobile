#!/usr/bin/env bash
# Build the phone stage "bench-kv": the libraries of the app from a private tree of HEAD plus the
# switch GGML_HEXAGON_FWHT, and llama-bench and memprobe against them.
#
#   JOBS=24 tools/stages/bench-kv/build.sh
#
# build/bench-kv/build.sh is a link to this file. The files of the stage go to build/bench-kv.
#
# The steps:
#   1. tests/sanitizers/llama-copy.sh makes build/bench-kv/src, the llama.cpp tree of HEAD (the
#      pin plus patches/series). git apply puts build/bench-kv/fwht-switch.patch on it.
#   2. In the Snapdragon container, under the lock build/.container.lock, CMake configures the
#      tree into build/bench-kv/android with the preset, the compiler flags (with -flto) and the
#      build number and commit of scripts/build-native.sh. It builds the libraries of the app
#      (LLAMA_LIBS of scripts/lib.sh), the DSP library v79 and llama-bench. The NDK clang++
#      compiles tools/memprobe/memprobe.cpp with the three app sources that it calls, as
#      tools/memprobe/build-phone.sh does.
#   3. The script copies the files of the stage into build/bench-kv/phone, writes SHA256SUMS, and
#      compares each library with build/hashes-native.txt (the libraries of the APK of 56ec1eb).
#
# Two more prefix maps record the sources of the private tree as third_party/llama.cpp and the
# build directory as build/native/llama, the paths of the app build. Thus a library that neither
# the switch nor a later patch changes can have the bytes of the APK library. The ggml commit
# string comes from git in the source directory: the private tree is not a git checkout, thus
# libggml-base.so gets a different string than "c6824a9-dirty" of the app build.
#
# Time: about 15 minutes with JOBS=24 (the LTO links take most of it). Disk: about 2 GB.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../scripts/lib.sh"
JOBS=${JOBS:-24}
cd "$REPO_ROOT"

readonly stage=build/bench-kv
readonly tree=$stage/src
readonly bdir=$stage/android
readonly out=$stage/phone
readonly repro="-ffile-prefix-map=/workspace=. -fdebug-prefix-map=/workspace=. -Werror=date-time"
readonly remap="-ffile-prefix-map=/workspace/$tree=./third_party/llama.cpp -ffile-prefix-map=/workspace/$bdir=./build/native/llama"

# 1. The tree.
tests/sanitizers/llama-copy.sh "$tree"
git apply --directory="$tree" "$stage/fwht-switch.patch"
cp -f android/snapdragon/CMakeUserPresets.json "$tree/CMakeUserPresets.json"

# 2. The build.
mkdir -p "$bdir"
SOURCE_DATE_EPOCH=$(source_date_epoch)
echo "bench-kv: SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH JOBS=$JOBS"
(
    flock 9
    container_run \
        -e SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
        -e LLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER" \
        -e LLAMA_BUILD_COMMIT_SHORT="${LLAMA_COMMIT:0:7}" \
        -e TARGETS="$LLAMA_LIBS htp-v79 llama-bench" \
        -e JOBS="$JOBS" \
        -e FLAGS_EXTRA="$repro $remap" \
        -e TREE="$tree" -e BDIR="$bdir" -e APP_SRC=android/app/src/main/cpp \
        "$SNAPDRAGON_IMAGE" bash -euo pipefail -c '
preset_flag() {
    python3 -c "
import json, sys
presets = json.load(open(sys.argv[1]))[\"configurePresets\"]
preset = [p for p in presets if p[\"name\"] == \"arm64-android-snapdragon\"][0]
print(preset[\"cacheVariables\"][sys.argv[2]])
" "$TREE/CMakeUserPresets.json" "$1"
}
c_flags="$(preset_flag CMAKE_C_FLAGS) $FLAGS_EXTRA"
cxx_flags="$(preset_flag CMAKE_CXX_FLAGS) $FLAGS_EXTRA"
export CFLAGS="$FLAGS_EXTRA" CXXFLAGS="$FLAGS_EXTRA"
cmake -S "$TREE" --preset arm64-android-snapdragon-release -B "$BDIR" \
    -DLLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER" \
    -DLLAMA_BUILD_COMMIT="$LLAMA_BUILD_COMMIT_SHORT" \
    -DCMAKE_C_FLAGS="$c_flags" \
    -DCMAKE_CXX_FLAGS="$cxx_flags"
# shellcheck disable=SC2086
cmake --build "$BDIR" -j"$JOBS" --target $TARGETS
cxx=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++
$cxx -O2 -std=c++17 $FLAGS_EXTRA -I"$TREE/include" -I"$TREE/common" -I"$TREE/src" -I"$TREE/ggml/include" \
    -I"$TREE/vendor" -I"$TREE/tools/mtmd" -I"$APP_SRC" tools/memprobe/memprobe.cpp "$APP_SRC/state_cache.cpp" \
    "$APP_SRC/cache_io.cpp" "$APP_SRC/spec_policy.cpp" -o "$BDIR/bin/memprobe" \
    -L"$BDIR/bin" -lmtmd -lllama-common -lllama -lggml -lggml-cpu -lggml-base -Wl,-rpath,"\$ORIGIN/../lib"
'
) 9> build/.container.lock > "$stage/build.log" 2>&1 || die "the build failed, refer to $stage/build.log"

# 3. The stage files and the comparison with the APK libraries.
rm -rf "$out"
mkdir -p "$out/bin" "$out/lib"
for lib in $LLAMA_LIBS llama-bench-impl; do
    cp -f "$bdir/bin/lib$lib.so" "$out/lib/"
done
cp -f "$bdir/ggml/src/ggml-hexagon/libggml-htp-v79.so" "$out/lib/"
cp -f "$bdir/bin/llama-bench" "$bdir/bin/memprobe" "$out/bin/"
cp -f tools/phone/gate.sh "$out/bin/"
(cd "$out" && sha256sum bin/* lib/* > SHA256SUMS)
cat "$out/SHA256SUMS"
for so in "$out"/lib/*.so; do
    name=$(basename "$so")
    want=$(grep -E "  $name\$" build/hashes-native.txt | cut -d' ' -f1 || true)
    [[ -n $want ]] || continue
    have=$(sha256sum "$so" | cut -d' ' -f1)
    if [[ $have == "$want" ]]; then
        echo "bench-kv: $name has the bytes of the APK library"
    else
        echo "bench-kv: $name differs from the APK library"
    fi
done
