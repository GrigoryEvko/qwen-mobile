#!/usr/bin/env bash
# Build the phone stage "gdnk": the libraries of the app from a private tree of HEAD plus the patches of
# the stage, and llama-bench, llama-perplexity and test-backend-ops against them. The three switches of
# the patches (GGML_HEXAGON_GDN_CONV_DMA, GGML_HEXAGON_GDN_CHUNK, GGML_HEXAGON_GDN_QKNORM) give the old
# and the new paths on the same libraries.
#
#   JOBS=24 [GDNK_STAGE=build/gdnk] tools/stages/gdnk/build.sh PATCH...
#
# PATCH is a llama.cpp patch file that is not yet in patches/ (wip/gdnk/NNNN-*.patch, a path relative to
# the root of the repository), applied in the order of the command line. build/gdnk/build.sh is a link
# to this file. The files of the stage go to GDNK_STAGE (the preset value is build/gdnk). A second stage
# directory keeps the files of a stage that the phone runs while a different set builds.
#
# The steps:
#   1. tests/sanitizers/llama-copy.sh makes GDNK_STAGE/src, the llama.cpp tree of HEAD (the pin plus
#      patches/series). git apply puts each PATCH on it, inside a git repository of its own: git apply
#      in the repository around build/ skips the files below an ignored path and still exits 0.
#   2. In the Snapdragon container, under the lock build/.container.lock, CMake configures the tree into
#      GDNK_STAGE/android with the preset, the flags (with -flto) and the build number and commit of
#      scripts/build-native.sh, and builds the libraries of the app (LLAMA_LIBS of scripts/lib.sh), the DSP
#      library v79, llama-bench, llama-perplexity and test-backend-ops.
#   3. The script copies the files of the stage into GDNK_STAGE/phone and writes SHA256SUMS.
#
# Time: about 15 minutes with JOBS=24 (the LTO links take most of it). Disk: about 2 GB.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../scripts/lib.sh"
JOBS=${JOBS:-24}
cd "$REPO_ROOT"

[[ $# -ge 1 ]] || die "give the patch files of the stage, in order"
for p in "$@"; do
    [[ -f $p ]] || die "no patch file $p"
done

readonly stage=${GDNK_STAGE:-build/gdnk}
[[ $stage == build/* ]] || die "GDNK_STAGE must be a directory below build/, not $stage"
readonly tree=$stage/src
readonly bdir=$stage/android
readonly out=$stage/phone
readonly repro="-ffile-prefix-map=/workspace=. -fdebug-prefix-map=/workspace=. -Werror=date-time"
readonly remap="-ffile-prefix-map=/workspace/$tree=./third_party/llama.cpp -ffile-prefix-map=/workspace/$bdir=./build/native/llama"

# 1. The tree and the patches of the stage.
tests/sanitizers/llama-copy.sh "$tree"
git -C "$tree" init -q
for p in "$@"; do
    git -C "$tree" apply --whitespace=nowarn "$REPO_ROOT/$p" || die "$p does not apply on the tree of HEAD"
    echo "gdnk: applied $p"
done
rm -rf "$tree/.git"
cp -f android/snapdragon/CMakeUserPresets.json "$tree/CMakeUserPresets.json"

# 2. The build.
mkdir -p "$bdir"
SOURCE_DATE_EPOCH=$(source_date_epoch)
echo "gdnk: SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH JOBS=$JOBS"
(
    flock 9
    container_run \
        -e SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
        -e LLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER" \
        -e LLAMA_BUILD_COMMIT_SHORT="${LLAMA_COMMIT:0:7}" \
        -e TARGETS="$LLAMA_LIBS htp-v79 llama-bench llama-perplexity test-backend-ops" \
        -e JOBS="$JOBS" \
        -e FLAGS_EXTRA="$repro $remap" \
        -e TREE="$tree" -e BDIR="$bdir" \
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
'
) 9> build/.container.lock > "$stage/build.log" 2>&1 || die "the build failed, refer to $stage/build.log"

# 3. The stage files.
rm -rf "$out"
mkdir -p "$out/bin" "$out/lib"
for lib in $LLAMA_LIBS llama-bench-impl llama-perplexity-impl; do
    cp -f "$bdir/bin/lib$lib.so" "$out/lib/"
done
cp -f "$bdir/ggml/src/ggml-hexagon/libggml-htp-v79.so" "$out/lib/"
cp -f "$bdir/bin/llama-bench" "$bdir/bin/llama-perplexity" "$bdir/bin/test-backend-ops" "$out/bin/"
cp -f tools/phone/gate.sh "$out/bin/"
(cd "$out" && sha256sum bin/* lib/* > SHA256SUMS)
cat "$out/SHA256SUMS"
