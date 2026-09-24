#!/usr/bin/env bash
# Build the phone stage "fak": the flash attention of HTP0 (the chunk cost model, the tile-layout
# kernel, the resident K and V and the decode spans) against the flash attention of HEAD, on one
# library set. The switch GGML_HEXAGON_FA_OPT (patches/hexagon-fa) selects the parts at run time.
#
#   JOBS=24 tools/stages/fak/build.sh
#
# build/fak/build.sh is a link to this file. The files of the stage go to build/fak.
#
# The steps:
#   1. tests/sanitizers/llama-copy.sh makes build/fak/tree, the llama.cpp tree of HEAD (the pin plus
#      patches/series). Then git apply puts each patch of build/fak/patches (the candidate patches,
#      in the order of their names) on the tree. A candidate that is in patches/series already is
#      not in build/fak/patches.
#   2. In the Snapdragon container, under the lock build/.container.lock, CMake configures the tree
#      into build/fak/android with the preset, the compiler flags (with -flto) and the build number
#      and commit of scripts/build-native.sh. It builds the libraries of the app (LLAMA_LIBS of
#      scripts/lib.sh), the DSP library v79, llama-bench, llama-perplexity and test-backend-ops. The
#      NDK clang++ compiles tools/memprobe/kvkl.cpp, as tools/memprobe/build-phone.sh does.
#   3. The script copies the files of the stage into build/fak/phone, and stage.py files writes the
#      test files, SHA256SUMS and build/fak/phone-commands.txt.
#
# Time: about 15 minutes with JOBS=24 (the LTO links take most of it). Disk: about 2 GB.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../scripts/lib.sh"
JOBS=${JOBS:-24}
cd "$REPO_ROOT"

readonly stage=build/fak
readonly tree=$stage/tree
readonly bdir=$stage/android
readonly out=$stage/phone
readonly repro="-ffile-prefix-map=/workspace=. -fdebug-prefix-map=/workspace=. -Werror=date-time"

# 1. The tree. The tree is below build/, which the repository ignores, and git apply inside the
# repository skips the files of an ignored path with the exit code 0. Thus the tree is its own git
# repository while the patches apply, as in tests/sanitizers/llama-copy.sh.
tests/sanitizers/llama-copy.sh "$tree"
git -C "$tree" init -q
shopt -s nullglob
for p in "$stage"/patches/*.patch; do
    git -C "$tree" apply --whitespace=nowarn "$REPO_ROOT/$p" || die "$p does not apply to $tree"
    echo "fak: applied $p"
done
shopt -u nullglob
rm -rf "$tree/.git"
cp -f android/snapdragon/CMakeUserPresets.json "$tree/CMakeUserPresets.json"

# 2. The build.
mkdir -p "$bdir"
SOURCE_DATE_EPOCH=$(source_date_epoch)
echo "fak: SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH JOBS=$JOBS"
(
    flock 9
    container_run \
        -e SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
        -e LLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER" \
        -e LLAMA_BUILD_COMMIT_SHORT="${LLAMA_COMMIT:0:7}" \
        -e TARGETS="$LLAMA_LIBS htp-v79 llama-bench llama-perplexity test-backend-ops" \
        -e JOBS="$JOBS" \
        -e FLAGS_EXTRA="$repro" \
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
    -DLLAMA_BUILD_TESTS=ON \
    -DCMAKE_C_FLAGS="$c_flags" \
    -DCMAKE_CXX_FLAGS="$cxx_flags"
# shellcheck disable=SC2086
cmake --build "$BDIR" -j"$JOBS" --target $TARGETS
cxx=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++
$cxx -O2 -std=c++17 $FLAGS_EXTRA -I"$TREE/include" -I"$TREE/common" -I"$TREE/ggml/include" -I"$TREE/vendor" \
    tools/memprobe/kvkl.cpp -o "$BDIR/bin/kvkl" -L"$BDIR/bin" -lllama-common -lllama -lggml -lggml-base \
    -Wl,-rpath,"\$ORIGIN/../lib"
'
) 9> build/.container.lock > "$stage/build.log" 2>&1 || die "the build failed, refer to $stage/build.log"

# 3. The stage files.
rm -rf "$out"
mkdir -p "$out/bin" "$out/lib"
for lib in $LLAMA_LIBS; do
    cp -f "$bdir/bin/lib$lib.so" "$out/lib/"
done
cp -f "$bdir/ggml/src/ggml-hexagon/libggml-htp-v79.so" "$out/lib/"
for tool in llama-bench llama-perplexity test-backend-ops kvkl; do
    cp -f "$bdir/bin/$tool" "$out/bin/"
    impl="$bdir/bin/lib$tool-impl.so"
    [[ -f $impl ]] && cp -f "$impl" "$out/lib/"
done
cp -f tools/phone/gate.sh "$out/bin/"
tools/stages/fak/stage.py files
