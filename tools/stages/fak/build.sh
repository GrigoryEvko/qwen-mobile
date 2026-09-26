#!/usr/bin/env bash
# Build the phone stage "fak" or "fak2": the flash attention of HTP0 (the chunk cost model, the tile-layout
# kernel, the resident K and V and the decode spans) against the flash attention of HEAD, on one
# library set. The switch GGML_HEXAGON_FA_OPT (patches/hexagon-fa) selects the parts at run time.
#
#   JOBS=24 [STAGE=fak2] tools/stages/fak/build.sh
#
# STAGE names the stage of stage.py (fak, the preset, or fak2). The files of the stage go to build/STAGE, and
# the candidate patches come from build/STAGE/patches. build/fak/build.sh is a link to this file.
#
# The steps:
#   1. tests/sanitizers/llama-copy.sh makes build/STAGE/tree, the llama.cpp tree of HEAD (the pin plus
#      patches/series). Then git apply puts each patch of build/STAGE/patches (the candidate patches,
#      in the order of their names) on the tree. A candidate that is in patches/series already is
#      not in build/STAGE/patches.
#   2. In the Snapdragon container, under the lock build/.container.lock, CMake configures the tree
#      into build/STAGE/android with the preset, the compiler flags (with -flto) and the build number
#      and commit of scripts/build-native.sh. It builds the libraries of the app (LLAMA_LIBS of
#      scripts/lib.sh), the DSP library v79, llama-bench, llama-perplexity and test-backend-ops. The
#      NDK clang++ compiles tools/memprobe/kvkl.cpp, as tools/memprobe/build-phone.sh does.
#   3. The script copies the files of the stage into build/STAGE/phone, and stage.py files writes the
#      test files, SHA256SUMS and build/STAGE/phone-commands.txt.
#
# Time: about 15 minutes with JOBS=24 (the LTO links take most of it). Disk: about 2 GB.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../scripts/lib.sh"
JOBS=${JOBS:-24}
STAGE=${STAGE:-fak}
cd "$REPO_ROOT"
source tools/stages/common/buildlib.sh

readonly stage=build/$STAGE
readonly tree=$stage/tree
readonly bdir=$stage/android
readonly out=$stage/phone

# 1. The tree. The tree is below build/, which the repository ignores, and git apply inside the
# repository skips the files of an ignored path with the exit code 0. Thus the tree is its own git
# repository while the patches apply, as in tests/sanitizers/llama-copy.sh.
tests/sanitizers/llama-copy.sh "$tree"
git -C "$tree" init -q
shopt -s nullglob
for p in "$stage"/patches/*.patch; do
    git -C "$tree" apply --whitespace=nowarn "$REPO_ROOT/$p" || die "$p does not apply to $tree"
    echo "$STAGE: applied $p"
done
shopt -u nullglob
rm -rf "$tree/.git"
cp -f android/snapdragon/CMakeUserPresets.json "$tree/CMakeUserPresets.json"

# 2. The build.
mkdir -p "$bdir"
echo "$STAGE: SOURCE_DATE_EPOCH=$(source_date_epoch) JOBS=$JOBS"
(
    flock 9
    # shellcheck disable=SC2016
    stage_build --tests --no-remap "$tree" "$bdir" "$LLAMA_LIBS htp-v79 llama-bench llama-perplexity test-backend-ops" '
"$(ndk_cxx)" -O2 -std=c++17 $FLAGS_EXTRA -I"$TREE/include" -I"$TREE/common" -I"$TREE/ggml/include" -I"$TREE/vendor" \
    tools/memprobe/kvkl.cpp -o "$BDIR/bin/kvkl" -L"$BDIR/bin" -lllama-common -lllama -lggml -lggml-base \
    -Wl,-rpath,"\$ORIGIN/../lib"'
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
tools/stages/fak/stage.py --stage "$STAGE" files
