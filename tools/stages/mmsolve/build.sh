#!/usr/bin/env bash
# Build the phone stage "mmsolve": the chunks of the HMX 2D matmul with the old cost model and with the cost
# of the kernel (GGML_HEXAGON_MM_SOLVER), in one library set.
#
#   JOBS=24 tools/stages/mmsolve/build.sh PATCH
#
# PATCH is the llama.cpp patch of the chunk solver (a patch file of patches/ or its draft). The files of the
# stage go to build/mmsolve.
#
# The steps:
#   1. tests/sanitizers/llama-copy.sh makes build/mmsolve/stage-src, the llama.cpp tree of HEAD (the pin plus
#      patches/series). GNU patch puts PATCH on it. The copy is not a git repository, and git apply in a
#      directory below an ignored path of this repository skips files and still exits with 0.
#   2. clang++ of the box builds tools/stages/mmsolve/plan.cpp against the tree, and "plan check" must pass.
#      "plan table" writes build/mmsolve/plan-table.txt.
#   3. In the Snapdragon container, under the lock build/.container.lock, CMake configures the tree into
#      build/mmsolve/android with the preset, the compiler flags (with -flto) and the build number and commit
#      of scripts/build-native.sh. It builds the libraries of the app (LLAMA_LIBS of scripts/lib.sh), the DSP
#      library v79, llama-bench and test-backend-ops. The NDK clang++ compiles tools/stages/mmsolve/mmcheck.cpp.
#   4. The script copies the files of the stage into build/mmsolve/phone and writes SHA256SUMS.
#   5. tools/stages/mmsolve/stage.py files writes the test files and build/mmsolve/phone-commands.txt.
#
# Time: about 15 minutes with JOBS=24 (the LTO links take most of it). Disk: about 2 GB.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../scripts/lib.sh"
JOBS=${JOBS:-24}
cd "$REPO_ROOT"

patch_file=${1:?usage: tools/stages/mmsolve/build.sh PATCH}
[[ -f $patch_file ]] || die "no patch file $patch_file"
patch_file=$(readlink -f "$patch_file")

readonly stage=build/mmsolve
readonly tree=$stage/stage-src
readonly bdir=$stage/android
readonly out=$stage/phone
readonly repro="-ffile-prefix-map=/workspace=. -fdebug-prefix-map=/workspace=. -Werror=date-time"
readonly remap="-ffile-prefix-map=/workspace/$tree=./third_party/llama.cpp -ffile-prefix-map=/workspace/$bdir=./build/native/llama"

# 1. The tree
mkdir -p "$stage"
tests/sanitizers/llama-copy.sh "$tree"
patch -p1 -N --dry-run --silent -d "$tree" < "$patch_file" > /dev/null || die "$patch_file does not apply to $tree"
patch -p1 -N --silent --no-backup-if-mismatch -d "$tree" < "$patch_file" || die "patch failed in $tree"
grep -q htp_mm_hmx_solve_2d_cost "$tree/ggml/src/ggml-hexagon/htp/matmul-ops.h" \
    || die "$tree has no htp_mm_hmx_solve_2d_cost after the patch"
cp -f android/snapdragon/CMakeUserPresets.json "$tree/CMakeUserPresets.json"
sha256sum "$patch_file" | sed "s|  .*|  $(basename "$patch_file")|" > "$stage/patch.sha256"

# 2. The plan and its checks, on the box
mkdir -p "$stage/bin"
clang++ -std=c++17 -O2 -Wall -Wextra -I"$tree/ggml/src/ggml-hexagon/htp" -I"$tree/ggml/src/ggml-hexagon" \
    -I"$tree/ggml/include" tools/stages/mmsolve/plan.cpp -o "$stage/bin/plan"
"$stage/bin/plan" check || die "plan check failed: the new chunk model breaks an invariant"
"$stage/bin/plan" table > "$stage/plan-table.txt"

# 3. The build
mkdir -p "$bdir"
SOURCE_DATE_EPOCH=$(source_date_epoch)
echo "mmsolve: SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH JOBS=$JOBS"
(
    flock 9
    container_run \
        -e SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
        -e LLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER" \
        -e LLAMA_BUILD_COMMIT_SHORT="${LLAMA_COMMIT:0:7}" \
        -e TARGETS="$LLAMA_LIBS htp-v79 llama-bench test-backend-ops" \
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
cxx=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++
# mmcheck calls only the C API of ggml, thus it links the C++ runtime statically and does not load libc++_shared.so.
$cxx -O2 -std=c++17 -Wall -Wextra $FLAGS_EXTRA -I"$TREE/ggml/include" tools/stages/mmsolve/mmcheck.cpp \
    -o "$BDIR/bin/mmcheck" -static-libstdc++ -L"$BDIR/bin" -lggml -lggml-cpu -lggml-base -Wl,-rpath,"\$ORIGIN/../lib"
'
) 9> build/.container.lock > "$stage/build.log" 2>&1 || die "the build failed, refer to $stage/build.log"

# 4. The stage files
rm -rf "$out"
mkdir -p "$out/bin" "$out/lib"
for lib in $LLAMA_LIBS llama-bench-impl; do
    cp -f "$bdir/bin/lib$lib.so" "$out/lib/"
done
cp -f "$bdir/ggml/src/ggml-hexagon/libggml-htp-v79.so" "$out/lib/"
cp -f "$bdir/bin/llama-bench" "$bdir/bin/test-backend-ops" "$bdir/bin/mmcheck" "$out/bin/"
cp -f tools/phone/gate.sh "$out/bin/"
(cd "$out" && sha256sum bin/* lib/* > SHA256SUMS)
cat "$out/SHA256SUMS"

# 5. The command file
tools/stages/mmsolve/stage.py files
