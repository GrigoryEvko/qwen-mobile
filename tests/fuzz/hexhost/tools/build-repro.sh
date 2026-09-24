#!/usr/bin/env bash
# Build llama.cpp libraries of a private tree byte-reproducibly, as scripts/build-native.sh builds
# the submodule: the tree appears at /workspace/third_party/llama.cpp and the build directory at
# /workspace/build/native/llama inside the container, with the same flags and environment.
# The private tree must be a git tree at the pinned commit with core.abbrev 7.
#
#   tests/fuzz/hexhost/tools/build-repro.sh TREE OUTDIR JOBS TARGET...
#
# TREE and OUTDIR are paths relative to the repository root.
#
# tools/stages/bench-kv/build.sh gets the same paths in the objects with prefix maps, from a tree of
# tests/sanitizers/llama-copy.sh that is not a git tree. This script mounts the tree at the paths
# of the app build instead.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
source scripts/lib.sh
readonly REPRO_FLAGS="-ffile-prefix-map=/workspace=. -fdebug-prefix-map=/workspace=. -Werror=date-time"
tree=$1
out=$2
jobs=$3
shift 3
targets="$*"
[[ -f $tree/CMakeLists.txt ]] || die "no llama.cpp tree at $tree"
[[ $(git -C "$tree" config core.abbrev) == 7 ]] || die "$tree needs core.abbrev 7"
cp android/snapdragon/CMakeUserPresets.json "$tree/CMakeUserPresets.json"
mkdir -p "$out"
SOURCE_DATE_EPOCH=$(source_date_epoch)
container_run \
    -v "$PWD/$tree:/workspace/third_party/llama.cpp" \
    -v "$PWD/$out:/workspace/build/native/llama" \
    -e SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
    -e LLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER" \
    -e LLAMA_BUILD_COMMIT_SHORT="${LLAMA_COMMIT:0:7}" \
    -e TARGETS="$targets" \
    -e JOBS="$jobs" \
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
    -DCMAKE_CXX_FLAGS="$cxx_flags" > build/native/llama/configure.log 2>&1
# shellcheck disable=SC2086
cmake --build build/native/llama -j"$JOBS" --target $TARGETS > build/native/llama/build.log 2>&1
'
sha256sum "$out"/bin/*.so "$out"/ggml/src/ggml-hexagon/libggml-htp-*.so 2> /dev/null || true
