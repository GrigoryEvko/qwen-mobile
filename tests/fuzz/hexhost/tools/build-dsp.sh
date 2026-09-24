#!/usr/bin/env bash
# Build the DSP libraries of one private llama.cpp tree in the Snapdragon container, with the
# release preset of the app. The build is not byte-reproducible: it is for diagnostic runs.
#
#   tests/fuzz/hexhost/tools/build-dsp.sh TREE OUTDIR [JOBS] [DSPS]
#
# TREE and OUTDIR are paths relative to the repository root. DSPS is a list such as "v79"
# (the preset value) or "v73 v75 v79 v81". The libraries are OUTDIR/ggml/src/ggml-hexagon/
# libggml-htp-<dsp>.so.
#
# tests/fuzz/ops/run.sh phone-build also builds a v79 DSP library (FUZZ_OPS_BUILD_DSP=1), but with
# the CMake of tests/fuzz/ops and its profile, not with the preset of the app.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
source scripts/lib.sh
tree=$1
out=$2
jobs=${3:-8}
dsps=${4:-v79}
[[ -f $tree/CMakeLists.txt ]] || die "no llama.cpp tree at $tree"
cp android/snapdragon/CMakeUserPresets.json "$tree/CMakeUserPresets.json"
targets=""
for d in $dsps; do
    targets+=" htp-$d"
done
mkdir -p "$out"
container_run -e T="$tree" -e O="$out" -e J="$jobs" -e TARGETS="$targets" "$SNAPDRAGON_IMAGE" bash -euo pipefail -c '
cmake -S "$T" --preset arm64-android-snapdragon-release -B "$O" > "$O/configure.log" 2>&1
# shellcheck disable=SC2086
cmake --build "$O" -j"$J" --target $TARGETS > "$O/build.log" 2>&1
'
for d in $dsps; do
    sha256sum "$out/ggml/src/ggml-hexagon/libggml-htp-$d.so"
done
