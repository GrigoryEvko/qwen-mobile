#!/usr/bin/env bash
# Build llama.cpp tools for the phone (arm64 Android) from one llama.cpp tree in the Snapdragon
# container, with the release preset of the app. The build is not byte-reproducible.
#
#   tests/fuzz/hexhost/tools/build-llama-tools.sh TREE OUTDIR JOBS TARGET...
#
# TREE and OUTDIR are paths relative to the repository root. The binaries are in OUTDIR/bin.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
source scripts/lib.sh
tree=$1
out=$2
jobs=$3
shift 3
targets="$*"
[[ -f $tree/CMakeLists.txt ]] || die "no llama.cpp tree at $tree"
cp android/snapdragon/CMakeUserPresets.json "$tree/CMakeUserPresets.json"
mkdir -p "$out"
container_run -e T="$tree" -e O="$out" -e J="$jobs" -e TARGETS="$targets" "$SNAPDRAGON_IMAGE" bash -euo pipefail -c '
cmake -S "$T" --preset arm64-android-snapdragon-release -B "$O" > "$O/configure.log" 2>&1
# shellcheck disable=SC2086
cmake --build "$O" -j"$J" --target $TARGETS > "$O/build.log" 2>&1
'
ls -la "$out/bin"
