#!/bin/bash
# Build the native libraries of the app. This script is the previous entry point. It starts
# scripts/build-native.sh, which holds the full procedure.
#
#   android/snapdragon/build.sh
#
# The build always uses the submodule third_party/llama.cpp with the patch series applied.
# A llama.cpp checkout out of the repository is not permitted. Its source can be different
# from the series, and then the app and the measurements do not agree.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

if [[ -n "${LLAMA_CPP_DIR:-}" ]]; then
    echo "error: LLAMA_CPP_DIR=$LLAMA_CPP_DIR is set, but this build uses only $REPO/third_party/llama.cpp." >&2
    echo "       Remove the variable. Put a source change into patches/ and run scripts/apply-patches.sh." >&2
    exit 1
fi

exec "$REPO/scripts/build-native.sh" "$@"
