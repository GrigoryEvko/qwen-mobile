#!/usr/bin/env bash
# Build the device program hmx_sustain.so (src/hmx_sustain.c) for run_main_on_hexagon, inside the
# pinned toolchain image. The flags and the lab runtime (src/dsp_lab.c) come from dsp-lib.sh. No
# vendor library is linked.
#
#   tools/hmx-bench/build-sustain.sh [out_dir]
#
# ARCH selects the core (v79 by default). The output directory is relative to the repository, and
# its preset value is build/hmx-bench/sustain-<ARCH>. The stage sweep uses build/sweep/phone/hmx.
# To run the program, put it next to run_main_on_hexagon and the skel of build-i8.sh (refer to that
# script and to tools/README.md for the .farf mask files).
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/dsp-lib.sh"

out_rel=${1:-build/hmx-bench/sustain-$HMX_ARCH}

build_dsp_programs "$out_rel" hmx_sustain=tools/hmx-bench/src/hmx_sustain.c
ls -la "$REPO_ROOT/$out_rel/hmx_sustain.so"
