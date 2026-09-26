#!/usr/bin/env bash
# Build the device programs of the int8 HMX work for run_main_on_hexagon, inside the pinned
# toolchain image. No vendor library is linked.
#
#   tools/hmx-bench/build-i8.sh [out_dir]
#
# Programs (each is a shared object with main()):
#   i8hello.so  the smallest check: the runtime starts, the HVX runs, one int8 product reads back
#   i8read.so   the cost of the int32 accumulator read (src/i8read.c)
#   i8probe.so  the bytecode interpreter of tools/htp-lab/lab/target_i8probe.c, thus one operation
#               stream runs in the simulator and on the phone
#
# ARCH selects the core (v79 by default). The output directory is relative to the repository, and
# its preset value is build/hmx-bench/i8-<ARCH>. The script copies run_main_on_hexagon of the SDK
# and tools/hmx-bench/build/librun_main_on_hexagon_skel.so next to the programs.
#
# The SDK does not supply librun_main_on_hexagon_skel.so. Make it one time from the SDK directory
# libs/run_main_on_hexagon: run "qaic -mdll" on inc/run_main_on_hexagon.idl, compile the skel that
# qaic writes together with src/run_main_on_hexagon_dsp.c, and link the two with -shared.
#
# To run a program, push it with run_main_on_hexagon, the skel and the .farf mask files to one
# directory of the phone, and start "ADSP_LIBRARY_PATH=<dir> ./run_main_on_hexagon 3 <program>.so".
# Refer to tools/README.md for the mask files.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/dsp-lib.sh"

out_rel=${1:-build/hmx-bench/i8-$HMX_ARCH}

build_dsp_programs "$out_rel" \
    i8hello=tools/hmx-bench/src/i8hello.c \
    i8read=tools/hmx-bench/src/i8read.c \
    i8probe=tools/htp-lab/lab/target_i8probe.c
copy_dsp_runner "$out_rel"
ls -la "$REPO_ROOT/$out_rel"
