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

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
ARCH="${ARCH:-v79}"
OUT_REL="${1:-build/hmx-bench/i8-${ARCH}}"
IMAGE="ghcr.io/snapdragon-toolchain/arm64-android:v0.7"
SKEL="${HERE}/build/librun_main_on_hexagon_skel.so"
[ -f "${SKEL}" ] || { echo "error: ${SKEL} is missing. Make it from the SDK as the header of this script tells." >&2; exit 1; }
mkdir -p "${REPO}/${OUT_REL}"

podman run --rm --userns=keep-id --security-opt label=disable -v "${REPO}:/repo" -w /repo "${IMAGE}" bash -euc '
SDK=$HEXAGON_SDK_ROOT
TOOLS=$HEXAGON_TOOLS_ROOT
VAR=$DEFAULT_TOOLS_VARIANT
ARCH='"${ARCH}"'
OUT=/repo/'"${OUT_REL}"'
INC="-I$SDK/rtos/qurt/compute${ARCH}/include -I$SDK/rtos/qurt/compute${ARCH}/include/qurt
     -I$SDK/rtos/qurt/compute${ARCH}/include/posix -I$SDK/ipc/fastrpc/rtld/ship/hexagon_${VAR}_${ARCH}
     -I$SDK/ipc/fastrpc/rpcmem/inc -I$SDK/rtos/qurt -I$SDK/utils/examples
     -isystem $SDK/incs -isystem $SDK/incs/stddef -isystem $SDK/ipc/fastrpc/incs
     -I/repo/tools/htp-lab/lab"
CFLAGS="-m${ARCH} -G0 -Wall -Wno-unused-function -fno-zero-initialized-in-bss -fdata-sections
        -fpic -fPIC -mhvx -mhvx-length=128B -mhmx -O2 -DLAB_DEVICE=1"
LDFLAGS="-m${ARCH} -G0 -fpic -Wl,-Bsymbolic -Wl,-L$TOOLS/Tools/target/hexagon/lib/${ARCH}/G0/pic
         -Wl,-L$TOOLS/Tools/target/hexagon/lib/ -Wl,--no-threads -Wl,--wrap=malloc -Wl,--wrap=calloc
         -Wl,--wrap=free -Wl,--wrap=realloc -Wl,--wrap=memalign -shared"
CC=$TOOLS/Tools/bin/hexagon-clang
$CC $INC $CFLAGS -c /repo/tools/hmx-bench/src/dsp_lab.c -o $OUT/dsp_lab.o
$CC $INC $CFLAGS -c /repo/tools/hmx-bench/src/i8read.c -o $OUT/i8read.o
$CC $INC $CFLAGS -c /repo/tools/htp-lab/lab/target_i8probe.c -o $OUT/i8probe.o
$CC $INC $CFLAGS -c /repo/tools/hmx-bench/src/i8hello.c -o $OUT/i8hello.o
$CC $LDFLAGS -o $OUT/i8hello.so -Wl,-soname,i8hello.so -Wl,--start-group $OUT/i8hello.o $OUT/dsp_lab.o -Wl,--end-group -lc
$CC $LDFLAGS -o $OUT/i8read.so -Wl,-soname,i8read.so -Wl,--start-group $OUT/i8read.o $OUT/dsp_lab.o -Wl,--end-group -lc
$CC $LDFLAGS -o $OUT/i8probe.so -Wl,-soname,i8probe.so -Wl,--start-group $OUT/i8probe.o $OUT/dsp_lab.o -Wl,--end-group -lc
rm -f $OUT/*.o
install -m 755 $SDK/libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon $OUT/run_main_on_hexagon
'
cp "${SKEL}" "${REPO}/${OUT_REL}/"
ls -la "${REPO}/${OUT_REL}"
