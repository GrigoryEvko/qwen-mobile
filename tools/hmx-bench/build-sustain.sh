#!/usr/bin/env bash
# Build the device program hmx_sustain.so (src/hmx_sustain.c) for run_main_on_hexagon, inside the pinned
# toolchain image and under the container lock. The flags and the lab runtime (src/dsp_lab.c) are those
# of build-i8.sh. No vendor library is linked.
#
#   tools/hmx-bench/build-sustain.sh [out_dir]
#
# ARCH selects the core (v79 by default). The output directory is relative to the repository, and its
# preset value is build/hmx-bench/sustain-<ARCH>. The stage sweep uses build/sweep/phone/hmx. To run the
# program, put it next to run_main_on_hexagon and the skel of build-i8.sh (refer to that script and to
# tools/README.md for the .farf mask files).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
ARCH="${ARCH:-v79}"
OUT_REL="${1:-build/hmx-bench/sustain-${ARCH}}"
IMAGE="ghcr.io/snapdragon-toolchain/arm64-android:v0.7"
mkdir -p "${REPO}/${OUT_REL}" "${REPO}/build"

(
    flock 9
    podman run --rm --userns=keep-id --security-opt label=disable -v "${REPO}:/repo" -w /repo "${IMAGE}" bash -euc '
SDK=$HEXAGON_SDK_ROOT
TOOLS=$HEXAGON_TOOLS_ROOT
VAR=$DEFAULT_TOOLS_VARIANT
ARCH='"${ARCH}"'
OUT=/repo/'"${OUT_REL}"'
TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT
INC="-I$SDK/rtos/qurt/compute${ARCH}/include -I$SDK/rtos/qurt/compute${ARCH}/include/qurt
     -I$SDK/rtos/qurt/compute${ARCH}/include/posix -I$SDK/ipc/fastrpc/rtld/ship/hexagon_${VAR}_${ARCH}
     -I$SDK/ipc/fastrpc/rpcmem/inc -I$SDK/rtos/qurt -I$SDK/utils/examples
     -isystem $SDK/incs -isystem $SDK/incs/stddef -isystem $SDK/ipc/fastrpc/incs
     -I/repo/tools/htp-lab/lab"
CFLAGS="-m${ARCH} -G0 -Wall -Werror -Wno-unused-function -fno-zero-initialized-in-bss -fdata-sections
        -fpic -fPIC -mhvx -mhvx-length=128B -mhmx -O2 -DLAB_DEVICE=1"
LDFLAGS="-m${ARCH} -G0 -fpic -Wl,-Bsymbolic -Wl,-L$TOOLS/Tools/target/hexagon/lib/${ARCH}/G0/pic
         -Wl,-L$TOOLS/Tools/target/hexagon/lib/ -Wl,--no-threads -Wl,--wrap=malloc -Wl,--wrap=calloc
         -Wl,--wrap=free -Wl,--wrap=realloc -Wl,--wrap=memalign -shared"
CC=$TOOLS/Tools/bin/hexagon-clang
$CC $INC $CFLAGS -c /repo/tools/hmx-bench/src/dsp_lab.c -o $TMP/dsp_lab.o
$CC $INC $CFLAGS -c /repo/tools/hmx-bench/src/hmx_sustain.c -o $TMP/hmx_sustain.o
$CC $LDFLAGS -o $OUT/hmx_sustain.so -Wl,-soname,hmx_sustain.so -Wl,--start-group $TMP/hmx_sustain.o $TMP/dsp_lab.o -Wl,--end-group -lc
'
) 9> "${REPO}/build/.container.lock"
ls -la "${REPO}/${OUT_REL}/hmx_sustain.so"
