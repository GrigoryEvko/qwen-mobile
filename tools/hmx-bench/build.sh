#!/usr/bin/env bash
# Build the HMX rate program for the Hexagon v79 DSP, inside the pinned toolchain
# image. The program links libhexkl_micro.a of the Qualcomm HexKL package.
#
#   tools/hmx-bench/build.sh
#
# The HexKL package is click-through licensed. This script reads it from the
# session scratchpad and never copies it into the repository.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEXKL="${HEXKL_ROOT:?set HEXKL_ROOT to the unpacked hexkl_addon directory}"
IMAGE="ghcr.io/snapdragon-toolchain/arm64-android:v0.7"
ARCH="v79"
OUT="$HERE/build"

mkdir -p "$OUT"

podman run --rm \
    -v "$HERE:/work:z" \
    -v "$HEXKL:/hexkl:ro,z" \
    -w /work \
    "$IMAGE" bash -euc '
SDK=$HEXAGON_SDK_ROOT
TOOLS=$HEXAGON_TOOLS_ROOT
VAR=$DEFAULT_TOOLS_VARIANT
ARCH='"$ARCH"'
LIB=/hexkl/lib/$(basename $SDK)/hexagon_${VAR}_${ARCH}
test -f $LIB/libhexkl_micro.a || { echo "no libhexkl_micro.a at $LIB"; exit 1; }
echo "sdk $(basename $SDK) tools $VAR arch $ARCH"

$TOOLS/Tools/bin/hexagon-clang \
    -I$SDK/rtos/qurt/compute${ARCH}/include \
    -I$SDK/rtos/qurt/compute${ARCH}/include/qurt \
    -I$SDK/rtos/qurt/compute${ARCH}/include/posix \
    -I$SDK/ipc/fastrpc/rtld/ship/hexagon_${VAR}_${ARCH} \
    -I$SDK/ipc/fastrpc/rpcmem/inc \
    -I/hexkl/include \
    -I$SDK/rtos/qurt \
    -I$SDK/utils/examples \
    -isystem $SDK/incs \
    -isystem $SDK/incs/stddef \
    -isystem $SDK/ipc/fastrpc/incs \
    -m${ARCH} -G0 -Wall -Wno-unused-function \
    -fno-zero-initialized-in-bss -fdata-sections \
    -mllvm -enable-xqf-gen=true -fpic -mhvx -mhvx-length=128B -O3 \
    -fPIC -o build/hmx_rate.o -c src/hmx_rate.c

$TOOLS/Tools/bin/hexagon-clang -m${ARCH} -G0 -fpic \
    -Wl,-Bsymbolic \
    -Wl,-L$TOOLS/Tools/target/hexagon/lib/${ARCH}/G0/pic \
    -Wl,-L$TOOLS/Tools/target/hexagon/lib/ \
    -Wl,--no-threads -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=free \
    -Wl,--wrap=realloc -Wl,--wrap=memalign -shared \
    -o build/hmx_rate.so -Wl,-soname,hmx_rate.so \
    -Wl,--start-group build/hmx_rate.o $LIB/libhexkl_micro.a -Wl,--end-group -lc

cp $SDK/libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon build/
ls -la build/
'
