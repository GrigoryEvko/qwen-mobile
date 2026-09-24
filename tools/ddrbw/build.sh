#!/usr/bin/env bash
# Build the DDR read probe in the Snapdragon container: the Android host program ddrbw and the DSP
# library libddrbw_skel.so for each Hexagon version.
#
#   tools/ddrbw/build.sh [VERSION...]
#
# VERSION is v73, v75, v79 or v81. The default is v79, the DSP of the phone. The container runs
# under the lock build/.container.lock, as each container build of the repository does.
#
# Output (build/ddrbw/, not in git):
#   bin/ddrbw                    The Android arm64 host program (API level 29 or a subsequent level)
#   <version>/libddrbw_skel.so   The DSP library for that Hexagon version
#   <version>/skel.disasm.txt    The disassembly of the library
#   cmake/                       The CMake build tree
#   build.log                    The output of the container
#   hashes.txt                   The SHA-256 of each output
#
# The DSP library compiles htp/dma-queue.c of third_party/llama.cpp with the toolchain file of the
# backend, thus build the probe from a tree with the patches applied (the submodule of the box).
#
# To run it on the phone: push bin/ddrbw and <version>/libddrbw_skel.so to one directory, and
# start "ADSP_LIBRARY_PATH=<dir> <dir>/ddrbw nsp --votes backend". The header of host/ddrbw.c
# gives the modes. Time: about 2 minutes.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../scripts/lib.sh"

readonly OUT_REL=build/ddrbw
readonly SRC_REL=tools/ddrbw

versions=("$@")
if [[ ${#versions[@]} -eq 0 ]]; then
    versions=(v79)
fi
for v in "${versions[@]}"; do
    case $v in
        v73 | v75 | v79 | v81) ;;
        *) die "the version $v is not one of v73, v75, v79, v81" ;;
    esac
done
dsps=$(IFS=';'; echo "${versions[*]}")

cd "$REPO_ROOT"
[[ -f third_party/llama.cpp/ggml/src/ggml-hexagon/htp/dma-queue.c ]] ||
    die "third_party/llama.cpp has no Hexagon backend: run 'git submodule update --init'"

rm -rf "$OUT_REL/cmake" "$OUT_REL/bin" "$OUT_REL"/v73 "$OUT_REL"/v75 "$OUT_REL"/v79 "$OUT_REL"/v81 "$OUT_REL/hashes.txt"
mkdir -p "$OUT_REL"

(
    flock 9
    container_run \
        -e DSPS="$dsps" \
        -e OUT_REL="$OUT_REL" \
        -e SRC_REL="$SRC_REL" \
        -e CFLAGS="-ffile-prefix-map=/workspace=." \
        "$SNAPDRAGON_IMAGE" bash -euo pipefail -c '
out=/workspace/$OUT_REL
cmake -S "$SRC_REL" -B "$OUT_REL/cmake" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-29 \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
    -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" \
    -DHEXAGON_TOOLS_ROOT="$HEXAGON_TOOLS_ROOT" \
    -DPREBUILT_LIB_DIR=android_aarch64 \
    -DDDRBW_DSPS="$DSPS" \
    -DDDRBW_OUT="$out" \
    -DLLAMA_DIR=/workspace/third_party/llama.cpp
cmake --build "$OUT_REL/cmake" -j"$(nproc)"
cmake --install "$OUT_REL/cmake" > /dev/null
tools=$HEXAGON_TOOLS_ROOT/Tools/bin
IFS=";" read -r -a list <<< "$DSPS"
for v in "${list[@]}"; do
    lib=$out/$v/libddrbw_skel.so
    [[ -f $lib ]] || { echo "ddrbw: error: $lib was not built" >&2; exit 1; }
    "$tools/hexagon-llvm-objdump" -d --no-show-raw-insn --mcpu=hexagon$v \
        --mattr=+hvx$v,+hvx-length128b "$lib" > "$out/$v/skel.disasm.txt"
done
'
) 9> build/.container.lock > "$OUT_REL/build.log" 2>&1 || die "the build failed, refer to $OUT_REL/build.log"

{
    sha256_table "$OUT_REL/bin/ddrbw"
    for v in "${versions[@]}"; do
        printf '%s  %s/libddrbw_skel.so\n' "$(sha256sum "$OUT_REL/$v/libddrbw_skel.so" | cut -d' ' -f1)" "$v"
    done
} > "$OUT_REL/hashes.txt"
echo "ddrbw: $OUT_REL/hashes.txt"
cat "$OUT_REL/hashes.txt"
