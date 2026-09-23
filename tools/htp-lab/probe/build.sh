#!/usr/bin/env bash
# Build the ISA silicon probe in the Snapdragon container: the Android host program isaprobe and
# one DSP library libisaprobe_skel.so for each Hexagon version.
#
#   tools/htp-lab/probe/build.sh [VERSION...]
#
# VERSION is v73, v75, v79 or v81. The default is the four versions. Each run removes the earlier
# output of the probe first, thus a new tools/htp-lab/isa/isa_kernels.c always replaces the stub.
#
# The census kernels: the DSP library compiles tools/htp-lab/isa/isa_kernels.c when that file
# exists, else tools/htp-lab/probe/isa_kernels_stub.c. The script tells which one it used.
#
# Output (build/isaprobe/, not in git):
#   bin/isaprobe                   The Android arm64 host program (API level 29 or a subsequent level)
#   <version>/libisaprobe_skel.so  The DSP library for that Hexagon version
#   <version>/skel.disasm.txt      The disassembly of the library
#   <version>/lab_ref.o            The census kernels compiled as the lab compiles them (no -fpic)
#   <version>/lab_ref.disasm.txt   Its disassembly
#   <version>/check_disasm.txt     The comparison of each census function (isa_core_<id>) of the
#                                  library with the same function of lab_ref.o
#   <version>/undefined.txt        The symbols that the DSP loader must find at load time
#   cmake/                         The CMake build tree
#   hashes.txt                     The SHA-256 of each output
#
# The build stops when a DSP library needs a lab_*, isa_* or __qf_* symbol that it does not have,
# because the DSP loader then rejects the library on the phone with no clear message. It also
# stops when a census function of the library has other instructions than the lab compile, because
# the chip result of that op then has no simulator result to compare with.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/lib.sh"

readonly OUT_REL=build/isaprobe
readonly PROBE_REL=tools/htp-lab/probe

versions=("$@")
if [[ ${#versions[@]} -eq 0 ]]; then
    versions=(v73 v75 v79 v81)
fi
for v in "${versions[@]}"; do
    case $v in
        v73 | v75 | v79 | v81) ;;
        *) die "the version $v is not one of v73, v75, v79, v81" ;;
    esac
done
dsps=$(IFS=';'; echo "${versions[*]}")

cd "$REPO_ROOT"
[[ -d third_party/llama.cpp/ggml/src/ggml-hexagon/htp ]] || die "third_party/llama.cpp has no Hexagon backend: run 'git submodule update --init'"

# Only the directories of this script. build/native belongs to the app build.
rm -rf "$OUT_REL/cmake" "$OUT_REL/bin" "$OUT_REL"/v73 "$OUT_REL"/v75 "$OUT_REL"/v79 "$OUT_REL"/v81 "$OUT_REL/hashes.txt"
mkdir -p "$OUT_REL"

if [[ -f tools/htp-lab/isa/isa_kernels.c ]]; then
    echo "isaprobe: census kernels tools/htp-lab/isa/isa_kernels.c"
else
    echo "isaprobe: census kernels $PROBE_REL/isa_kernels_stub.c (tools/htp-lab/isa/isa_kernels.c does not exist)"
fi

container_run \
    -e DSPS="$dsps" \
    -e OUT_REL="$OUT_REL" \
    -e PROBE_REL="$PROBE_REL" \
    -e CFLAGS="-ffile-prefix-map=/workspace=." \
    "$SNAPDRAGON_IMAGE" bash -euo pipefail -c '
out=/workspace/$OUT_REL
cmake -S "$PROBE_REL" -B "$OUT_REL/cmake" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-29 \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
    -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" \
    -DHEXAGON_TOOLS_ROOT="$HEXAGON_TOOLS_ROOT" \
    -DPREBUILT_LIB_DIR=android_aarch64 \
    -DISAPROBE_DSPS="$DSPS" \
    -DISAPROBE_OUT="$out" \
    -DLLAMA_DIR=/workspace/third_party/llama.cpp > "$OUT_REL/cmake.log"
cmake --build "$OUT_REL/cmake" -j"$(nproc)"
cmake --install "$OUT_REL/cmake" > /dev/null

tools=$HEXAGON_TOOLS_ROOT/Tools/bin
IFS=";" read -r -a list <<< "$DSPS"
status=0
for v in "${list[@]}"; do
    lib=$out/$v/libisaprobe_skel.so
    [[ -f $lib ]] || { echo "isaprobe: error: $lib was not built" >&2; exit 1; }
    "$tools/hexagon-llvm-objdump" -d --no-show-raw-insn --mcpu=hexagon$v \
        --mattr=+hvx$v,+hvx-length128b,+hvx-qfloat,+hvx-ieee-fp "$lib" > "$out/$v/skel.disasm.txt"
    "$tools/hexagon-nm" -D --undefined-only "$lib" | while read -r _ name; do
        if [[ -n ${name:-} ]]; then echo "$name"; fi
    done | sort -u > "$out/$v/undefined.txt"
    missing=$(grep -E "^(lab_|isa_|__qf_)" "$out/$v/undefined.txt" || true)
    if [[ -n $missing ]]; then
        echo "isaprobe: error: the $v library needs symbols that it does not define: $(echo $missing)." >&2
        echo "isaprobe: error: add lab_* functions to $PROBE_REL/dsp/isaprobe_dsp.c, remove the calls from the" >&2
        echo "isaprobe: error: census kernels, or link the helper archive (dsp/CMakeLists.txt, ISAPROBE_LIBGCC)." >&2
        status=1
    fi
    # The census functions of the library must be those of the lab compile (lab_ref.o).
    if [[ -f $out/$v/lab_ref.o ]]; then
        "$tools/hexagon-llvm-objdump" -d --no-show-raw-insn --mcpu=hexagon$v \
            --mattr=+hvx$v,+hvx-length128b,+hvx-qfloat,+hvx-ieee-fp "$out/$v/lab_ref.o" > "$out/$v/lab_ref.disasm.txt"
        python3 "$PROBE_REL/check_disasm.py" "$out/$v/skel.disasm.txt" "$out/$v/lab_ref.disasm.txt" \
            > "$out/$v/check_disasm.txt" || status=1
        echo "isaprobe $v: $(tail -n 1 "$out/$v/check_disasm.txt")"
    fi
done
exit $status
'

{
    sha256_table "$OUT_REL/bin/isaprobe"
    for v in "${versions[@]}"; do
        printf '%s  %s/libisaprobe_skel.so\n' "$(sha256sum "$OUT_REL/$v/libisaprobe_skel.so" | cut -d' ' -f1)" "$v"
    done
} > "$OUT_REL/hashes.txt"
echo "isaprobe: $OUT_REL/hashes.txt"
cat "$OUT_REL/hashes.txt"
