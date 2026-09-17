#!/usr/bin/env bash
# htp-lab: the kernel lab for the Hexagon HTP kernels of llama.cpp.
#
# The lab compiles one DSP kernel of the llama.cpp Hexagon backend with hexagon-clang, runs it in
# the cycle-accurate simulator (hexagon-sim in timing mode), and reports the cycles per call, the
# stalls by type, the per-function profile, and the per-packet stall table. It is the Nsight
# Compute equivalent for the kernel work on the phone.
#
# Usage:
#   tools/htp-lab/run.sh build                    Build all lab programs
#   tools/htp-lab/run.sh run <target> [-- args]   Build, then run and profile one target
#   tools/htp-lab/run.sh all                      Run every target with its default arguments
#   tools/htp-lab/run.sh shell                    Open a shell in the container
#
# Targets:
#   gdn_conv   The fused GDN conv step (gdn-conv-ops.c). Args: --n_ch 6144 --threads 1 --iters 20
#   q4         The Q4_0 tiled dot kernels with the Q8 activation (hvx-mm-kernels-tiled.h) for
#              1 to 4 activation rows. Args: --k 2048 --rows 4 --ct 8 --iters 10
#   hmx        The HMX tile MAC microbenchmark, F16 against int8. Args: --dot_tiles 64 --iters 8
#   A target name with the suffix "_after" runs the same program against the kernel directory with
#   the proposal patches (tools/htp-lab/proposals/*.patch) applied to a copy in the output directory.
#
# Environment:
#   LLAMA_DIR  The llama.cpp checkout, read only. Default: /home/grigory/Downloads/llama.cpp
#   IMAGE      The container image with the Hexagon SDK. Default: ghcr.io/snapdragon-toolchain/arm64-android:v0.7
#   SIM_ARGS   More options for hexagon-sim, for example "--plimit 60000000"
#   TAG        The name of the output directory suffix of "run" (out/<target>-<tag>). Default: run
#   MODE       "functional" runs "run" without the timing model. Default: timing
#   NO_BUILD   Set to 1 to skip the build step of "run" (for parallel runs of built programs)
#
# Output (tools/htp-lab/out/, not in git):
#   build/                       The CMake build directory (Ninja)
#   htp-proposed/                The copy of the kernel directory with the proposal patches applied
#   <target>/stdout.txt          The program output with the "lab:" result lines
#   <target>/pa.json             The per-packet profile of hexagon-sim (--packet_analyze)
#   <target>/pa.html             The same profile as the HTML page of hexagon-profiler
#   <target>/pmu.txt             The PMU counters of the full run
#   <target>/gmon.t_0            The per-function profile (hexagon-gprof reads it)
#   <target>/report.txt          The summary that lab/report.py prints
#
# Simulator flags that work in this SDK (Hexagon Tools 19.0.07, hexagon-sim 3.38):
#   --mv79 --timing              The v79na_1 core, cycle-accurate mode (8 threads, 6 HVX contexts,
#                                8 MB VTCM at 0xd9000000, 1 MB L2)
#   --profile                    Writes gmon.t_<thread> for hexagon-gprof
#   --packet_analyze pa.json     Writes the per-packet stalls and events (needs --timing)
#   --pmu_statsfile pmu.txt      Writes all PMU counters of the run
#   --uarchtrace file            Writes a per-cycle trace with the stall reason of each thread
#   --plimit N                   Stops the simulation after N cycles
#   The simulator needs libncurses.so.5. The script makes a shim from libncurses.so.6 of the image.
#   The program reads the cycle counter through s31:30 (PCYCLE), which counts without setup in the
#   standalone runtime. The user counter c15:14 counts only after SSR bit 23 and SYSCFG bit 6 are set.
#   The HMX needs SSR bit 26 (XE2) for the thread. Refer to lab/lab.c. Without that bit an HMX
#   instruction raises the exception 0x18.
#
# The limit of this SDK: the simulator runs the HMX instructions in functional mode, but the timing
# model does not retire them. After the store of the accumulator the first read of the result never
# completes: the thread stalls in DUNCACHED_DEMAND_MISS_CYCLES until the cycle limit. The v79na_1
# stats have no HMX counter. Thus the hmx target runs in functional mode for the correctness result,
# and its timing run needs --plimit and gives no cycle number.
#
# The comparison of each kernel against its scalar reference runs inside the program: the
# "lab: check" lines report the number of elements outside the tolerance.
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${LAB_DIR}/../.." && pwd)"
LLAMA_DIR="${LLAMA_DIR:-/home/grigory/Downloads/llama.cpp}"
IMAGE="${IMAGE:-ghcr.io/snapdragon-toolchain/arm64-android:v0.7}"
SIM_ARGS="${SIM_ARGS:-}"
OUT_REL="tools/htp-lab/out"

usage() {
    sed -n '2,60p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

in_container() {
    # Runs the given script text inside the container with the repository and llama.cpp mounted
    podman run --rm --userns=keep-id --security-opt label=disable \
        -v "${REPO_DIR}:/repo" -v "${LLAMA_DIR}:/llama" -w /repo \
        -e "SIM_ARGS=${SIM_ARGS}" \
        "${IMAGE}" bash -c "$1"
}

# The shell prologue of every container command: the tools path and the ncurses shim
read -r -d '' PROLOGUE <<'EOF' || true
set -euo pipefail
export TOOLS=/opt/hexagon/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools
export PATH="$TOOLS/bin:$PATH"
mkdir -p /tmp/shim
ln -sf /usr/lib/x86_64-linux-gnu/libncurses.so.6 /tmp/shim/libncurses.so.5
ln -sf /usr/lib/x86_64-linux-gnu/libtinfo.so.6 /tmp/shim/libtinfo.so.5
export LD_LIBRARY_PATH=/tmp/shim
OUT=/repo/tools/htp-lab/out
EOF

# Builds all programs. The proposal patches are applied to a copy of the kernel directory.
read -r -d '' BUILD <<'EOF' || true
mkdir -p "$OUT"
rm -rf "$OUT/htp-proposed"
cp -r /llama/ggml/src/ggml-hexagon/htp "$OUT/htp-proposed"
PROPOSED=""
for p in /repo/tools/htp-lab/proposals/*.patch; do
    [ -e "$p" ] || continue
    git -C /repo apply -p5 --directory=tools/htp-lab/out/htp-proposed "$p"
    PROPOSED="$OUT/htp-proposed"
done
cmake -G Ninja -S /repo/tools/htp-lab -B "$OUT/build" \
    -DCMAKE_TOOLCHAIN_FILE=/repo/tools/htp-lab/toolchain.cmake \
    -DLLAMA_DIR=/llama -DHTP_PROPOSED_DIR="$PROPOSED" > "$OUT/cmake.log"
ninja -C "$OUT/build"
EOF

# Runs one program under the simulator and writes the profile files.
# run_target <target> <tag> [args...]: the output goes to out/<target>-<tag>.
# MODE=functional runs without the timing model (no profile files), the default is the timing mode.
read -r -d '' RUNFN <<'EOF' || true
run_target() {
    local target="$1"; local tag="$2"; shift 2
    local dir="$OUT/$target-$tag"
    local elf="$OUT/build/lab_$target"
    [ -x "$elf" ] || { echo "no program $elf"; return 1; }
    rm -rf "$dir"; mkdir -p "$dir"; cd "$dir"
    echo "== run $target-$tag: $*"
    if [ "${MODE:-timing}" = "functional" ]; then
        hexagon-sim --mv79 $SIM_ARGS "$elf" -- "$@" 2>&1 | tee stdout.txt
        grep "^lab:" stdout.txt > report.txt || true
        return 0
    fi
    hexagon-sim --mv79 --timing --profile --packet_analyze pa.json --pmu_statsfile pmu.txt $SIM_ARGS \
        "$elf" -- "$@" 2>&1 | tee stdout.txt
    hexagon-profiler --packet_analyze --json=pa.json --elf="$elf" -o pa.html > /dev/null 2>&1 || true
    hexagon-nm -S -n "$elf" > symbols.txt
    hexagon-llvm-objdump -d --no-show-raw-insn "$elf" > disasm.txt
    python3 /repo/tools/htp-lab/lab/report.py --pa pa.json --symbols symbols.txt --disasm disasm.txt \
        --stdout stdout.txt --target "$target-$tag" | tee report.txt
}
EOF

cmd="${1:-help}"
shift || true
case "$cmd" in
    build)
        in_container "${PROLOGUE}
${BUILD}"
        ;;
    run)
        target="${1:?target}"; shift || true
        [ "${1:-}" = "--" ] && shift
        build_step="${BUILD}"
        [ -n "${NO_BUILD:-}" ] && build_step=""
        in_container "${PROLOGUE}
${build_step}
${RUNFN}
export MODE=${MODE:-timing}
run_target ${target} ${TAG:-run} $*"
        ;;
    all)
        in_container "${PROLOGUE}
${BUILD}
${RUNFN}
run_target q4 k2048 --k 2048 --rows 4 --ct 8 --iters 10
run_target q4 k6144 --k 6144 --rows 4 --ct 4 --iters 10
run_target q4_after k2048 --k 2048 --rows 4 --ct 8 --iters 10
run_target q4_after k6144 --k 6144 --rows 4 --ct 4 --iters 10
run_target gdn_conv t1 --n_ch 6144 --threads 1 --iters 20
run_target gdn_conv t6 --n_ch 6144 --threads 6 --iters 20
run_target gdn_conv_after t1 --n_ch 6144 --threads 1 --iters 20
run_target gdn_conv_after t6 --n_ch 6144 --threads 6 --iters 20
MODE=functional run_target hmx functional --dot_tiles 64 --col_tiles 4 --iters 2
SIM_ARGS=\"\$SIM_ARGS --plimit 60000000\" run_target hmx timing --dot_tiles 64 --col_tiles 4 --iters 8"
        ;;
    shell)
        podman run --rm -it --userns=keep-id --security-opt label=disable \
            -v "${REPO_DIR}:/repo" -v "${LLAMA_DIR}:/llama" -w /repo "${IMAGE}" bash
        ;;
    *)
        usage
        ;;
esac
