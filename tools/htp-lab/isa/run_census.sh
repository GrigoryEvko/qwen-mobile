#!/usr/bin/env bash
# Runs the HVX instruction census on v73, v75, v79 and v81 and compares the results with the CPU
# oracle (strict IEEE, round to nearest even at each step).
#
# Steps:
#   1. gen      Probe the compiler and generate isa_kernels.c (only with GEN=1, the files are in git)
#   2. build    Build the lab target "isa" for each version, one version after the other, because
#               the builds of run.sh share the output directory
#   3. variants Build the same program for v79 and v81 with more compiler flags. Each variant is a
#               version of the report with a suffix: "x" is -mllvm -enable-xqf-gen=true (the flag of the
#               SDK build, hexagon_arch.cmake XQF_ARGS), "s" is -mhvx-qfloat=strict-ieee, "i" is
#               -mhvx-qfloat=ieee
#   4. run      Run all versions in parallel in the functional simulator (about 2 minutes)
#   5. bench    Run the cost bench of all versions in the timing simulator
#   6. disasm   Compile isa_kernels.c for each version and write the disassembly
#   7. compare  Run compare.py: census.csv and the table
#
# Usage: tools/htp-lab/isa/run_census.sh [TAG]   (the default tag is "c")
# Environment: GEN=1 runs step 1. ARCHS="v73 v75 v79 v81" selects the versions. VARIANT_ARCHS="v79 v81"
# and VARIANTS="x s i" select the variant builds ("" for none). SKIP_RUN=1 skips steps 2 to 5.
# BENCH_TRIPS (default 32).
set -euo pipefail

ISA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${ISA_DIR}/../../.." && pwd)"
IMAGE="${IMAGE:-ghcr.io/snapdragon-toolchain/arm64-android:v0.7}"
TAG="${1:-c}"
ARCHS="${ARCHS:-v73 v75 v79 v81}"
VARIANT_ARCHS="${VARIANT_ARCHS-v79 v81}"
VARIANTS="${VARIANTS-x s i}"
BENCH_TRIPS="${BENCH_TRIPS:-32}"
OUT_REL="tools/htp-lab/out-isa"
OUT="${REPO_DIR}/${OUT_REL}"
TOOLS=/opt/hexagon/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools
# The flags of the lab build (CMakeLists.txt)
LAB_FLAGS="-mhvx-length=128B -mhmx -O2 -g -G0 -fvectorize -fno-zero-initialized-in-bss -DNDEBUG=1 -DHTP_MAX_NTHREADS=10"
# The compiler flags of each variant suffix
variant_flags() {
    case "$1" in
        x) echo "-mllvm -enable-xqf-gen=true" ;;
        s) echo "-mhvx-qfloat=strict-ieee" ;;
        i) echo "-mhvx-qfloat=ieee" ;;
        *) echo "run_census: unknown variant $1" >&2; exit 1 ;;
    esac
}
# The labels of the variant builds, for example v79x
VARIANT_LABELS=""
for a in ${VARIANT_ARCHS}; do
    for v in ${VARIANTS}; do
        VARIANT_LABELS="${VARIANT_LABELS} ${a}${v}"
    done
done

in_container() {
    podman run --rm --userns=keep-id --security-opt label=disable -v "${REPO_DIR}:/repo" -w /repo "${IMAGE}" bash -c "$1"
}

# The shell prologue of the container commands: the tools and the ncurses shim of hexagon-sim
PROLOGUE="set -euo pipefail
export PATH=${TOOLS}/bin:\$PATH
mkdir -p /tmp/shim
ln -sf /usr/lib/x86_64-linux-gnu/libncurses.so.6 /tmp/shim/libncurses.so.5
ln -sf /usr/lib/x86_64-linux-gnu/libtinfo.so.6 /tmp/shim/libtinfo.so.5
export LD_LIBRARY_PATH=/tmp/shim"

cd "${REPO_DIR}"
mkdir -p "${OUT}"

if [ "${GEN:-0}" = "1" ]; then
    in_container "python3 tools/htp-lab/isa/gen_census.py probe && python3 tools/htp-lab/isa/gen_census.py generate"
fi

if [ "${SKIP_RUN:-0}" != "1" ]; then
    for a in ${ARCHS}; do
        ARCH=$a LAB_TARGETS=isa LAB_OUT=${OUT_REL} PROPOSALS=none tools/htp-lab/run.sh build > "${OUT}/build-${a}.log" 2>&1
    done
    for l in ${VARIANT_LABELS}; do
        a=${l:0:3}
        in_container "${PROLOGUE}
mkdir -p ${OUT_REL}/build-var-$l
hexagon-clang -m$a -mhvx=$a ${LAB_FLAGS} $(variant_flags ${l:3}) -Wno-format -I tools/htp-lab/lab \
    tools/htp-lab/lab/target_isa.c tools/htp-lab/lab/lab.c -o ${OUT_REL}/build-var-$l/lab_isa" \
            > "${OUT}/build-var-${l}.log" 2>&1
    done

    # The functional census and the timing bench of each version, all in parallel
    pids=""
    for a in ${ARCHS}; do
        ARCH=$a MODE=functional LAB_TARGETS=isa LAB_OUT=${OUT_REL} TAG=${TAG} PROPOSALS=none NO_BUILD=1 \
            tools/htp-lab/run.sh run isa > "${OUT}/run-${a}-${TAG}.log" 2>&1 &
        pids="${pids} $!"
    done
    for l in ${VARIANT_LABELS}; do
        a=${l:0:3}
        in_container "${PROLOGUE}
d=${OUT_REL}/isa-$l-${TAG}; rm -rf \$d; mkdir -p \$d; cd \$d
echo 'lab: tree variant $l: $(variant_flags ${l:3})' > tree.txt
{ cat tree.txt; hexagon-sim --m$a /repo/${OUT_REL}/build-var-$l/lab_isa; } > stdout.txt 2>&1
grep '^lab:' stdout.txt > report.txt || true" > "${OUT}/run-$l-${TAG}.log" 2>&1 &
        pids="${pids} $!"
    done
    for p in ${pids}; do
        wait "$p" || echo "run_census: a run ended with an error, refer to ${OUT}/run-*.log"
    done
    pids=""
    for a in ${ARCHS}; do
        ARCH=$a MODE=timing LAB_TARGETS=isa LAB_OUT=${OUT_REL} TAG=bench PROPOSALS=none NO_BUILD=1 \
            tools/htp-lab/run.sh run isa -- --bench ${BENCH_TRIPS} > "${OUT}/run-${a}-bench.log" 2>&1 &
        pids="${pids} $!"
    done
    for l in ${VARIANT_LABELS}; do
        a=${l:0:3}
        in_container "${PROLOGUE}
d=${OUT_REL}/isa-$l-bench; rm -rf \$d; mkdir -p \$d; cd \$d
hexagon-sim --m$a --timing /repo/${OUT_REL}/build-var-$l/lab_isa -- --bench ${BENCH_TRIPS} > stdout.txt 2>&1" \
            > "${OUT}/run-$l-bench.log" 2>&1 &
        pids="${pids} $!"
    done
    for p in ${pids}; do
        wait "$p" || echo "run_census: a run ended with an error, refer to ${OUT}/run-*.log"
    done
fi

# The disassembly of isa_kernels.c for each version with the lab flags (and with the flags of each
# variant), plus the library functions that the v79 compiler calls for the conversions
in_container "
set -e
dis() {
    a=\$1; label=\$2; shift 2
    ${TOOLS}/bin/hexagon-clang -m\$a -mhvx=\$a ${LAB_FLAGS} \"\$@\" -c tools/htp-lab/isa/isa_kernels.c -o /tmp/isa_kernels-\$label.o
    ${TOOLS}/bin/hexagon-llvm-objdump -d -r --no-show-raw-insn --mcpu=hexagon\$a \
        --mattr=+hvx\$a,+hvx-length128b,+hvx-qfloat,+hvx-ieee-fp /tmp/isa_kernels-\$label.o > ${OUT_REL}/objdump-\$label.txt
    rm -f /tmp/isa_kernels-\$label.o
}
for a in ${ARCHS}; do
    dis \$a \$a
    elf=${OUT_REL}/build-\$a/lab_isa
    [ \$a = v79 ] && elf=${OUT_REL}/build/lab_isa
    syms=\$(${TOOLS}/bin/hexagon-nm \$elf | awk '/ T __qf_convert/ {print \$3}' | paste -sd, -)
    if [ -n \"\$syms\" ]; then
        ${TOOLS}/bin/hexagon-llvm-objdump -d --no-show-raw-insn --mcpu=hexagon\$a \
            --mattr=+hvx\$a,+hvx-length128b,+hvx-qfloat,+hvx-ieee-fp --disassemble-symbols=\$syms \$elf \
            > ${OUT_REL}/objdump-libcalls-\$a.txt
    else
        : > ${OUT_REL}/objdump-libcalls-\$a.txt
    fi
done
$(for l in ${VARIANT_LABELS}; do echo "dis ${l:0:3} $l $(variant_flags ${l:3})"; done)
"

python3 "${ISA_DIR}/compare.py" --tag "${TAG}" --archs ${ARCHS} ${VARIANT_LABELS}
