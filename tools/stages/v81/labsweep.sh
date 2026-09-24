#!/usr/bin/env bash
# Run the kernel lab targets of the Qwen3.5 4B path on the v79 phone core (v79na_1) and on the probable v81
# phone core (v81na_2), with the flags of the shipped DSP library, and compare the result lines of the two
# cores.
#
#   LLAMA_DIR=build/v81/src tools/stages/v81/labsweep.sh [TARGET...]
#
#   LLAMA_DIR  The patched llama.cpp tree (tools/stages/v81/build.sh makes build/v81/src). Required.
#   LAB_OUT    The lab output directory, relative to the repository. The preset value is build/v81/sweep.
#
# "--mv81" alone selects v81dgb_1, a core with 2 MB of VTCM, 4 HVX contexts and an HMX with no f16 path, thus
# the v81 runs give SIM_CORE=v81na_2 (12 threads, 8 HVX contexts, 8 MB VTCM, the HMX of v79na_1). The runs are
# in the functional mode of the simulator (the timing model does not retire HMX instructions).
#
# For each target the script writes $LAB_OUT/<target>-<arch>.txt (the "lab:" lines without the lines that name
# the tree or the core) and prints "same" when the two files are equal, else the lines that differ. A target
# that prints output hashes (gemvop, q8, q8oracle, exact) thus proves bit identity. The other targets print
# their errors against their float64 or CPU references with many digits.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../.." && pwd)
cd "$REPO_ROOT"
: "${LLAMA_DIR:?set LLAMA_DIR to the patched tree, for example build/v81/src}"
LLAMA_DIR=$(readlink -f "$LLAMA_DIR")
export LLAMA_DIR
export LAB_OUT=${LAB_OUT:-build/v81/sweep}
# MODE=timing runs the timing model: the cycles and the packets of each hot loop (the HMX instructions do not
# retire in that model, thus the HMX targets need their --hmx 0 form there).
export PROFILE=release PROPOSALS=none MODE=${MODE:-functional}

# target|arguments. The arguments are the Qwen3.5 4B shapes where the target has them.
CASES=(
    "gemvop|--threads 4"
    "q8|--k 2560 --rows 4 --ct 4 --iters 1"
    "q8oracle|"
    "mmchunks|--threads 4"
    "f16act|--threads 4"
    "gdnk|--h 4 --hk 2 --d 128 --tokens 96 --k 1 --threads 4 --hmx 1 --check 1"
    "gdntail|--d 128 --iters 1"
    "gdn_conv|--n_ch 8192 --threads 1 --iters 1"
    "gdn_chunk|--n_ch 8192 --tokens 16 --threads 1 --iters 1"
    "gdnconvdma|--n_ch 8192 --tokens 76 --slots 1 --threads 4 --dma 1 --copy 1 --iters 1"
    "fadq|--rows 256 --iters 1"
    "fwht|--rows 64 --iters 1"
    "rope|--iters 1"
    "reduce|--n 2560 --nsm 512 --iters 1 --range 4"
    "act|--nc 9216 --iters 1 --range 8"
    "unary|--n 2048 --rows 8 --iters 1 --range 12"
    "gate|--kind 0 --d 128 --heads 32 --tokens 64 --threads 4"
    "binary|--n 9216 --iters 1"
    "f16rne|--iters 1 --sweep 1"
    "exact|--vectors 1024 --timing 0"
    "hmx|--dot_tiles 64 --col_tiles 4 --iters 1"
    "v81fa|--threads 4"
    "v81move|--threads 4"
)
# The timing cases: HVX kernels only, with the shapes of the 4B
if [[ $MODE == timing ]]; then
    CASES=(
        "q8|--k 2560 --rows 4 --ct 4 --iters 3"
        "act|--nc 9216 --iters 3 --range 8"
        "reduce|--n 2560 --nsm 512 --iters 5 --range 4"
        "gdn_conv|--n_ch 8192 --threads 1 --iters 5"
        "gdn_chunk|--n_ch 2048 --tokens 16 --threads 1 --iters 2"
        "rope|--iters 20"
        "binary|--n 9216 --iters 4"
        "unary|--n 2048 --rows 8 --iters 2 --range 12"
        "f16rne|--iters 20 --sweep 0"
        "gdnk|--h 2 --hk 1 --d 128 --tokens 64 --threads 2 --hmx 0 --check 0"
        "fwht|--rows 64 --iters 3"
    )
fi
if [[ $# -gt 0 ]]; then
    sel=()
    for c in "${CASES[@]}"; do
        for t in "$@"; do
            [[ ${c%%|*} == "$t" ]] && sel+=("$c")
        done
    done
    CASES=("${sel[@]}")
fi
targets=""
for c in "${CASES[@]}"; do
    targets+=" ${c%%|*}"
done
export LAB_TARGETS="$targets"

# One target for each build: each LTO link of the lab starts a thread for each CPU, and parallel links stop
# with "pthread_create failed". The builds of one LAB_OUT share its copy of the kernel directory, thus they
# run one after the other.
for arch in v79 v81; do
    : > "$LAB_OUT-build-$arch.log"
    for t in $targets; do
        ARCH=$arch LAB_TARGETS=$t tools/htp-lab/run.sh build >> "$LAB_OUT-build-$arch.log" 2>&1 || true
    done
done

run_one() {
    # run_one TARGET ARCH ARGS...: one functional run, then the lab lines without the tree and core lines
    local t=$1 arch=$2
    shift 2
    local core=$arch
    [[ $arch == v81 ]] && core=v81na_2
    ARCH=$arch SIM_CORE=$core NO_BUILD=1 TAG=sweep tools/htp-lab/run.sh run "$t" -- "$@" > "$LAB_OUT/$t-$arch.log" 2>&1 || true
    grep "^lab:" "$LAB_OUT/$t-$arch.log" | grep -v "^lab: tree\|^lab: core\|^lab: NOTICE" > "$LAB_OUT/$t-$arch.txt" || true
}

mkdir -p "$LAB_OUT"
# At most JOBS simulator runs at a time (the preset value 16, the test job limit of the build box)
jobs_max=${JOBS:-16}
for c in "${CASES[@]}"; do
    t=${c%%|*}
    args=${c#*|}
    for arch in v79 v81; do
        while [[ $(jobs -rp | wc -l) -ge $jobs_max ]]; do
            wait -n || true
        done
        # shellcheck disable=SC2086
        run_one "$t" "$arch" $args &
    done
done
wait || true

if [[ $MODE == timing ]]; then
    # The cycle lines of the two cores, side by side
    for c in "${CASES[@]}"; do
        t=${c%%|*}
        echo "== $t (v79na_1 | v81na_2)"
        paste -d'|' <(grep -i -E "cycle|cyc/|packets|us_at|per_vec" "$LAB_OUT/$t-v79.txt") \
            <(grep -i -E "cycle|cyc/|packets|us_at|per_vec" "$LAB_OUT/$t-v81.txt") | head -24
    done
    exit 0
fi
for c in "${CASES[@]}"; do
    t=${c%%|*}
    a=$LAB_OUT/$t-v79.txt
    b=$LAB_OUT/$t-v81.txt
    n=$(wc -l < "$a")
    if [[ $n -eq 0 ]]; then
        echo "$t: no result lines (refer to $LAB_OUT/$t-v79.log)"
    elif cmp -s "$a" "$b"; then
        echo "$t: same ($n lines)"
    else
        echo "$t: DIFFER"
        diff "$a" "$b" | head -20 || true
    fi
done
