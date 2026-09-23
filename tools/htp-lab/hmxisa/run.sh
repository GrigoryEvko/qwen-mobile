#!/usr/bin/env bash
# Runs the HMX census (lab target hmxisa) on the Hexagon cores and compares the results.
#
# Usage:
#   tools/htp-lab/hmxisa/run.sh            Build v73, v75, v79 and v81, run the default cores, compare
#   tools/htp-lab/hmxisa/run.sh --extra    Also run the other cores (tools/htp-lab/out-hmxisa/hmxisa-<arch>-<core>)
#   tools/htp-lab/hmxisa/run.sh --compare  Only compare the existing runs
#
# The default cores are the phone cores of each version: v73na_1, v75na_1, v79na_1 and v81na_2.
# The core goes in SIM_ARGS, because "--mv81" alone selects v81dgb_1, a core with an HMX that has no
# f16 path (each f16 case gives zeros there). A second "-m" option of hexagon-sim replaces the first.
#
# The builds run one after the other: they share the patched copy of the kernel directory in LAB_OUT.
# The runs are functional (the simulator does not time HMX instructions) and take 2 minutes each.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LAB="${REPO_DIR}/tools/htp-lab/run.sh"
export LAB_OUT=tools/htp-lab/out-hmxisa
export LAB_TARGETS=hmxisa
export PROPOSALS=none

# arch:core:tag. The tag "c" is the default census, compare.py reads these four.
DEFAULT_CORES="v73:v73na_1:c v75:v75na_1:c v79:v79na_1:c v81:v81na_2:c"
EXTRA_CORES="v73:v73nc_1:nc1 v73:v73q:q v75:v75qa_1:qa1 v79:v79qa_1:qa1 v81:v81na_3:na3 v81:v81nd_1:nd1 v81:v81dgb_1:dgb1"

build_all() {
    local a
    for a in v73 v75 v79 v81; do
        ARCH=$a "${LAB}" build > "${REPO_DIR}/${LAB_OUT}/build-log-${a}.txt" 2>&1 || {
            echo "error: the build for ${a} failed, refer to ${LAB_OUT}/build-log-${a}.txt" >&2
            exit 1
        }
    done
}

# run_cores "<arch:core:tag> ...": runs each core in parallel and waits for all of them
run_cores() {
    local spec a core tag pids=()
    for spec in $1; do
        IFS=: read -r a core tag <<< "${spec}"
        ARCH=$a MODE=functional NO_BUILD=1 TAG=$tag SIM_ARGS="--m${core}" \
            "${LAB}" run hmxisa -- > /dev/null 2>&1 &
        pids+=($!)
    done
    local p fail=0
    for p in "${pids[@]}"; do
        wait "$p" || fail=1
    done
    [ "$fail" = 0 ] || { echo "error: at least one census run failed" >&2; exit 1; }
}

case "${1:-}" in
    --compare)
        ;;
    --extra)
        build_all
        run_cores "${DEFAULT_CORES}"
        run_cores "${EXTRA_CORES}"
        ;;
    "")
        build_all
        run_cores "${DEFAULT_CORES}"
        ;;
    *)
        sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
        exit 1
        ;;
esac
python3 "${REPO_DIR}/tools/htp-lab/hmxisa/compare.py"
