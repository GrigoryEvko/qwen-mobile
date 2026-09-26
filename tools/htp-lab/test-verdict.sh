#!/usr/bin/env bash
# The test of the verdict of tools/htp-lab/run.sh.
#
# hexagon-sim gives the exit code 0 for each program, thus run.sh reads the verdict of a run from its
# output. This test runs each case of the lab target selftest (lab/target_selftest.c) and makes sure
# that run.sh gives the exit code 0 for the case that passes and a non-zero exit code for each
# failure: a check with a value outside its tolerance, a status other than 0, a stop before the
# checks, an abort, an exception of the simulator, and the cycle limit. Then it runs "all" on the
# target selftest, one time with the preset cycle limit (exit 0) and one time with a cycle limit that
# stops the program (exit 1, and the summary names the target).
#
# Usage: tools/htp-lab/test-verdict.sh
# The output goes to tools/htp-lab/out/verdict-test (LAB_OUT), and the script removes it at its end.
# The exit code is 0 when each case gives the expected exit code.
set -uo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${LAB_DIR}/run.sh"
export LAB_OUT="tools/htp-lab/out/verdict-test"
export LAB_TARGETS="selftest"
export PROPOSALS=none
export MODE=functional
LOG="$(mktemp)"
trap 'rm -f "${LOG}"; rm -rf "${LAB_DIR}/out/verdict-test"' EXIT

failures=0

# expect <name> <expected exit code: 0 or nonzero> <command...>: runs the command and compares its
# exit code with the expectation. Prints one line for each case.
expect() {
    local name="$1" want="$2" rc
    shift 2
    "$@" > "${LOG}" 2>&1
    rc=$?
    local verdict
    verdict=$(grep -E "^lab: (verdict|all)" "${LOG}" | tail -1)
    if { [ "${want}" = 0 ] && [ "${rc}" = 0 ]; } || { [ "${want}" = nonzero ] && [ "${rc}" != 0 ]; }; then
        echo "ok    ${name}: exit ${rc} (${verdict})"
    else
        echo "FAIL  ${name}: exit ${rc}, the test expects ${want} (${verdict})"
        failures=$((failures + 1))
    fi
}

"${RUN}" build > "${LOG}" 2>&1 || { cat "${LOG}"; echo "the build of the target selftest failed"; exit 1; }

export NO_BUILD=1
for c in pass check status stop abort exception; do
    want=nonzero
    [ "${c}" = pass ] && want=0
    expect "run --case ${c}" "${want}" env TAG="${c}" "${RUN}" run selftest -- --case "${c}"
done
expect "run --case hang with --plimit 20000000" nonzero \
    env TAG=hang SIM_ARGS="--plimit 20000000" "${RUN}" run selftest -- --case hang
expect "run of a target with no program" nonzero env TAG=none "${RUN}" run no_such_target

expect "all on selftest" 0 env ALL_TARGETS=selftest "${RUN}" all
expect "all on selftest with --plimit 1000" nonzero env ALL_TARGETS=selftest ALL_PLIMIT=1000 "${RUN}" all
if ! grep -q "^lab: all fail selftest: .*cycle limit" "${LOG}"; then
    echo "FAIL  the summary of all does not name the target selftest and its cycle limit"
    failures=$((failures + 1))
fi

if [ "${failures}" != 0 ]; then
    echo "test-verdict: ${failures} cases failed"
    exit 1
fi
echo "test-verdict: every case gave the expected exit code"
