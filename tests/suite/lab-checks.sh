#!/usr/bin/env bash
# Run the kernel-lab simulator checks of the DSP code in one DSP profile.
#
# Usage:
#   tests/suite/lab-checks.sh --profile <debug|release> [--archs "v73 v75 v79 v81"]
#
# The DSP code (ggml/src/ggml-hexagon/htp/*.c) runs only on Hexagon, thus no
# host sanitizer applies. This step is part of the "none" configuration. For
# each Hexagon version it:
#   1. Builds the lab targets "exact" and "q8oracle" with
#      tools/htp-lab/run.sh and PROFILE=<profile>: release gives the flags of
#      the shipped DSP library, debug the same flags with live asserts.
#   2. Runs the two programs in hexagon-sim (MODE=functional).
#   3. Checks the outputs with tools/htp-lab/exact/check.py and check_q8.py.
# Then it runs tools/htp-lab/probe/test/q8-ref-sim-test.sh: the ggml scalar
# reference q8_0_ref.c on x86-64 and on the Hexagon scalar unit (v73, v79)
# must give the same bits. That test has its own fixed flags (-O2
# -ffp-contract=off -fno-fast-math), thus the two profiles run the same test.
# tools/htp-lab/probe/test/oracle-sim-test.sh does not run: it needs the
# output of a full simulator census (tools/htp-lab/isa/run_census.sh).
# Each check writes one record to build/fuzz/matrix-lab-<profile>/results.jsonl
# (area htp-lab, sanitizer none).
#
# Requirements: podman and the Hexagon SDK image of tools/htp-lab/run.sh
# (ghcr.io/snapdragon-toolchain/arm64-android:v0.7, approximately 10 GB),
# python3 with numpy. RAM: less than 2 GB. Time: 20 to 40 s for each version.
# Without the image the step writes the status missing-prerequisite.
#
# Exit status: 0 if all checks pass, 1 if a check fails, 2 if the step
# cannot run.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

readonly LAB_IMAGE="${IMAGE:-ghcr.io/snapdragon-toolchain/arm64-android:v0.7}"
PROFILE=""
ARCHS="v73 v75 v79 v81"

# Read the command-line options into the global variables.
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --profile) PROFILE="$2"; shift 2 ;;
            --archs) ARCHS="$2"; shift 2 ;;
            -h|--help) head -26 "${BASH_SOURCE[0]}" | tail -25; exit 0 ;;
            *) suite_die "The option '$1' is not known. Use --help for the usage." ;;
        esac
    done
    suite_is_profile "$PROFILE" || suite_die "--profile must be debug or release, not '$PROFILE'."
}

# Print the run directory of one lab target, as tools/htp-lab/run.sh names it.
# Arguments: the output directory, the target, the Hexagon version, the tag.
run_dir() {
    if [[ "$3" == "v79" ]]; then
        echo "$1/$2-$4"
    else
        echo "$1/$2-$3-$4"
    fi
}

main() {
    parse_args "$@"
    suite_require jq rg python3
    local out_rel="build/fuzz/matrix-lab-$PROFILE"
    local out="$SUITE_REPO_ROOT/$out_rel"
    local results="$out/results.jsonl" logs="$out/logs" failed=0 v tag=matrix
    mkdir -p "$logs"
    rm -f "$results"
    if ! command -v podman > /dev/null || ! podman image exists "$LAB_IMAGE"; then
        for v in $ARCHS; do
            suite_record "$results" htp-lab "exact-$v" "$PROFILE" none test 0 0 0 missing-prerequisite \
                "podman and the image $LAB_IMAGE are necessary (podman pull $LAB_IMAGE)" ""
            suite_record "$results" htp-lab "q8oracle-$v" "$PROFILE" none test 0 0 0 missing-prerequisite \
                "podman and the image $LAB_IMAGE are necessary (podman pull $LAB_IMAGE)" ""
        done
        suite_log "$PROFILE: podman or the image $LAB_IMAGE is missing. The lab checks do not run."
        return 2
    fi
    for v in $ARCHS; do
        local exact_dir q8_dir
        exact_dir="$(run_dir "$out" exact "$v" "$tag")"
        q8_dir="$(run_dir "$out" q8oracle "$v" "$tag")"
        rm -rf "$exact_dir" "$q8_dir"
        # The build and the simulation of "exact". The record of the build
        # and run step fails if the program does not build or does not run.
        suite_run_record "$results" htp-lab "exact-$v.sim" "$PROFILE" none 600 "$logs/exact-$v.sim.log" \
            env PROPOSALS=none ARCH="$v" MODE=functional LAB_TARGETS="exact q8oracle" \
            LAB_OUT="$out_rel" PROFILE="$PROFILE" TAG="$tag" \
            "$SUITE_REPO_ROOT/tools/htp-lab/run.sh" run exact || failed=1
        suite_run_record "$results" htp-lab "q8oracle-$v.sim" "$PROFILE" none 600 "$logs/q8oracle-$v.sim.log" \
            env PROPOSALS=none ARCH="$v" MODE=functional LAB_TARGETS="exact q8oracle" NO_BUILD=1 \
            LAB_OUT="$out_rel" PROFILE="$PROFILE" TAG="$tag" \
            "$SUITE_REPO_ROOT/tools/htp-lab/run.sh" run q8oracle || failed=1
        suite_run_record "$results" htp-lab "exact-$v" "$PROFILE" none 300 "$logs/exact-$v.check.log" \
            python3 "$SUITE_REPO_ROOT/tools/htp-lab/exact/check.py" "$exact_dir" || failed=1
        suite_run_record "$results" htp-lab "q8oracle-$v" "$PROFILE" none 300 "$logs/q8oracle-$v.check.log" \
            python3 "$SUITE_REPO_ROOT/tools/htp-lab/exact/check_q8.py" "$q8_dir" || failed=1
    done
    suite_run_record "$results" htp-lab "q8-ref-sim-test" "$PROFILE" none 600 "$logs/q8-ref-sim-test.log" \
        "$SUITE_REPO_ROOT/tools/htp-lab/probe/test/q8-ref-sim-test.sh" || failed=1
    suite_record "$results" htp-lab "oracle-sim-test" "$PROFILE" none test 0 0 0 excluded \
        "needs the output of a full simulator census run (tools/htp-lab/isa/run_census.sh), which the suite does not make" ""
    local pass fail
    pass="$(jq -s '[.[] | select(.status == "pass")] | length' "$results")"
    fail="$(jq -s '[.[] | select(.status != "pass" and .status != "excluded")] | length' "$results")"
    suite_log "$PROFILE: $pass pass, $fail not pass. Refer to $results."
    return $failed
}

main "$@"
