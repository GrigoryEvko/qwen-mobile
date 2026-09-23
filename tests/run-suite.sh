#!/usr/bin/env bash
# The driver of the test suite and the fuzz suite of the C and C++ code.
#
# Usage:
#   tests/run-suite.sh <test|fuzz|all> <none|asan|ubsan|tsan|msan|all> [options]
#
# Options:
#   --profile debug|release|all   The profiles (rule R11). The preset value is all.
#   --budget-seconds N            The fuzz time of each fuzz target, in each
#                                 configuration. The preset value is 300.
#   --jobs N                      The parallel jobs of each step (compile jobs,
#                                 fuzzers). The preset value is 8.
#   --test-jobs N                 The parallel ctest tests of the llama step.
#                                 The preset value is 4.
#   --areas LIST                  The fuzz areas, comma separated. The preset
#                                 value is core,ops,hexhost,app,quant.
#   --steps LIST                  The steps, comma separated. The preset value
#                                 is each step. Refer to --list-steps.
#   --no-container                Do not run the steps that need the Hexagon
#                                 SDK container. Their records get the status
#                                 missing-prerequisite.
#   --build-msan-libcxx           Build the MSan libc++ if it is missing
#                                 (tests/sanitizers/build-msan-libcxx.sh).
#   --list-steps                  Write the steps with their requirements and stop.
#   -h, --help                    Write this text and stop.
#
# The matrix: {debug, release} x {none, asan, ubsan, tsan, msan} x {test, fuzz}.
# Rule R1: each build and each run has one sanitizer, or none.
#
# The steps, in this sequence (--list-steps gives the requirements):
#   rules        tests/sanitizers/check-rules.sh --no-builds. A violation stops
#                the driver before each other step. After the other steps,
#                check-rules.sh runs again with the build directories
#                (rules-after), thus a stray file in the root of the
#                repository or a build with the wrong flags fails the run.
#   llama        tests/suite/llama-ctest.sh: the llama.cpp ctest suite.
#   app-host     tests/suite/app-host-test.sh: the host unit tests of the app.
#   probe-host   tests/suite/probe-host-test.sh: the ISA probe host test.
#   lab          tests/suite/lab-checks.sh: the kernel-lab simulator checks
#                of the DSP code (the "none" configuration only).
#   supp-repro   tests/sanitizers/supp-repro.sh: rule R13 (ubsan only).
#   tsan-selftest  tests/sanitizers/tsan-death-selftest.sh: the first step of
#                each tsan configuration, in the test and the fuzz suite. A
#                failure is an environment failure: no other tsan step of
#                that profile runs.
#   areas        tests/fuzz/<area>/run.sh <test|fuzz> <config> --profile <p>.
# The test suite runs rules, llama, app-host, probe-host, lab, supp-repro and
# the areas in test mode. The fuzz suite runs the areas in fuzz mode.
#
# Each step has a hard wall-time limit (timeout -s KILL). The sanitizer runtime
# options come from tests/sanitizers/env.sh. The fuzz targets use
# -rss_limit_mb=4096 through their run.sh.
#
# Output:
#   build/fuzz/matrix/summary.json   One record for each suite, profile,
#                                    configuration, step and target: status
#                                    (pass, fail, timeout, excluded,
#                                    missing-prerequisite), seconds,
#                                    executions, findings and crash files.
#                                    Also one record for each step.
#   build/fuzz/matrix/summary.txt    The short text summary.
#   build/fuzz/matrix/logs/          The log of each step.
#
# Exit status: 0 if each step ran and each record passed (excluded records
# carry a written reason), 1 if a step or a record failed, had a finding or
# timed out, 2 for a usage error. A missing prerequisite is a failure.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/suite/lib.sh"

readonly MATRIX="$SUITE_REPO_ROOT/build/fuzz/matrix"
readonly ALL_STEPS="rules,llama,app-host,probe-host,lab,supp-repro,areas"
SUITES=""
CONFIGS=""
PROFILES="debug release"
BUDGET=300
JOBS=8
TEST_JOBS=4
AREAS="core,ops,hexhost,app,quant"
STEPS="$ALL_STEPS"
NO_CONTAINER=0
BUILD_MSAN=0
RUN_ID=""
STEP_RECORDS=""
LAST_STEP_STATUS=""

# Write the header comment of this file as the usage text.
print_usage() {
    local line
    while IFS= read -r line; do
        [[ "$line" == "#!"* ]] && continue
        [[ "$line" != "#"* ]] && break
        line="${line#\#}"
        echo "${line# }"
    done < "${BASH_SOURCE[0]}"
}

# Write the requirements of each step, for the CI workflow.
list_steps() {
    cat <<'EOF'
step        suite  configs            container  RAM (GB)  minutes (28-core host, 8 jobs)
rules       test   (once)             no         <1        <1
llama       test   all five           no         2-6       build 1-12 each, ctest 4-40 each (msan, tsan and release are the slow ones)
app-host    test   all five           no         <1        <1 each
probe-host  test   all five           yes        <2        <1 each (in the Snapdragon image)
lab         test   none only          yes        <2        about 6 for the four Hexagon versions
supp-repro  test   ubsan only         no         2         1 (debug), 3 (release)
tsan-selftest both tsan only         no         <1        <1 (first step of each tsan configuration)
areas       test   all five           per area   per area  per area (refer to tests/fuzz/<area>/run.sh --help)
areas       fuzz   all five           per area   per area  build + budget x ceil(targets / jobs)
The container is the Hexagon SDK image (podman). Without it, use --no-container.
The MSan libc++ (msan steps) takes about 1 minute to build: --build-msan-libcxx.
EOF
}

# Read the command-line options into the global variables.
parse_args() {
    case "${1:-}" in
        -h|--help) print_usage; exit 0 ;;
        --list-steps) list_steps; exit 0 ;;
    esac
    [[ $# -ge 2 ]] || { print_usage >&2; exit 2; }
    case "$1" in
        test|fuzz) SUITES="$1" ;;
        all) SUITES="test fuzz" ;;
        *) suite_log "The suite must be test, fuzz or all, not '$1'."; exit 2 ;;
    esac
    case "$2" in
        all) CONFIGS="$SUITE_CONFIGS" ;;
        *) suite_is_config "$2" || { suite_log "The configuration must be one of: $SUITE_CONFIGS, all."; exit 2; }
           CONFIGS="$2" ;;
    esac
    shift 2
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --profile)
                case "$2" in
                    all) PROFILES="debug release" ;;
                    debug|release) PROFILES="$2" ;;
                    *) suite_log "--profile must be debug, release or all."; exit 2 ;;
                esac
                shift 2 ;;
            --budget-seconds) BUDGET="$2"; shift 2 ;;
            --jobs) JOBS="$2"; shift 2 ;;
            --test-jobs) TEST_JOBS="$2"; shift 2 ;;
            --areas) AREAS="$2"; shift 2 ;;
            --steps) STEPS="$2"; shift 2 ;;
            --no-container) NO_CONTAINER=1; shift ;;
            --build-msan-libcxx) BUILD_MSAN=1; shift ;;
            --list-steps) list_steps; exit 0 ;;
            *) suite_log "The option '$1' is not known. Use --help."; exit 2 ;;
        esac
    done
    [[ "$BUDGET" =~ ^[0-9]+$ && "$JOBS" =~ ^[0-9]+$ && "$TEST_JOBS" =~ ^[0-9]+$ ]] \
        || { suite_log "--budget-seconds, --jobs and --test-jobs need integers."; exit 2; }
}

# Return 0 if the step is selected.
want_step() {
    [[ ",$STEPS," == *",$1,"* ]]
}

# Run one step with a hard wall-time limit, and write its step record.
# Arguments: the suite, the step name, the profile, the configuration, the
# limit in seconds, the results.jsonl file that the step writes (or ""),
# then the command.
# The step record: {kind: "step", suite, step, profile, sanitizer, status,
# exit_code, seconds, log}. A step that writes its records to a file that
# accumulates (the areas) gets only the new lines of this run.
run_step() {
    local suite="$1" step="$2" profile="$3" config="$4" limit="$5" results="$6"
    shift 6
    local log="$MATRIX/logs/$RUN_ID/$suite-$step-$profile-$config.log" rc=0 t0 secs before=0 status
    mkdir -p "$(dirname "$log")"
    # "append:<file>": the step appends to a file that accumulates (the
    # areas), thus only the lines after the present end belong to this run.
    # A plain file: the step writes the file again from the start.
    if [[ "$results" == append:* ]]; then
        results="${results#append:}"
        [[ -f "$results" ]] && before="$(wc -l < "$results")"
    fi
    suite_log "[$suite] $step $profile-$config: start (limit $((limit / 60)) min)."
    t0="$(date +%s.%N)"
    timeout -s KILL "$limit" "$@" > "$log" 2>&1 || rc=$?
    secs="$(suite_elapsed "$t0")"
    case $rc in
        0) status=pass ;;
        137) status=timeout ;;
        2) status=missing-prerequisite ;;
        *) status=fail ;;
    esac
    suite_log "[$suite] $step $profile-$config: $status (exit $rc, $secs s)."
    LAST_STEP_STATUS="$status"
    jq -nc --arg suite "$suite" --arg step "$step" --arg profile "$profile" --arg san "$config" \
        --arg status "$status" --argjson rc "$rc" --argjson seconds "$secs" --arg log "$log" \
        '{kind: "step", suite: $suite, step: $step, profile: $profile, sanitizer: $san,
          status: $status, exit_code: $rc, seconds: $seconds, log: $log}' >> "$STEP_RECORDS"
    # The target records of this step, with the suite and the step added.
    if [[ -n "$results" && -f "$results" ]]; then
        tail -n +"$((before + 1))" "$results" \
            | jq -c --arg suite "$suite" --arg step "$step" \
                '. + {kind: "target", suite: $suite, step: $step,
                      status: (.status // (if (.findings // 0) > 0 then "fail" else "pass" end))}' \
            >> "$MATRIX/records-$RUN_ID.jsonl" || true
    fi
    return 0
}

# Record a step that cannot run, with its reason.
# Arguments: the suite, the step, the profile, the configuration, the reason.
skip_step() {
    jq -nc --arg suite "$1" --arg step "$2" --arg profile "$3" --arg san "$4" --arg reason "$5" \
        '{kind: "step", suite: $suite, step: $step, profile: $profile, sanitizer: $san,
          status: "missing-prerequisite", exit_code: 2, seconds: 0, reason: $reason, log: ""}' \
        >> "$STEP_RECORDS"
    suite_log "[$1] $2 $3-$4: missing prerequisite: $5"
}

# Return 0 if the MSan libc++ is there. Build it first if --build-msan-libcxx.
have_msan_libcxx() {
    local prefix="${FUZZ_MSAN_PREFIX:-$SUITE_REPO_ROOT/build/fuzz/msan-libcxx/install}"
    if [[ -e "$prefix/lib/libc++.so" ]]; then
        return 0
    fi
    if [[ $BUILD_MSAN -eq 1 ]]; then
        suite_log "Build the MSan libc++ (tests/sanitizers/build-msan-libcxx.sh)."
        timeout -s KILL 3600 "$SUITE_REPO_ROOT/tests/sanitizers/build-msan-libcxx.sh" --jobs "$JOBS" \
            > "$MATRIX/logs/$RUN_ID/msan-libcxx.log" 2>&1 && return 0
        suite_log "The build of the MSan libc++ failed. Refer to $MATRIX/logs/$RUN_ID/msan-libcxx.log."
    fi
    return 1
}

# Return 0 if podman and the Hexagon SDK images are there.
have_container() {
    [[ $NO_CONTAINER -eq 0 ]] || return 1
    command -v podman > /dev/null || return 1
    # shellcheck source=/dev/null
    ( source "$SUITE_REPO_ROOT/scripts/lib.sh" && podman image exists "$SNAPDRAGON_IMAGE" ) || return 1
    podman image exists ghcr.io/snapdragon-toolchain/arm64-android:v0.7 || return 1
}

# The steps of the test suite for one profile and one configuration.
test_steps() {
    local p="$1" c="$2" area
    if [[ "$c" == msan ]] && ! have_msan_libcxx; then
        for area in llama app-host probe-host supp-repro; do
            want_step "$area" && skip_step test "$area" "$p" "$c" "the MSan libc++ is missing: run tests/sanitizers/build-msan-libcxx.sh or give --build-msan-libcxx"
        done
        return 0
    fi
    if want_step llama; then
        run_step test llama "$p" "$c" $((150 * 60)) "$SUITE_REPO_ROOT/build/fuzz/matrix-llama-$p-$c/results.jsonl" \
            "$SUITE_REPO_ROOT/tests/suite/llama-ctest.sh" "$c" --profile "$p" --jobs "$JOBS" --test-jobs "$TEST_JOBS"
    fi
    if want_step app-host; then
        run_step test app-host "$p" "$c" $((20 * 60)) "$SUITE_REPO_ROOT/build/fuzz/matrix-app-host-$p-$c/results.jsonl" \
            "$SUITE_REPO_ROOT/tests/suite/app-host-test.sh" "$c" --profile "$p" --jobs "$JOBS"
    fi
    if want_step probe-host; then
        if have_container; then
            run_step test probe-host "$p" "$c" $((20 * 60)) "$SUITE_REPO_ROOT/build/fuzz/matrix-probe-$p-$c/results.jsonl" \
                "$SUITE_REPO_ROOT/tests/suite/probe-host-test.sh" "$c" --profile "$p"
        else
            skip_step test probe-host "$p" "$c" "podman and the Snapdragon image are necessary (or --no-container is set)"
        fi
    fi
    if want_step lab && [[ "$c" == none ]]; then
        if have_container; then
            run_step test lab "$p" "$c" $((60 * 60)) "$SUITE_REPO_ROOT/build/fuzz/matrix-lab-$p/results.jsonl" \
                "$SUITE_REPO_ROOT/tests/suite/lab-checks.sh" --profile "$p"
        else
            skip_step test lab "$p" "$c" "podman and the Hexagon SDK image are necessary (or --no-container is set)"
        fi
    fi
    if want_step supp-repro && [[ "$c" == ubsan ]]; then
        run_step test supp-repro "$p" "$c" $((60 * 60)) "$SUITE_REPO_ROOT/build/fuzz/matrix-supp-repro-$p/results.jsonl" \
            "$SUITE_REPO_ROOT/tests/sanitizers/supp-repro.sh" --profile "$p" --jobs "$JOBS"
    fi
    if want_step areas; then
        for area in ${AREAS//,/ }; do
            area_step test "$area" "$p" "$c"
        done
    fi
}

# Run one area in one mode. The area writes to build/fuzz/<area>-<p>-<c>.
# Arguments: the mode, the area, the profile, the configuration.
area_step() {
    local mode="$1" area="$2" p="$3" c="$4" script="$SUITE_REPO_ROOT/tests/fuzz/$2/run.sh" limit
    if [[ ! -x "$script" ]]; then
        skip_step "$mode" "area-$area" "$p" "$c" "tests/fuzz/$area/run.sh is missing"
        return 0
    fi
    # The limit: 60 minutes for the builds, then the budget for up to 40
    # targets in $JOBS parallel jobs, with a margin of 50 %.
    limit=$(( 60 * 60 ))
    [[ "$mode" == fuzz ]] && limit=$(( limit + BUDGET * 40 * 3 / (2 * (JOBS > 0 ? JOBS : 1)) ))
    run_step "$mode" "area-$area" "$p" "$c" "$limit" "append:$SUITE_REPO_ROOT/build/fuzz/$area-$p-$c/results.jsonl" \
        "$script" "$mode" "$c" --profile "$p" --budget-seconds "$BUDGET" --jobs "$JOBS"
}

# Write summary.json and summary.txt from the records of this run.
write_summary() {
    local records="$MATRIX/records-$RUN_ID.jsonl"
    touch "$records"
    jq -s --slurpfile steps "$STEP_RECORDS" --arg run "$RUN_ID" \
        '{run: $run, steps: $steps, targets: .}' "$records" > "$MATRIX/summary.json"
    {
        echo "Suite run $RUN_ID. Records: $MATRIX/summary.json. Logs: $MATRIX/logs/$RUN_ID/."
        echo
        echo "Steps (status, exit code, seconds):"
        jq -r '.steps[] | "  \(.suite)  \(.step)  \(.profile)-\(.sanitizer)  \(.status)  exit \(.exit_code)  \(.seconds) s\(if .reason then "  (" + .reason + ")" else "" end)"' "$MATRIX/summary.json"
        echo
        echo "Targets (pass / fail / timeout / excluded / missing-prerequisite, findings):"
        jq -r '.targets | group_by([.suite, .step, .profile, .sanitizer])[]
               | "  \(.[0].suite)  \(.[0].step)  \(.[0].profile)-\(.[0].sanitizer):  "
                 + "\([.[] | select(.status == "pass")] | length) / "
                 + "\([.[] | select(.status == "fail" or .status == "no-match")] | length) / "
                 + "\([.[] | select(.status == "timeout")] | length) / "
                 + "\([.[] | select(.status == "excluded")] | length) / "
                 + "\([.[] | select(.status == "missing-prerequisite")] | length), "
                 + "findings \([.[] | .findings // 0] | add)"' "$MATRIX/summary.json"
        echo
        echo "Each target that did not pass:"
        jq -r '.targets[] | select(.status != "pass" and .status != "excluded")
               | "  \(.suite) \(.step) \(.profile)-\(.sanitizer) \(.target): \(.status) \(.reason // "") \(.findings // 0) finding(s)"' \
            "$MATRIX/summary.json"
    } > "$MATRIX/summary.txt"
}

main() {
    parse_args "$@"
    suite_require jq rg yq timeout
    # One working directory for each start of the driver. Each test of the
    # suite scripts runs in its own directory under build/.
    cd "$SUITE_REPO_ROOT"
    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$MATRIX/logs/$RUN_ID"
    STEP_RECORDS="$MATRIX/steps-$RUN_ID.jsonl"
    : > "$STEP_RECORDS"
    : > "$MATRIX/records-$RUN_ID.jsonl"
    suite_log "Run $RUN_ID: suites [$SUITES], profiles [$PROFILES], configurations [$CONFIGS], steps [$STEPS], areas [$AREAS], budget $BUDGET s, jobs $JOBS."

    if want_step rules; then
        local rc=0
        "$SUITE_REPO_ROOT/tests/sanitizers/check-rules.sh" --no-builds --areas "$AREAS" \
            > "$MATRIX/logs/$RUN_ID/rules.log" 2>&1 || rc=$?
        jq -nc --argjson rc "$rc" --arg log "$MATRIX/logs/$RUN_ID/rules.log" \
            '{kind: "step", suite: "test", step: "rules", profile: "-", sanitizer: "-",
              status: (if $rc == 0 then "pass" else "fail" end), exit_code: $rc, seconds: 0, log: $log}' \
            >> "$STEP_RECORDS"
        if [[ $rc -ne 0 ]]; then
            cat "$MATRIX/logs/$RUN_ID/rules.log" >&2
            suite_log "check-rules.sh found violations. The driver stops before the other steps."
            write_summary
            exit 1
        fi
    fi

    local suite p c area
    for suite in $SUITES; do
        for p in $PROFILES; do
            for c in $CONFIGS; do
                # The TSan death callback must work before any tsan step: a
                # deadlock there turns each TSan report into a hung job.
                if [[ "$c" == tsan ]]; then
                    run_step "$suite" tsan-selftest "$p" "$c" $((10 * 60)) \
                        "$SUITE_REPO_ROOT/build/fuzz/matrix-tsan-selftest-$p/results.jsonl" \
                        "$SUITE_REPO_ROOT/tests/sanitizers/tsan-death-selftest.sh" --profile "$p"
                    if [[ "$LAST_STEP_STATUS" != pass ]]; then
                        skip_step "$suite" "tsan-steps" "$p" "$c" \
                            "environment failure: the TSan death callback self-test did not pass, thus no tsan step runs in the $p profile"
                        continue
                    fi
                fi
                if [[ "$suite" == test ]]; then
                    test_steps "$p" "$c"
                elif want_step areas; then
                    for area in ${AREAS//,/ }; do
                        area_step fuzz "$area" "$p" "$c"
                    done
                fi
            done
        done
    done

    # The rules again, after the steps: a step that wrote a file into the root
    # of the repository, or a build with the wrong flags, fails the run.
    if want_step rules; then
        local rc_after=0
        "$SUITE_REPO_ROOT/tests/sanitizers/check-rules.sh" --areas "$AREAS" \
            > "$MATRIX/logs/$RUN_ID/rules-after.log" 2>&1 || rc_after=$?
        jq -nc --argjson rc "$rc_after" --arg log "$MATRIX/logs/$RUN_ID/rules-after.log" \
            '{kind: "step", suite: "test", step: "rules-after", profile: "-", sanitizer: "-",
              status: (if $rc == 0 then "pass" else "fail" end), exit_code: $rc, seconds: 0, log: $log}' \
            >> "$STEP_RECORDS"
        [[ $rc_after -eq 0 ]] || suite_log "check-rules.sh after the steps found violations. Refer to $MATRIX/logs/$RUN_ID/rules-after.log."
    fi

    write_summary
    cat "$MATRIX/summary.txt"
    local bad
    bad="$(jq '[.steps[] | select(.status != "pass")] + [.targets[] | select(.status != "pass" and .status != "excluded")] | length' "$MATRIX/summary.json")"
    [[ "$bad" -eq 0 ]]
}

main "$@"
