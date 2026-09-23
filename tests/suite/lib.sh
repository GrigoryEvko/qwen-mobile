#!/usr/bin/env bash
# Shared functions of the test suite scripts (tests/run-suite.sh and
# tests/suite/*.sh). Source this file. Do not run it.
#
# The functions write the result records of the suite. Each record is one
# JSON line with the schema of the fuzz area contract (rules R3 and R11):
#   {area, target, profile, sanitizer, mode, seconds, executions, findings,
#    crash_files}
# The suite adds these fields:
#   status   pass, fail, timeout, excluded or missing-prerequisite
#   reason   the cause of a status that is not pass
#   log      the path of the log file of the target

SUITE_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly SUITE_REPO_ROOT
readonly SUITE_CONFIGS="none asan ubsan tsan msan"
readonly SUITE_PROFILES="debug release"
# The directory of the profile and sanitizer files (cmake -C). The
# environment variable SUITE_SANITIZER_FILES gives a private copy, for a test
# of a change of these files before it goes into tests/sanitizers.
SUITE_SANITIZER_FILES="${SUITE_SANITIZER_FILES:-$SUITE_REPO_ROOT/tests/sanitizers}"

# The report lines of the five sanitizers. Each report of ASan, LSan, TSan
# and MSan ends with one SUMMARY line. UBSan writes one "runtime error:" line
# for each report.
readonly SUITE_SUMMARY_REGEX='^SUMMARY: (Address|Leak|Thread|Memory|UndefinedBehavior)Sanitizer'
readonly SUITE_UBSAN_REGEX='runtime error:'
readonly SUITE_DEADLY_REGEX='^(AddressSanitizer|ThreadSanitizer|MemorySanitizer|UndefinedBehaviorSanitizer|LeakSanitizer):DEADLYSIGNAL|^==[0-9]+==ERROR: [A-Za-z]+Sanitizer'

# Write a message to stderr with the name of the calling script.
suite_log() {
    echo "[$(basename "$0")] $*" >&2
}

# Write an error message to stderr and stop with exit status 2.
# The exit status 2 means "the step could not run", not "a test failed".
suite_die() {
    suite_log "ERROR: $*"
    exit 2
}

# Stop with a clear message if a necessary program is not in PATH.
# Arguments: the program names.
suite_require() {
    local prog
    for prog in "$@"; do
        command -v "$prog" > /dev/null \
            || suite_die "The program '$prog' is necessary and is not in PATH. Install it (Ubuntu: apt-get install $prog)."
    done
}

# Return 0 if the argument is one of the five configurations.
suite_is_config() {
    [[ " $SUITE_CONFIGS " == *" ${1:-} "* ]]
}

# Return 0 if the argument is one of the two profiles.
suite_is_profile() {
    [[ " $SUITE_PROFILES " == *" ${1:-} "* ]]
}

# Print the -C options of cmake for one profile and one configuration.
# Arguments: the profile, the configuration.
suite_cmake_init_args() {
    printf '%s\n' "-C" "$SUITE_SANITIZER_FILES/profile-$1.cmake" \
        "-C" "$SUITE_SANITIZER_FILES/$2.cmake"
}

# Print a fingerprint of the sanitizer and profile files and of the compiler.
# A different value means that a build directory must start again empty.
# Arguments: the profile, the configuration, then more text to include.
suite_fingerprint() {
    local profile="$1" config="$2"
    shift 2
    {
        printf '%s\n' "$@"
        cat "$SUITE_SANITIZER_FILES/common.cmake" \
            "$SUITE_SANITIZER_FILES/profile-$profile.cmake" \
            "$SUITE_SANITIZER_FILES/$config.cmake"
        clang --version | head -1
    } | sha256sum | cut -d' ' -f1
}

# Print the number of sanitizer reports in a log file.
# A report counts one time: the SUMMARY line of ASan, LSan, TSan or MSan, or
# the "runtime error:" line of UBSan. A deadly signal without a SUMMARY line
# (for example a nested error) counts as one report.
# Argument: the log file. Complexity: O(size of the file).
suite_count_findings() {
    local file="$1" n=0 u=0 d=0
    [[ -f "$file" ]] || { echo 0; return 0; }
    n="$(rg -c -e "$SUITE_SUMMARY_REGEX" "$file" || true)"
    u="$(rg -c -e "$SUITE_UBSAN_REGEX" "$file" || true)"
    n="${n:-0}"
    u="${u:-0}"
    # UBSan with print_summary=1 writes a SUMMARY line and a "runtime error:"
    # line for the same report. Count the larger number, not the sum.
    if (( u > n )); then n=$u; fi
    if (( n == 0 )); then
        d="$(rg -c -e "$SUITE_DEADLY_REGEX" "$file" || true)"
        (( ${d:-0} > 0 )) && n=1
    fi
    echo "$n"
}

# Append one result record to a JSON-lines file.
# Arguments: file area target profile sanitizer mode seconds executions
#            findings status reason log [crash_file...]
suite_record() {
    local file="$1" area="$2" target="$3" profile="$4" san="$5" mode="$6" secs="$7" \
        execs="$8" findings="$9" status="${10}" reason="${11}" log="${12}"
    shift 12
    mkdir -p "$(dirname "$file")"
    jq -nc \
        --arg area "$area" --arg target "$target" --arg profile "$profile" --arg san "$san" \
        --arg mode "$mode" --argjson seconds "$secs" --argjson executions "$execs" \
        --argjson findings "$findings" --arg status "$status" --arg reason "$reason" \
        --arg log "$log" --args \
        '{area: $area, target: $target, profile: $profile, sanitizer: $san, mode: $mode,
          seconds: $seconds, executions: $executions, findings: $findings,
          crash_files: $ARGS.positional, status: $status, reason: $reason, log: $log}' \
        "$@" >> "$file"
}

# Run one command with a timeout, write its log, and append its record.
# The command runs in its own working directory under build/ (the directory
# of the log, subdirectory work), thus a file that a test writes into its
# working directory (for example dump_state.bin of test-save-load-state)
# never goes into the root of the repository.
# The status is pass only if the exit status is 0 and the log has no
# sanitizer report.
# Arguments: file area target profile sanitizer timeout_s log, then the command.
# Return status: 0 for pass, 1 if not.
suite_run_record() {
    local file="$1" area="$2" target="$3" profile="$4" san="$5" limit="$6" log="$7"
    shift 7
    local rc=0 t0 secs findings status reason="" work
    work="$(dirname "$log")/work"
    mkdir -p "$work"
    t0="$(date +%s.%N)"
    (cd "$work" && timeout -s KILL "$limit" "$@") > "$log" 2>&1 || rc=$?
    secs="$(suite_elapsed "$t0")"
    findings="$(suite_count_findings "$log")"
    if [[ $rc -eq 137 ]]; then
        status="timeout"; reason="killed after $limit s"
    elif [[ $rc -ne 0 ]]; then
        status="fail"; reason="exit status $rc"
    elif [[ "$findings" -gt 0 ]]; then
        status="fail"; reason="exit status 0, but the log has $findings sanitizer report(s)"
    else
        status="pass"
    fi
    suite_record "$file" "$area" "$target" "$profile" "$san" test "$secs" 1 "$findings" \
        "$status" "$reason" "$log"
    [[ "$status" == "pass" ]]
}

# Print the wall time in seconds since an epoch value with milliseconds.
# Argument: the start time from "date +%s.%N".
suite_elapsed() {
    local now
    now="$(date +%s.%N)"
    # jq does the floating-point subtraction, because bash cannot.
    jq -n --argjson a "$1" --argjson b "$now" '(($b - $a) * 1000 | round) / 1000'
}
