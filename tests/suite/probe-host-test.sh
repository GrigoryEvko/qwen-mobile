#!/usr/bin/env bash
# Run the host test of the ISA probe (tools/htp-lab/probe/test/host-test.sh)
# with one sanitizer configuration and one profile.
#
# Usage:
#   tests/suite/probe-host-test.sh <none|asan|ubsan|tsan|msan> --profile <debug|release>
#
# host-test.sh compiles host/isaprobe.c with the mock of FastRPC in the
# Snapdragon container and runs 60 checks. Its flags are fixed in the file:
#   flags="$flags -fsanitize=address,undefined -fno-sanitize-recover=all"
# That is two sanitizers in one build (rule R1), and the file has no switch
# for the flags. This script does not edit tools/. It writes a copy of
# host-test.sh to build/fuzz/matrix-probe-<profile>-<config>/host-test.sh,
# with three exact text replacements, and runs the copy:
#   1. The sanitizer line gets the flags of the configuration.
#   2. release: "-O1 -g" becomes "-O2 -g", the optimization of the phone
#      build of isaprobe (tools/htp-lab/probe/CMakeLists.txt). debug keeps -O1.
#   3. container_run gets the sanitizer runtime options (-e), with
#      exitcode=86, thus a report never gives an exit code that a check
#      expects (0, 1, 2, 3, 4 or 99).
# If one of the original lines is not found, the script stops with status 2
# and tells what changed. A switch for the flags in host-test.sh itself makes
# the copy unnecessary.
#
# Each PASS or FAIL line of host-test.sh becomes one record in
# build/fuzz/matrix-probe-<profile>-<config>/results.jsonl (area probe-host).
#
# Requirements: podman and the Snapdragon image of scripts/lib.sh
# (clang 21 with the five sanitizer runtimes), python3. Time: 1 to 2 minutes.
# RAM: less than 2 GB.
#
# Exit status: 0 if all checks pass, 1 if not, 2 if the step cannot run.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
source "$SUITE_REPO_ROOT/tests/sanitizers/env.sh"

readonly ORIGINAL="$SUITE_REPO_ROOT/tools/htp-lab/probe/test/host-test.sh"
readonly LINE_SAN='flags="$flags -fsanitize=address,undefined -fno-sanitize-recover=all"'
readonly LINE_OPT='flags="-std=c11 -D_GNU_SOURCE -O1 -g -Wall -Wextra -Werror -ffp-contract=off -fno-fast-math"'
readonly LINE_SRC='source "$(dirname "${BASH_SOURCE[0]}")/../../../../scripts/lib.sh"'
readonly LINE_RUN='container_run "$SNAPDRAGON_IMAGE" bash'
CONFIG=""
PROFILE=""

# Read the command-line options into the global variables.
parse_args() {
    [[ $# -ge 1 ]] || suite_die "Usage: $0 <config> --profile <debug|release>"
    CONFIG="$1"
    shift
    suite_is_config "$CONFIG" || suite_die "The configuration '$CONFIG' is not known. Use one of: $SUITE_CONFIGS."
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --profile) PROFILE="$2"; shift 2 ;;
            *) suite_die "The option '$1' is not known." ;;
        esac
    done
    suite_is_profile "$PROFILE" || suite_die "--profile must be debug or release, not '$PROFILE'."
}

# Print the clang flags of one configuration for the C code of the probe.
san_flags() {
    case "$1" in
        none) echo "" ;;
        asan) echo "-fsanitize=address -fno-omit-frame-pointer" ;;
        ubsan) echo "-fsanitize=undefined -fno-sanitize-recover=undefined" ;;
        tsan) echo "-fsanitize=thread" ;;
        msan) echo "-fsanitize=memory -fsanitize-memory-track-origins=2" ;;
    esac
}

# Print the "-e NAME=VALUE" options of the sanitizer runtime for the
# container. The paths of this repository become /workspace, and the
# symbolizer of the host is removed, because the container has its own.
container_env_args() {
    local var value
    sanitizer_env "$CONFIG"
    for var in ASAN_OPTIONS LSAN_OPTIONS UBSAN_OPTIONS TSAN_OPTIONS MSAN_OPTIONS; do
        value="${!var:-}"
        [[ -z "$value" ]] && continue
        value="${value//$SUITE_REPO_ROOT//workspace}"
        value="$(printf '%s' "$value" | tr ':' '\n' | rg -v '^external_symbolizer_path=' | tr '\n' ':')"
        value="${value%:}:exitcode=86"
        printf -- '-e %s=%s ' "$var" "$value"
    done
}

# Replace one line that must occur exactly one time.
# Arguments: the name of the variable that holds the text, the old line, the new line.
replace_once() {
    local -n text_ref="$1"
    local old="$2" new="$3" count
    count="$(printf '%s\n' "$text_ref" | rg -c -F -x -- "$old" || true)"
    [[ "$count" == "1" ]] \
        || suite_die "$ORIGINAL has the line [$old] ${count:-0} times, not 1 time. host-test.sh changed: update tests/suite/probe-host-test.sh."
    text_ref="${text_ref/"$old"/"$new"}"
}

main() {
    parse_args "$@"
    suite_require jq rg podman
    local out="$SUITE_REPO_ROOT/build/fuzz/matrix-probe-$PROFILE-$CONFIG"
    local results="$out/results.jsonl" copy="$out/host-test.sh" log="$out/host-test.log"
    local text env_args opt_line rc=0 t0 secs findings
    mkdir -p "$out"
    rm -f "$results"
    source "$SUITE_REPO_ROOT/scripts/lib.sh"
    if ! podman image exists "$SNAPDRAGON_IMAGE"; then
        suite_record "$results" probe-host host-test "$PROFILE" "$CONFIG" test 0 0 0 missing-prerequisite \
            "podman and the image $SNAPDRAGON_IMAGE are necessary" ""
        return 2
    fi

    text="$(cat "$ORIGINAL")"
    replace_once text "$LINE_SAN" "flags=\"\$flags $(san_flags "$CONFIG")\""
    opt_line="$LINE_OPT"
    [[ "$PROFILE" == "release" ]] && opt_line="${LINE_OPT/-O1 -g/-O2 -g}"
    replace_once text "$LINE_OPT" "$opt_line"
    replace_once text "$LINE_SRC" "source \"$SUITE_REPO_ROOT/scripts/lib.sh\""
    env_args="$(container_env_args)"
    # The -e options must stay separate words, thus they are not quoted in
    # the copy. No value has a space.
    text="${text/"$LINE_RUN"/container_run $env_args \"\$SNAPDRAGON_IMAGE\" bash}"
    printf '%s\n' "$text" > "$copy"
    chmod +x "$copy"

    t0="$(date +%s.%N)"
    timeout -s KILL 900 "$copy" > "$log" 2>&1 || rc=$?
    secs="$(suite_elapsed "$t0")"
    findings="$(suite_count_findings "$log")"

    # One record for each check line of host-test.sh.
    local line name status
    while IFS= read -r line; do
        if [[ "$line" == PASS\ * ]]; then
            name="${line#PASS }"; name="${name% (exit *}"; status=pass
        else
            name="${line#FAIL }"; name="${name%%:*}"; status=fail
        fi
        suite_record "$results" probe-host "$name" "$PROFILE" "$CONFIG" test 0 1 0 "$status" \
            "$([[ $status == fail ]] && echo "$line")" "$log"
    done < <(rg '^(PASS|FAIL) ' "$log" || true)
    # The record of the whole run carries the time, the exit status and the
    # sanitizer reports of all checks.
    local status_all=pass reason=""
    if [[ $rc -eq 137 ]]; then status_all=timeout; reason="killed after 900 s"
    elif [[ $rc -ne 0 ]]; then status_all=fail; reason="exit status $rc, $(rg -c '^FAIL ' "$log" || echo 0) checks failed"
    elif [[ "$findings" -gt 0 ]]; then status_all=fail; reason="$findings sanitizer report(s)"
    fi
    suite_record "$results" probe-host host-test "$PROFILE" "$CONFIG" test "$secs" 1 "$findings" \
        "$status_all" "$reason" "$log"
    suite_log "$PROFILE-$CONFIG: $(rg -c '^PASS ' "$log" || echo 0) pass, $(rg -c '^FAIL ' "$log" || echo 0) fail, exit $rc, $secs s."
    [[ "$status_all" == "pass" ]]
}

main "$@"
