#!/usr/bin/env bash
# Build and run the host unit tests of the app (android/app/src/test/cpp) in
# one profile and one sanitizer configuration.
#
# Usage:
#   tests/suite/app-host-test.sh <none|asan|ubsan|tsan|msan> --profile <debug|release> [--jobs N]
#
# The tests: cache_test (the caches, their disk tier, the image cache and
# its SHA-256) and spec_policy_test (the draft length policy). The project
# tests/suite/app-host/CMakeLists.txt compiles the same sources as
# android/app/src/test/cpp/CMakeLists.txt, with only the flags of the
# configuration. That file adds -fsanitize=address,undefined to each
# target, which rule R1 does not permit. The script stops if the two files
# do not list the same executables.
#
# Output: build/fuzz/matrix-app-host-<profile>-<config>/results.jsonl.
# Requirements: cmake, ninja, clang, ld.lld, jq, rg, and for msan the MSan
# libc++. No container. Time: less than 1 minute. RAM: less than 1 GB.
#
# Exit status: 0 if the tests pass with no report, 1 if not, 2 if the step
# cannot run.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
source "$SUITE_REPO_ROOT/tests/sanitizers/env.sh"

readonly APP_TEST_DIR="$SUITE_REPO_ROOT/android/app/src/test/cpp"
readonly SUITE_PROJECT="$SUITE_REPO_ROOT/tests/suite/app-host"
CONFIG=""
PROFILE=""
JOBS=8

# Read the command-line options into the global variables.
parse_args() {
    [[ $# -ge 1 ]] || suite_die "Usage: $0 <config> --profile <debug|release> [--jobs N]"
    CONFIG="$1"
    shift
    suite_is_config "$CONFIG" || suite_die "The configuration '$CONFIG' is not known. Use one of: $SUITE_CONFIGS."
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --profile) PROFILE="$2"; shift 2 ;;
            --jobs) JOBS="$2"; shift 2 ;;
            *) suite_die "The option '$1' is not known." ;;
        esac
    done
    suite_is_profile "$PROFILE" || suite_die "--profile must be debug or release, not '$PROFILE'."
}

# Stop if android/ has an executable that the suite project does not build.
check_drift() {
    local theirs ours
    theirs="$(rg -o -r '$1' '^add_executable\((\w+)' "$APP_TEST_DIR/CMakeLists.txt" | sort | tr '\n' ' ')"
    ours="$(rg -o -r '$1' '^add_executable\((\w+)' "$SUITE_PROJECT/CMakeLists.txt" | sort | tr '\n' ' ')"
    [[ "$theirs" == "$ours" ]] \
        || suite_die "android/app/src/test/cpp builds [$theirs], tests/suite/app-host builds [$ours]. Add the new test to tests/suite/app-host/CMakeLists.txt."
}

main() {
    parse_args "$@"
    suite_require cmake ninja clang clang++ jq rg
    export CCACHE_DISABLE=1
    check_drift
    local build="$SUITE_REPO_ROOT/build/fuzz/matrix-app-host-$PROFILE-$CONFIG"
    local results="$build/results.jsonl" fp stamp init_args failed=0 t
    fp="$(suite_fingerprint "$PROFILE" "$CONFIG" "$(cat "$SUITE_PROJECT/CMakeLists.txt")")"
    stamp="$build/.matrix-fingerprint"
    if [[ ! -f "$stamp" || "$(cat "$stamp")" != "$fp" ]]; then
        rm -rf "$build"
        mkdir -p "$build"
        mapfile -t init_args < <(suite_cmake_init_args "$PROFILE" "$CONFIG")
        cmake -G Ninja -S "$SUITE_PROJECT" -B "$build" "${init_args[@]}" \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > "$build/configure.log" 2>&1 \
            || { tail -20 "$build/configure.log" >&2; suite_die "$PROFILE-$CONFIG: the configure step failed."; }
        echo "$fp" > "$stamp"
    fi
    nice -n 10 cmake --build "$build" -j "$JOBS" > "$build/build.log" 2>&1 \
        || { tail -30 "$build/build.log" >&2; suite_die "$PROFILE-$CONFIG: the build failed. Refer to $build/build.log."; }
    rm -f "$results"
    sanitizer_env "$CONFIG"
    for t in cache_test spec_policy_test; do
        suite_run_record "$results" app-host "$t" "$PROFILE" "$CONFIG" 300 "$build/logs/$t.log" \
            "$build/$t" || failed=1
    done
    suite_log "$PROFILE-$CONFIG: $(jq -s '[.[] | select(.status == "pass")] | length' "$results") of 2 pass."
    return $failed
}

main "$@"
