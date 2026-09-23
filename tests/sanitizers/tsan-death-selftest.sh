#!/usr/bin/env bash
# The self-test of the TSan death callback (tests/sanitizers/fuzz_death.h).
#
# Usage:
#   tests/sanitizers/tsan-death-selftest.sh --profile <debug|release>
#
# The script builds tests/sanitizers/repro/tsan_death_selftest.c as C and as
# C++, with libFuzzer, TSan and the flags of the profile, and replays one
# input that starts a deliberate data race. With halt_on_error=1 each run
# must, in 30 s:
#   - write the report "WARNING: ThreadSanitizer: data race",
#   - write $FUZZ_ARTIFACT_DIR/crash-tsan-<pid> with the bytes of the input,
#   - stop with the exit status of TSan (66), not with the kill of the limit.
# A hang (the deadlock of a death callback that enters a TSan interceptor)
# gives the status timeout. tests/run-suite.sh runs this script as the first
# step of each tsan configuration, and a failure stops that configuration as
# an environment failure.
#
# Output: build/fuzz/matrix-tsan-selftest-<profile>/results.jsonl.
# Requirements: clang with the TSan and libFuzzer runtimes, ld.lld (release),
# jq, rg. No container. Time: less than 1 minute. RAM: less than 1 GB.
#
# Exit status: 0 if the two runs pass, 1 if not, 2 if the step cannot run.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../suite" && pwd)/lib.sh"
source "$SUITE_REPO_ROOT/tests/sanitizers/env.sh"

readonly SAN_DIR="$SUITE_REPO_ROOT/tests/sanitizers"
readonly SRC="$SAN_DIR/repro/tsan_death_selftest.c"
PROFILE=""

# Print the value of one quoted set() of a profile file.
profile_value() {
    rg -o -r '$1' "set\\($1 \"([^\"]*)\"" "$SAN_DIR/profile-$PROFILE.cmake" | head -1
}

main() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --profile) PROFILE="$2"; shift 2 ;;
            *) suite_die "The option '$1' is not known. Use --profile debug|release." ;;
        esac
    done
    suite_is_profile "$PROFILE" || suite_die "--profile must be debug or release, not '$PROFILE'."
    suite_require clang clang++ jq rg
    export CCACHE_DISABLE=1

    local out="$SUITE_REPO_ROOT/build/fuzz/matrix-tsan-selftest-$PROFILE"
    local results="$out/results.jsonl" failed=0 type flags opt link lang compiler prog art log rc crash
    rm -rf "$out"
    mkdir -p "$out"
    type="$( [[ "$PROFILE" == release ]] && echo RELEASE || echo DEBUG )"
    flags="$(profile_value SANMATRIX_PROFILE_FLAGS)"
    opt="$(profile_value "CMAKE_C_FLAGS_$type")"
    link="$(profile_value SANMATRIX_PROFILE_LINK_FLAGS)"
    printf 'R-selftest-input-0123456789' > "$out/input.bin"

    for lang in c c++; do
        compiler=clang
        [[ "$lang" == c++ ]] && compiler=clang++
        prog="$out/selftest-$lang"
        # shellcheck disable=SC2086
        if ! "$compiler" -x "$lang" $flags $opt -g -fno-omit-frame-pointer -fsanitize=fuzzer,thread \
                -I"$SAN_DIR" "$SRC" -o "$prog" $link -lpthread > "$out/build-$lang.log" 2>&1; then
            suite_record "$results" tsan-selftest "death-callback-$lang" "$PROFILE" tsan test 0 0 0 fail \
                "the self-test does not build (refer to $out/build-$lang.log)" "$out/build-$lang.log"
            failed=1
            continue
        fi
        art="$out/art-$lang"
        log="$out/run-$lang.log"
        mkdir -p "$art"
        sanitizer_env tsan
        rc=0
        (cd "$out" && FUZZ_ARTIFACT_DIR="$art" timeout -s KILL 30 "$prog" -runs=1 \
            -artifact_prefix="$art/" "$out/input.bin") > "$log" 2>&1 || rc=$?
        crash="$(find "$art" -name 'crash-tsan-*' -type f 2> /dev/null | head -1)"
        if [[ $rc -eq 137 ]]; then
            suite_record "$results" tsan-selftest "death-callback-$lang" "$PROFILE" tsan test 30 1 0 timeout \
                "the run did not stop in 30 s: the death callback waits for a TSan lock" "$log"
            failed=1
        elif ! rg -q 'WARNING: ThreadSanitizer: data race' "$log"; then
            suite_record "$results" tsan-selftest "death-callback-$lang" "$PROFILE" tsan test 0 1 0 fail \
                "exit $rc, but no TSan report of the deliberate race" "$log"
            failed=1
        elif [[ -z "$crash" ]] || ! cmp -s "$crash" "$out/input.bin"; then
            suite_record "$results" tsan-selftest "death-callback-$lang" "$PROFILE" tsan test 0 1 1 fail \
                "exit $rc and a report, but no crash-tsan-<pid> file with the input bytes in $art" "$log"
            failed=1
        elif [[ $rc -ne 66 ]]; then
            suite_record "$results" tsan-selftest "death-callback-$lang" "$PROFILE" tsan test 0 1 1 fail \
                "exit $rc, not the TSan exit status 66" "$log"
            failed=1
        else
            suite_record "$results" tsan-selftest "death-callback-$lang" "$PROFILE" tsan test 0 1 0 pass \
                "the report, the crash file $(basename "$crash") with the input, exit 66" "$log"
        fi
    done
    jq -r '"[tsan-selftest \(.profile)] \(.status) \(.target): \(.reason)"' "$results"
    return $failed
}

main "$@"
