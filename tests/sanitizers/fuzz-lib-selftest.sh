#!/usr/bin/env bash
# The self-test of the shared shell library of the fuzz areas (tests/sanitizers/fuzz-lib.sh).
#
# Usage:
#   tests/sanitizers/fuzz-lib-selftest.sh
#
# The cases need no build and no container. tests/run-suite.sh runs them in
# the step "rules", before each other step:
#   no-input      A result line of the test mode with 0 executions is a
#                 finding: the line has findings 1 and names the missing
#                 inputs, and the writer gives the status 1. A target that
#                 runs no input cannot fail.
#   input         A result line of the test mode with executions and no
#                 finding stays a pass, and a fuzz line with 0 executions
#                 stays as the area wrote it.
#   no-input-dirs fuzz_no_input names the target and each directory that the
#                 test mode read, and fuzz_count_inputs counts the files of
#                 the directories that exist.
#   job-error     A background job that stops with an error makes
#                 fuzz_wait_jobs give the status 1, and the jobs that pass do
#                 not.
#   flags         The flag readers read each initial cache file.
#
# Output: one line for each case. Exit status: 0 if each case passes, 1 if
# not.

set -euo pipefail

# shellcheck source=./fuzz-lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/fuzz-lib.sh"

SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
FAILED=0

# Record the result of one case.
# Arguments: the name of the case, then the command that gives 0 when the case passes.
check_case() {
    local name=$1
    shift
    if "$@"; then
        echo "fuzz-lib self-test: pass $name"
    else
        echo "fuzz-lib self-test: FAIL $name"
        FAILED=1
    fi
}

# A test line with 0 executions: the status 1, findings 1, and a crash file entry that names it.
case_no_input() {
    local file="$SCRATCH/no-input.jsonl" rc=0
    fuzz_result_line "$file" area target release none test 1 0 0 2> "$SCRATCH/no-input.err" || rc=$?
    [[ $rc == 1 ]] \
        && jq -e '.findings == 1 and (.crash_files | length) == 1 and (.crash_files[0] | startswith("no input"))' \
            "$file" > /dev/null \
        && rg -q -F "area target" "$SCRATCH/no-input.err"
}

# A test line with executions, and a fuzz line with 0 executions: written as given, the status 0.
case_input() {
    local file="$SCRATCH/input.jsonl"
    fuzz_result_line "$file" area target release none test 1 3 0 \
        && fuzz_result_line "$file" area target release none fuzz 1 0 0 \
        && jq -s -e 'length == 2 and all(.findings == 0 and (.crash_files | length) == 0)' "$file" > /dev/null
}

# The message and the count of the inputs.
case_no_input_dirs() {
    local text
    mkdir -p "$SCRATCH/seeds/t" "$SCRATCH/regress/u"
    : > "$SCRATCH/regress/u/one.bin"
    : > "$SCRATCH/regress/u/two.bin"
    text=$(fuzz_no_input area t "$SCRATCH/seeds/t" "$SCRATCH/regress/t" 2> "$SCRATCH/dirs.err")
    [[ $text == "no input in $SCRATCH/seeds/t $SCRATCH/regress/t" ]] \
        && rg -q -F "area t:" "$SCRATCH/dirs.err" \
        && rg -q -F "$SCRATCH/regress/t" "$SCRATCH/dirs.err" \
        && [[ $(fuzz_count_inputs "$SCRATCH/seeds/t" "$SCRATCH/regress/t") == 0 ]] \
        && [[ $(fuzz_count_inputs "$SCRATCH/seeds/t" "$SCRATCH/regress/u") == 2 ]]
}

# Three jobs, one of which stops at an unset variable. The wait runs in this shell, because a
# subshell cannot wait for the jobs of its parent.
case_job_error() {
    local out="$SCRATCH/jobs.out" rc=0
    job_ok() { echo "ok $1"; }
    job_broken() { local x=$9; echo "never $x"; }
    fuzz_start_job 2 a job_ok 1 > /dev/null 2>&1
    fuzz_start_job 2 b job_broken > /dev/null 2>&1
    fuzz_start_job 2 c job_ok 3 > /dev/null 2>&1
    fuzz_wait_jobs > "$out" || rc=$?
    [[ $rc == 1 ]] && rg -q -F "the job b stopped with an error" "$out" && ! rg -q -e "job (a|c) " "$out"
}

# The flag readers give the flags of the initial cache files.
case_flags() {
    [[ $(fuzz_sanitizer_flags asan) == *-fsanitize=address* ]] \
        && [[ -z $(fuzz_sanitizer_flags none) ]] \
        && [[ $(fuzz_sanitizer_link_flags msan) == *"-L$FUZZ_MSAN_PREFIX/lib"* ]] \
        && [[ $(fuzz_profile_flags release) == *-flto* ]] \
        && [[ $(fuzz_profile_flags debug) != *-flto* ]] \
        && [[ $(fuzz_profile_opt_flags release) == "-O3 -DNDEBUG" ]] \
        && [[ $(fuzz_profile_cxx_flags debug none) == -D_GLIBCXX_ASSERTIONS ]] \
        && [[ $(fuzz_scalar_x86_options) == *-DGGML_AVX2=OFF* ]]
}

check_case no-input case_no_input
check_case input case_input
check_case no-input-dirs case_no_input_dirs
check_case job-error case_job_error
check_case flags case_flags
exit $FAILED
