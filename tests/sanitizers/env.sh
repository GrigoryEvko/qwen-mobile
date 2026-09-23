#!/usr/bin/env bash
# The runtime options of the five sanitizer configurations.
#
# Each configuration has the same options in each area and each suite, thus
# a report means the same thing everywhere. Rule R1: one sanitizer for each
# run. The function exports the variables of one sanitizer and clears the
# variables of the other sanitizers.
#
# Usage, from a bash script:
#   source tests/sanitizers/env.sh
#   sanitizer_env <none|asan|ubsan|tsan|msan>
#
# Usage, from a different shell:
#   eval "$(tests/sanitizers/env.sh <none|asan|ubsan|tsan|msan>)"
#
# The options:
#   all      halt_on_error=1: the first report stops the run, and the exit
#            status is not 0.
#            allocator_may_return_null=1: a very large allocation gives NULL,
#            as glibc does. Without it, the sanitizer stops at the
#            allocation, and the code that handles a NULL result is not tested.
#            The fuzzers use -rss_limit_mb and -malloc_limit_mb for the
#            memory limit.
#   asan     detect_leaks=1 (LeakSanitizer), detect_stack_use_after_return=1,
#            check_initialization_order=1, strict_init_order=1,
#            detect_odr_violation=2.
#   ubsan    print_stacktrace=1, report_error_type=1 and the suppressions of
#            tests/sanitizers/ubsan.supp (rule R5).
#   tsan     second_deadlock_stack=1.
#   msan     poison_in_dtor=1.
# The suppression files tests/sanitizers/<config>.supp are used if they exist
# (rule R6). LeakSanitizer reads tests/sanitizers/lsan.supp, because its
# format is different from the format of ASan.

# The directory of this file. It stays correct when a script sources it.
SANITIZER_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Export the runtime options of one configuration.
# Argument: none, asan, ubsan, tsan or msan.
# Return status: 1 if the configuration is not known.
sanitizer_env() {
    local value="${1:-}" dir="$SANITIZER_ENV_DIR" common symbolizer
    unset ASAN_OPTIONS LSAN_OPTIONS UBSAN_OPTIONS TSAN_OPTIONS MSAN_OPTIONS
    common="halt_on_error=1:allocator_may_return_null=1:print_summary=1"
    symbolizer="$(command -v llvm-symbolizer || true)"
    if [[ -n "$symbolizer" ]]; then
        common="$common:external_symbolizer_path=$symbolizer"
    fi
    case "$value" in
        none) ;;
        asan)
            export ASAN_OPTIONS="$common:detect_leaks=1:detect_stack_use_after_return=1:check_initialization_order=1:strict_init_order=1:detect_odr_violation=2"
            [[ -f "$dir/asan.supp" ]] && ASAN_OPTIONS="$ASAN_OPTIONS:suppressions=$dir/asan.supp"
            export LSAN_OPTIONS="print_suppressions=0"
            [[ -f "$dir/lsan.supp" ]] && LSAN_OPTIONS="$LSAN_OPTIONS:suppressions=$dir/lsan.supp"
            ;;
        ubsan)
            export UBSAN_OPTIONS="$common:print_stacktrace=1:report_error_type=1:suppressions=$dir/ubsan.supp"
            ;;
        tsan)
            export TSAN_OPTIONS="$common:second_deadlock_stack=1"
            [[ -f "$dir/tsan.supp" ]] && TSAN_OPTIONS="$TSAN_OPTIONS:suppressions=$dir/tsan.supp"
            ;;
        msan)
            export MSAN_OPTIONS="$common:poison_in_dtor=1"
            ;;
        *)
            echo "sanitizer_env: the configuration '$value' is not known. Use none, asan, ubsan, tsan or msan." >&2
            return 1
            ;;
    esac
    return 0
}

# When the file runs as a program, write the export lines for eval.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    sanitizer_env "${1:-}"
    for var in ASAN_OPTIONS LSAN_OPTIONS UBSAN_OPTIONS TSAN_OPTIONS MSAN_OPTIONS; do
        if [[ -n "${!var:-}" ]]; then
            printf 'export %s=%q\n' "$var" "${!var}"
        else
            printf 'unset %s\n' "$var"
        fi
    done
fi
