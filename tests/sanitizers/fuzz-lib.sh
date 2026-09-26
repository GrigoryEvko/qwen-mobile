#!/usr/bin/env bash
# The shared shell functions of the fuzz areas. Source this file. Do not run it.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/../../sanitizers/fuzz-lib.sh"
#
# The file gives one name and one implementation for each thing that every
# area does (tests/fuzz/<area>/run.sh):
#   - the vocabulary: the configurations, the profiles, the build directory
#   - the settings: FUZZ_BUDGET, FUZZ_JOBS, FUZZ_BUILD_JOBS, FUZZ_PHONE_SERIAL
#   - the usage text from the header comment of a script
#   - one result line of results.jsonl (rules R3 and R11)
#   - the runtime options of one sanitizer (tests/sanitizers/env.sh)
#   - the restart loop of a libFuzzer run, the executions of its log, and the
#     crash files that the run wrote
#   - the phone: the sanitizer runtime library, the runtime options, the
#     thermal command and the process check
#   - the private copy of llama.cpp (tests/sanitizers/llama-copy.sh) and the
#     Android ASan runtime (tests/sanitizers/build-asan-android-runtime.sh)
#
# The exit codes. tests/run-suite.sh reads them: 2 is a missing prerequisite,
# each other value that is not 0 is a failure.
#   0  each target of the run passed
#   1  a target has a finding, or a step of the run failed
#   2  the run could not start: a usage error, or a prerequisite is missing
#
# The library adds no sanitizer flag and no libFuzzer option: the area keeps
# its own command line, because rule L9 reads it in the file of the area.

# The directory of this file, and the root of the repository.
FUZZ_SHARED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FUZZ_REPO="$(cd "$FUZZ_SHARED_DIR/../.." && pwd)"

# The vocabulary of the matrix (rules R1 and R11). TSan and MSan have no
# runtime for Android: the TSan runtime of NDK r29 stops each thread of the
# SM8750, and Android has no MSan.
FUZZ_HOST_CONFIGS="none asan ubsan tsan msan"
FUZZ_PHONE_CONFIGS="none asan hwasan ubsan"
FUZZ_PROFILES="debug release"

# The settings. The canonical name is FUZZ_*; the name without the prefix
# stays valid, because the areas used it before.
FUZZ_BUDGET=${FUZZ_BUDGET:-${BUDGET:-600}}
FUZZ_JOBS=${FUZZ_JOBS:-${JOBS:-4}}
FUZZ_BUILD_JOBS=${FUZZ_BUILD_JOBS:-${BUILD_JOBS:-8}}
FUZZ_PHONE_SERIAL=${FUZZ_PHONE_SERIAL:-${ADB_SERIAL:-${PHONE:-192.168.14.130:5555}}}

# The Android ASan runtime of compiler-rt 22.1.8. The runtime of NDK r29 stops
# each new thread of the SM8750: bionic resets the PAC key through prctl, and
# the prctl interceptor of that runtime then fails its own AUTIASP.
FUZZ_ASAN_RT_NAME="libclang_rt.asan-aarch64-android.so"
FUZZ_ASAN_RT_DIR="$FUZZ_REPO/build/fuzz/asan-android-runtime"

# The runtime options of one sanitizer (the same options in each area).
# shellcheck source=./env.sh
source "$FUZZ_SHARED_DIR/env.sh"

# Write a message to stderr and stop with the code 1.
# Arguments: the parts of the message.
fuzz_die() {
    echo "run.sh: $*" >&2
    exit 1
}

# Write a message to stderr and stop with the code that the first argument
# gives. Use the code 2 for a usage error and for a missing prerequisite.
# Arguments: the code, then the parts of the message.
fuzz_die_code() {
    local code=$1
    shift
    echo "run.sh: $*" >&2
    exit "$code"
}

# Print the header comment of a script as its usage text: each comment line
# after the first line, until the first line that is not a comment.
# Argument: the path of the script.
fuzz_usage_text() {
    local line first=1
    while IFS= read -r line; do
        if [[ $first == 1 ]]; then
            first=0
            continue
        fi
        [[ $line == \#* ]] || break
        line=${line#\#}
        echo "${line# }"
    done < "$1"
}

# Return 0 if the argument is a configuration of the host or of the phone.
# Arguments: the configuration, then "host" or "phone".
fuzz_is_config() {
    local list=$FUZZ_HOST_CONFIGS
    [[ ${2:-host} == phone ]] && list=$FUZZ_PHONE_CONFIGS
    [[ " $list " == *" ${1:-} "* ]]
}

# Stop when the argument is not a configuration of the host or of the phone.
# Arguments: the configuration, then "host" or "phone".
fuzz_check_config() {
    local where=${2:-host}
    fuzz_is_config "${1:-}" "$where" \
        || fuzz_die_code 2 "the configuration '${1:-}' is not known for the $where. Host: $FUZZ_HOST_CONFIGS. Phone: $FUZZ_PHONE_CONFIGS."
}

# Return 0 if the argument is a profile.
# Argument: the profile.
fuzz_is_profile() {
    [[ " $FUZZ_PROFILES " == *" ${1:-} "* ]]
}

# Stop when the argument is not a profile.
# Argument: the profile.
fuzz_check_profile() {
    fuzz_is_profile "${1:-}" \
        || fuzz_die_code 2 "the profile '${1:-}' is not known. Use one of: $FUZZ_PROFILES."
}

# Print the build directory of one area, one profile and one configuration.
# A tag keeps the builds of a private source tree apart.
# Arguments: the area, the profile, the configuration, and the tag (or "").
fuzz_build_dir() {
    echo "$FUZZ_REPO/build/fuzz/$1-$2-$3${4:+-$4}"
}

# Print the Android build directory of one area, one profile and one
# configuration.
# Arguments: the area, the profile, the configuration, and the tag (or "").
fuzz_android_build_dir() {
    echo "$FUZZ_REPO/build/fuzz/$1-android-$2-$3${4:+-$4}"
}

# The install directory of the MSan libc++ (tests/sanitizers/build-msan-libcxx.sh), as
# tests/sanitizers/msan.cmake reads it.
FUZZ_MSAN_PREFIX=${FUZZ_MSAN_PREFIX:-$FUZZ_REPO/build/fuzz/msan-libcxx/install}

# Print the flags of one profile that each object of the build gets, as
# tests/sanitizers/common.cmake and tests/sanitizers/profile-<profile>.cmake give them: the frame
# pointer and the debug information of every configuration, then the flags of the profile.
# A build that configures a project with the two -C files does not need this function. A build that
# compiles one file with a direct compiler call does.
# Argument: the profile.
fuzz_profile_flags() {
    fuzz_check_profile "${1:-}"
    echo "-g -fno-omit-frame-pointer $(rg -o -r '$1' 'SANMATRIX_PROFILE_FLAGS "([^"]*)"' "$FUZZ_SHARED_DIR/profile-$1.cmake")"
}

# Print the optimizer flags of one profile (CMAKE_<LANG>_FLAGS_<TYPE> of the profile file).
# Argument: the profile.
fuzz_profile_opt_flags() {
    local type=DEBUG
    fuzz_check_profile "${1:-}"
    [[ $1 == release ]] && type=RELEASE
    rg -o -r '$1' "CMAKE_C_FLAGS_$type \"([^\"]*)\"" "$FUZZ_SHARED_DIR/profile-$1.cmake"
}

# Print the link flags of one profile (SANMATRIX_PROFILE_LINK_FLAGS of the profile file).
# Argument: the profile.
fuzz_profile_link_flags() {
    fuzz_check_profile "${1:-}"
    rg -o -r '$1' 'SANMATRIX_PROFILE_LINK_FLAGS "([^"]*)"' "$FUZZ_SHARED_DIR/profile-$1.cmake"
}

# Print the flags that the debug profile adds to each C++ object: the library asserts. The release
# profile adds none, because the shipped build has none. The msan configuration uses the MSan
# libc++, thus it gets the hardening mode of that library.
# Arguments: the profile, the configuration.
fuzz_profile_cxx_flags() {
    fuzz_check_profile "${1:-}"
    [[ $1 == debug ]] || return 0
    if [[ ${2:-} == msan ]]; then
        echo "-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE"
    else
        echo "-D_GLIBCXX_ASSERTIONS"
    fi
}

# Print the compile flags of one sanitizer. The rule SAN-FLAGS of
# tests/sanitizers/check-rules.sh compares each list with the file
# tests/sanitizers/<config>.cmake, thus the two cannot drift.
# Argument: the configuration.
fuzz_sanitizer_flags() {
    fuzz_check_config "${1:-}" host
    case $1 in
        none)  echo "" ;;
        asan)  echo "-fsanitize=address -fno-optimize-sibling-calls" ;;
        ubsan) echo "-fsanitize=undefined -fno-sanitize-recover=undefined" ;;
        tsan)  echo "-fsanitize=thread" ;;
        msan)  echo "-fsanitize=memory -fsanitize-memory-track-origins=2" ;;
    esac
}

# Print the link flags of one sanitizer.
# Argument: the configuration.
fuzz_sanitizer_link_flags() {
    fuzz_check_config "${1:-}" host
    case $1 in
        none)  echo "" ;;
        asan)  echo "-fsanitize=address" ;;
        ubsan) echo "-fsanitize=undefined" ;;
        tsan)  echo "-fsanitize=thread" ;;
        msan)  echo "-fsanitize=memory -stdlib=libc++ -L$FUZZ_MSAN_PREFIX/lib -Wl,-rpath,$FUZZ_MSAN_PREFIX/lib -lc++ -lc++abi" ;;
    esac
}

# Print the flags that one sanitizer adds to each C++ object. MSan must see each store, thus the C++
# code uses the MSan libc++ and not libstdc++.
# Argument: the configuration.
fuzz_sanitizer_cxx_flags() {
    fuzz_check_config "${1:-}" host
    case $1 in
        msan) echo "-stdlib=libc++ -nostdinc++ -isystem $FUZZ_MSAN_PREFIX/include/c++/v1" ;;
        *)    echo "" ;;
    esac
}

# Export the runtime options of one sanitizer, and clear the options of the
# other sanitizers (tests/sanitizers/env.sh).
# Argument: the configuration.
fuzz_sanitizer_env() {
    fuzz_check_config "${1:-}" host
    sanitizer_env "$1" || fuzz_die_code 2 "the runtime options of '$1' are not known"
}

# Print the runtime options of one sanitizer as NAME=VALUE lines, for a command
# that takes its environment as a list (env, or an adb command).
# Argument: the configuration.
fuzz_sanitizer_assignments() {
    local var
    (
        fuzz_sanitizer_env "$1"
        for var in ASAN_OPTIONS LSAN_OPTIONS UBSAN_OPTIONS TSAN_OPTIONS MSAN_OPTIONS; do
            [[ -n ${!var:-} ]] && echo "$var=${!var}"
        done
        true
    )
}

# Append one result line to a results file (rules R3 and R11).
# Arguments: the results file, the area, the target, the profile, the
# configuration, the mode, the seconds, the executions, the findings, then the
# crash files.
fuzz_result_line() {
    local file=$1 area=$2 target=$3 profile=$4 config=$5 mode=$6 seconds=$7 executions=$8 findings=$9
    shift 9
    mkdir -p "$(dirname "$file")"
    jq -nc --arg area "$area" --arg target "$target" --arg profile "$profile" --arg sanitizer "$config" \
        --arg mode "$mode" --argjson seconds "$seconds" --argjson executions "$executions" \
        --argjson findings "$findings" --args \
        '{area: $area, target: $target, profile: $profile, sanitizer: $sanitizer, mode: $mode,
          seconds: $seconds, executions: $executions, findings: $findings,
          crash_files: $ARGS.positional}' "$@" >> "$file"
}

# Print the executions of a libFuzzer log: the sum of the last progress count
# of each start. The count of a start begins at 1 again, thus a count that is
# smaller than the count before it ends a start.
# Only the progress lines of libFuzzer count. A target can write a line that
# starts with "#" and digits (a rendered template), and the arithmetic of bash
# overflows on such a number.
# Argument: the log file. Complexity: O(the lines of the log).
fuzz_libfuzzer_executions() {
    local log=$1 execs=0 prev=0 n
    [[ -f $log ]] || { echo 0; return 0; }
    while read -r n; do
        if (( n < prev )); then
            execs=$(( execs + prev ))
        fi
        prev=$n
    done < <({ rg -o -e '^#[0-9]{1,15}[[:space:]]+(INITED|NEW|REDUCE|pulse|DONE|RELOAD)' "$log" || true; } \
             | { rg -o -e '^#[0-9]+' || true; } | tr -d '#')
    echo $(( execs + prev ))
}

# Print each crash file that a run wrote: a file of the artifact directory
# that is newer than the marker file. A slow unit (an input that took more
# than 10 s) is not a finding, thus it does not count.
# Arguments: the artifact directory, the marker file.
fuzz_new_artifacts() {
    [[ -d $1 && -f $2 ]] || return 0
    find "$1" -type f -newer "$2" ! -name 'slow-unit-*' | sort
}

# Run a libFuzzer target again and again until the budget ends.
#
# libFuzzer stops at the first crash. Thus the loop starts it again with the
# time that is left, at most $2 times, and it stops at a start that gives the
# code 0 (the budget ended with no crash).
#
# The function calls the callback with the seconds that are left and the number
# of the start. The callback runs the target, and it keeps the outer
# "timeout -s KILL" of rule L9 in the file of the area: a sanitizer report can
# hang in the death callback of libFuzzer, and the kill then gives the code
# 137. A start that ends with 137 writes a hang record to the artifact
# directory, and the hang is a finding.
#
# FUZZ_ROUND_HOOK, if it is set, names a function that the loop calls after a
# start that did not end with the code 0. The loop gives it the number of the
# start, the log file, the code of the start and the seconds of the start. A
# hook that gives a code other than 0 stops the loop (for example when the same
# report ends each start).
#
# Arguments: the budget in seconds, the maximum number of starts, the log file,
# the artifact directory, the name of the callback function.
# The result: FUZZ_STARTS holds the number of starts.
fuzz_rounds() {
    local budget=$1 max_starts=$2 log=$3 artifacts=$4 callback=$5
    local start=$SECONDS left rc t0
    FUZZ_STARTS=0
    while :; do
        left=$(( budget - (SECONDS - start) ))
        (( left > 5 && FUZZ_STARTS < max_starts )) || break
        FUZZ_STARTS=$(( FUZZ_STARTS + 1 ))
        rc=0
        t0=$SECONDS
        "$callback" "$left" "$FUZZ_STARTS" || rc=$?
        echo "run.sh: start $FUZZ_STARTS ended with the code $rc after $(( SECONDS - start )) s" >> "$log"
        if (( rc == 137 )); then
            {
                echo "hang: the outer timeout stopped the start $FUZZ_STARTS after $(( SECONDS - t0 )) s"
                tail -n 40 "$log"
            } > "$artifacts/hang-start-$FUZZ_STARTS.txt"
        fi
        (( rc == 0 )) && break
        if [[ -n ${FUZZ_ROUND_HOOK:-} ]]; then
            "$FUZZ_ROUND_HOOK" "$FUZZ_STARTS" "$log" "$rc" "$(( SECONDS - t0 ))" || break
        fi
    done
}

# Wait until less than $1 background jobs of the calling shell run.
# Argument: the maximum number of jobs.
fuzz_wait_for_slot() {
    while (( $(jobs -rp | wc -l) >= $1 )); do
        sleep 5
    done
}

# The background jobs of one mode: the process ID and the name of each job.
FUZZ_JOB_PIDS=()
FUZZ_JOB_NAMES=()

# Start one job of a mode in the background when a slot is free. The job
# inherits the standard output of the call, thus a redirection of the call
# applies to the job.
# Arguments: the maximum number of jobs, the name of the job, then the command.
fuzz_start_job() {
    local max=$1 name=$2
    shift 2
    fuzz_wait_for_slot "$max"
    "$@" &
    FUZZ_JOB_PIDS+=("$!")
    FUZZ_JOB_NAMES+=("$name")
}

# Wait for each job that fuzz_start_job started, then forget the jobs. A job
# gives the code 0 when it ran its target, also with a finding: the result line
# holds the finding. A job with another code stopped at an error of the script
# (a failed build step, an unset variable) before it wrote its result line.
# Without this check the mode ends with the code 0 and the target has no result.
# Output: one line for each job that stopped with an error.
# Return status: 1 if a job stopped with an error.
fuzz_wait_jobs() {
    local i status=0
    for i in "${!FUZZ_JOB_PIDS[@]}"; do
        if ! wait "${FUZZ_JOB_PIDS[$i]}"; then
            echo "run.sh: the job ${FUZZ_JOB_NAMES[$i]} stopped with an error, thus it has no result line"
            status=1
        fi
    done
    FUZZ_JOB_PIDS=()
    FUZZ_JOB_NAMES=()
    return $status
}

# Print the sanitizer runtime library that a phone build needs next to its
# executables, or nothing. The none and ubsan builds link the UBSan runtime
# statically (-static-libsan, refer to the vptr text in the CMakeLists.txt of
# the area), thus they need no library.
# Argument: the configuration.
fuzz_phone_runtime() {
    case $1 in
        asan)   echo "$FUZZ_ASAN_RT_NAME" ;;
        hwasan) echo libclang_rt.hwasan-aarch64-android.so ;;
        *)      echo "" ;;
    esac
}

# Print the runtime options of one sanitizer for a run on the phone, as one
# NAME=VALUE word. The options are the options of the host
# (tests/sanitizers/env.sh) less the ones that the phone has no use for: the
# phone has no leak check (detect_leaks=0), and abort_on_error=1 gives a
# tombstone of debuggerd with the stack of each thread.
# Arguments: the configuration, and the path of the suppression file on the
# phone (or "" for no file).
fuzz_phone_options() {
    local config=$1 supp=${2:-} common="halt_on_error=1:allocator_may_return_null=1:abort_on_error=1"
    case $config in
        asan)   echo "ASAN_OPTIONS=$common:detect_leaks=0${supp:+:suppressions=$supp}" ;;
        hwasan) echo "HWASAN_OPTIONS=$common" ;;
        ubsan)  echo "UBSAN_OPTIONS=$common:print_stacktrace=1:report_error_type=1${supp:+:suppressions=$supp}" ;;
        *)      echo "" ;;
    esac
}

# Print the adb command that reads the thermal status of the phone. The
# benchmark protocol reads it before and after each run.
fuzz_phone_thermal_cmd() {
    echo "timeout -s KILL 30 adb -s $FUZZ_PHONE_SERIAL shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
}

# Print the adb command that looks for a process of a run that stays, and reads
# the thermal status again. A process that stays holds the model in memory.
# Argument: the pattern of the process name. The brackets of the pattern keep
# the command line of the shell itself out of the match.
fuzz_phone_check_cmd() {
    echo "timeout -s KILL 30 adb -s $FUZZ_PHONE_SERIAL shell 'pgrep -a -f \"$1\"; dumpsys thermalservice | grep \"Thermal Status\"'"
}

# Make or refresh a private copy of the patched llama.cpp tree with
# tests/sanitizers/llama-copy.sh: the llama.cpp commit that HEAD pins with the
# patch series of HEAD. A landing during a build does not change the tree of
# the build, and the rule LLAMA-COPY of check-rules.sh reads the stamp of the
# copy. The lock keeps two runs from one copy at the same time.
# Arguments: the directory of the copy, then the options of llama-copy.sh (for
# example --ggml).
fuzz_llama_copy() {
    local dest=$1
    shift
    mkdir -p "$(dirname "$dest")"
    flock "$dest.lock" "$FUZZ_SHARED_DIR/llama-copy.sh" "$@" "$dest" > /dev/null \
        || fuzz_die "tests/sanitizers/llama-copy.sh could not make the copy $dest"
}

# Print the path of the Android ASan runtime of compiler-rt 22.1.8, and stop
# when the file is missing or when its sha256 is not the sha256 of its build.
# Each Android ASan run puts this runtime first in LD_LIBRARY_PATH.
fuzz_asan_runtime_path() {
    local file="$FUZZ_ASAN_RT_DIR/$FUZZ_ASAN_RT_NAME"
    [[ -f $file && -f $file.sha256 ]] \
        || fuzz_die_code 2 "no $file: run tests/sanitizers/build-asan-android-runtime.sh"
    [[ "$(sha256sum "$file" | cut -d' ' -f1)" == "$(cut -d' ' -f1 "$file.sha256")" ]] \
        || fuzz_die "$file does not have the sha256 of $file.sha256: build it again with tests/sanitizers/build-asan-android-runtime.sh"
    echo "$file"
}
