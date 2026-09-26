#!/usr/bin/env bash
# The fuzz harnesses of the llama.cpp core and of our patches to it.
#
# Usage:
#   tests/fuzz/core/run.sh test <config> [--profile P] [--budget-seconds N] [--jobs N] [target ...]
#   tests/fuzz/core/run.sh fuzz <config> [--profile P] [--budget-seconds N] [--jobs N] [target ...]
#   tests/fuzz/core/run.sh data
#   tests/fuzz/core/run.sh phone-build <config> [--profile P]
#   tests/fuzz/core/run.sh phone-commands <config> [--profile P]
#
# The configurations (one sanitizer for each build and each run, never two):
# none, asan, ubsan, tsan, msan on the host, and none, asan, hwasan, ubsan on the
# phone (the lists of tests/sanitizers/fuzz-lib.sh). TSan runs on the x86 host
# only: the TSan runtime of NDK r29 stops on the SM8750 (a CHECK failure in
# tsan_rtl.cpp). MSan is not available on Android.
# The profiles: debug and release (the flags of the shipped build). Without
# --profile, a mode runs the two profiles, debug first. The build directory is
# build/fuzz/core-<profile>-<config> (build/fuzz/core-android-<profile>-<config>
# for the phone). A host build takes its flags from the shared files
# tests/sanitizers/profile-<profile>.cmake and tests/sanitizers/<config>.cmake,
# and a run takes its runtime options from tests/sanitizers/env.sh.
#
# Modes:
#   test            Build, then run each target one time over its seeds with the
#                   switches of the known findings on (a seed failure is a new
#                   finding). Then run each regression input
#                   (tests/fuzz/core/regress/<target>--<finding>) two times:
#                   with all switches less the switch of its finding, where it must
#                   show its report or pass when the configuration cannot show
#                   it, and with all switches, where it must pass (rule R13). The
#                   exit code is 1 when a target has a finding. Each run has a
#                   limit of 600 s.
#   fuzz            Build, then mutate each target for the budget, with the
#                   switches of the known findings on (FUZZ_KNOWN=1).
#   data            Write build/fuzz/core/data: the vocab-only GGUF of the 2B
#                   model, its chat template, and the tiny random qwen35
#                   models (f32 and q8_0). The other modes need these files.
#   phone-build     Copy the llama.cpp tree (the submodule, or FUZZ_LLAMA_DIR) into
#                   build/fuzz/core-android-src (with the suffix -<tag>), and
#                   build the device targets for arm64 Android with one profile
#                   and one sanitizer, plus the DSP library of v79 (no sanitizer:
#                   the DSP code has none), in the container of scripts/build-native.sh.
#   phone-commands  Print the adb commands that push one phone build and run the
#                   device targets on HTP0 and on the CPU of the phone. This script
#                   does not touch the phone.
#
# The targets: fuzz_gguf fuzz_model_load fuzz_tokenizer fuzz_chat fuzz_sampler
# fuzz_recurrent fuzz_npu_decode tsan_threadpool tsan_decode tsan_topset.
# The tsan configuration runs only the targets with threads: fuzz_recurrent,
# fuzz_npu_decode and the three tsan_* targets.
#
# Old mode names (aliases): cpu-asan = fuzz asan, cpu-tsan = fuzz tsan,
# regress = test asan.
#
# Options and environment:
#   --profile P         debug or release. The default is the two, in sequence.
#   --budget-seconds N  Seconds for each target in the fuzz mode (FUZZ_BUDGET,
#                       default 600).
#   --jobs N            Targets that run at the same time (FUZZ_JOBS, default 4).
#   FUZZ_BUILD_JOBS     Build jobs (default 8).
#   FUZZ_KNOWN          1 (the default) turns on the switch of each known finding
#                       in the fuzz mode. 0 turns them off.
#   FUZZ_MODEL_SRC      The Qwen3.5 GGUF of the data mode (default
#                       weights/gguf/Qwen3.5-2B-Q8_0.gguf).
#   FUZZ_LLAMA_DIR      A private llama.cpp tree for the test, fuzz and phone modes
#                       (the test of a fix before it lands). The default is the submodule.
#   FUZZ_TREE_TAG       The name of that tree. The build directory gets it as a
#                       suffix: build/fuzz/core-<profile>-<config>-<tag>, and
#                       build/fuzz/core-android-<profile>-<config>-<tag> for the
#                       phone. The phone directory of phone-commands gets it too.
#                       It is necessary with FUZZ_LLAMA_DIR.
#   FUZZ_PHONE_SERIAL   The phone of phone-commands (default 192.168.14.130:5555).
#
# Each fuzz run uses nice 10, -rss_limit_mb=4096 and -timeout=30 (a test run
# -timeout=60). fuzz_npu_decode gets -timeout=120 in a sanitizer build, because
# one input can run for 30 s there. libFuzzer stops
# at the first crash, thus the script starts it again on the same corpus until
# the budget is spent (at most 200 starts). A start that does not stop 120 s
# after its budget gets SIGKILL: a sanitizer report can hang in the death
# callback of libFuzzer (seen with TSan). The logs, the corpus and the crash
# inputs of a target are in build/fuzz/core-<profile>-<config>/runs/<target>. Each
# run writes one JSON line for each target to build/fuzz/core-<profile>-<config>/results.jsonl:
# {area, target, sanitizer, profile, mode, seconds, executions, findings, crash_files}.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../../sanitizers/fuzz-lib.sh
source "$HERE/../../sanitizers/fuzz-lib.sh"

AREA=core
REPO=$FUZZ_REPO
SHARED=$FUZZ_SHARED_DIR
DATA="$REPO/build/fuzz/core/data"
BUDGET=$FUZZ_BUDGET
JOBS=$FUZZ_JOBS
BUILD_JOBS=$FUZZ_BUILD_JOBS
KNOWN=${FUZZ_KNOWN:-1}
LLAMA_DIR=${FUZZ_LLAMA_DIR:-}
TREE_TAG=${FUZZ_TREE_TAG:-}
PHONE_BASE=/data/local/tmp/qwen/fuzz/core
SHIPPED_MARCH="-march=armv8.7a+fp16+dotprod+i8mm"

ALL_TARGETS="fuzz_gguf fuzz_model_load fuzz_tokenizer fuzz_chat fuzz_sampler fuzz_recurrent fuzz_npu_decode tsan_threadpool tsan_decode tsan_topset"
TSAN_TARGETS="fuzz_recurrent fuzz_npu_decode tsan_threadpool tsan_decode tsan_topset"
PHONE_TARGETS="fuzz_npu_decode fuzz_recurrent"

# The switch of each known finding (rule R8 and R13).
declare -A SWITCH=(
)

# The report of each known finding, as an extended regular expression on the run log. A
# regression input "reproduces" its finding when its log matches this expression.
declare -A EXPECT=(
)

# Print the header comment of this file as the usage text, then stop with the status $1 (2 when it
# is not given: a wrong command line).
usage() {
    fuzz_usage_text "${BASH_SOURCE[0]}"
    exit "${1:-2}"
}

# The switches of all known findings, as NAME=1 words. $1, if given, is a switch to leave out.
known_env() {
    local v
    for v in $(printf '%s\n' "${SWITCH[@]}" | sort -u); do
        [[ $v == "${1:-}" ]] || printf '%s=1 ' "$v"
    done
}

# The -max_len of each target: the largest seed and some room.
max_len() {
    case $1 in
        fuzz_gguf|fuzz_model_load) echo 65536 ;;
        fuzz_chat)                 echo 16384 ;;
        fuzz_tokenizer|fuzz_sampler) echo 4096 ;;
        fuzz_recurrent)            echo 1024 ;;
        *)                         echo 256 ;;
    esac
}

# The -timeout of libFuzzer for one input on the host. $1 is the mode (fuzz or test), $2 the target,
# $3 the configuration. One input of fuzz_npu_decode (two contexts that decode the same tokens) runs
# for up to about 30 s in a sanitizer build (26.3 s and 30.4 s in debug-ubsan), thus it gets 120 s
# there, as the ops area has.
unit_timeout() {
    if [[ $2 == fuzz_npu_decode && $3 != none ]]; then
        echo 120
    elif [[ $1 == fuzz ]]; then
        echo 30
    else
        echo 60
    fi
}

# The first error line of a run log on stdin: the signature of a finding.
signature() {
    grep -oE "FUZZ FAILURE: (P[0-9]+: )?[a-zA-Z ,'-]+|[a-z_./-]+:[0-9]+: fatal error|runtime error: [a-z0-9.+ -]*is outside the range|runtime error: [a-z -]+|ERROR: [A-Za-z]+Sanitizer: [a-z-]+|WARNING: ThreadSanitizer: [a-z ]+|WARNING: MemorySanitizer: [a-z-]+|[a-z_./-]+:[0-9]+: GGML_ASSERT\([^)]*\)|GGML_ABORT|what\(\): .*|Assertion .*|libFuzzer: [a-z-]+|cannot write shared[a-z ]+" \
        | head -n 1 || true
}

# The build directory of the profile $1 and the configuration $2 (with the suffix of a private tree).
tree_dir() {
    fuzz_build_dir "$AREA" "$1" "$2" "$TREE_TAG"
}

# The phone build directory of the profile $1 and the configuration $2, relative to the repository.
phone_rel() {
    local dir
    dir=$(fuzz_android_build_dir "$AREA" "$1" "$2" "$TREE_TAG")
    echo "${dir#"$REPO"/}"
}

# Configure and build the host tree of the profile $1 and the configuration $2 with the targets $3.
build_tree() {
    local profile=$1 san=$2 targets=$3
    local dir
    dir=$(tree_dir "$profile" "$san")
    [[ -f "$SHARED/profile-$profile.cmake" && -f "$SHARED/$san.cmake" ]] \
        || fuzz_die "the shared files $SHARED/profile-$profile.cmake and $SHARED/$san.cmake are necessary"
    local -a src=()
    [[ -n $LLAMA_DIR ]] && src=(-DFUZZ_LLAMA_DIR="$LLAMA_DIR")
    mkdir -p "$dir"
    cmake -G Ninja -S "$HERE" -B "$dir" "${src[@]}" \
        -DFUZZ_TARGETS="${targets// /;}" \
        -DCMAKE_AR="$(command -v llvm-ar)" -DCMAKE_RANLIB="$(command -v llvm-ranlib)" \
        -C "$SHARED/profile-$profile.cmake" -C "$SHARED/$san.cmake" > "$dir/configure.log" 2>&1 \
        || fuzz_die "the configure of $dir failed. Read $dir/configure.log."
    # shellcheck disable=SC2086
    nice -n 10 cmake --build "$dir" -j"$BUILD_JOBS" --target $targets > "$dir/build.log" 2>&1 \
        || fuzz_die "the build of $dir failed. Read $dir/build.log."
}

# Fuzz one target for the budget. $1 is the profile, $2 the configuration, $3 the target.
fuzz_one() {
    local profile=$1 san=$2 fz=$3
    local dir
    dir=$(tree_dir "$profile" "$san")
    local out="$dir/runs/$fz"
    mkdir -p "$out/corpus" "$out/artifacts"
    local log="$out/log.txt"
    : > "$log"
    # A file of the artifacts directory that is not newer than this mark comes from an earlier
    # run, thus it is not a finding of this run.
    local mark="$out/.fuzz-start"
    touch "$mark"
    fuzz_sanitizer_env "$san"
    local -a known=()
    [[ "$KNOWN" == 1 ]] && read -r -a known <<< "$(known_env)"
    # One start of libFuzzer with the seconds of $1 that are left. The outer kill of rule L9 gives
    # 120 s more than that time: a sanitizer report can hang in the death callback of libFuzzer.
    core_fuzz_start() {
        env "${known[@]}" GGML_NO_BACKTRACE=1 FUZZ_DATA_DIR="$DATA" FUZZ_ARTIFACT_DIR="$out/artifacts" \
            timeout -s KILL $(( $1 + 120 )) nice -n 10 "$dir/$fz" "$out/corpus" "$HERE/seeds/$fz" \
                -max_total_time="$1" -rss_limit_mb=4096 -timeout="$(unit_timeout fuzz "$fz" "$san")" \
                -max_len="$(max_len "$fz")" -artifact_prefix="$out/artifacts/" -print_final_stats=1 \
                >> "$log" 2>&1
    }
    local start=$SECONDS execs cov
    fuzz_rounds "$BUDGET" 200 "$log" "$out/artifacts" core_fuzz_start
    execs=$(fuzz_libfuzzer_executions "$log")
    cov=$(rg -o -e 'cov: [0-9]+ ft: [0-9]+' "$log" | tail -n 1 || true)
    local -a arts=()
    mapfile -t arts < <(fuzz_new_artifacts "$out/artifacts" "$mark")
    fuzz_result_line "$dir/results.jsonl" "$AREA" "$fz" "$profile" "$san" fuzz "$(( SECONDS - start ))" \
        "$execs" "${#arts[@]}" "${arts[@]}"
    echo "$AREA-$profile-$san${TREE_TAG:+-$TREE_TAG} $fz: $(( SECONDS - start )) s, $FUZZ_STARTS starts, $execs executions, $cov, corpus $(find "$out/corpus" -type f | wc -l), crash files ${#arts[@]}"
}

# Test one target. The seeds run with all the switches on: a failure is a finding with no switch.
# Each regression input with a switch in SWITCH runs two times: (A) with all the switches less its
# own, where it must show its report (EXPECT) or pass when this configuration cannot show the
# defect, and (B) with all the switches, where it must pass (rule R13: the switch holds). A
# regression input with no switch (its defect has a fix in the code) runs one time with all the
# switches, and it must pass.
test_one() {
    local profile=$1 san=$2 fz=$3
    local dir
    dir=$(tree_dir "$profile" "$san")
    local out="$dir/runs/$fz"
    mkdir -p "$out"
    local log="$out/test-log.txt"
    : > "$log"
    fuzz_sanitizer_env "$san"
    local start=$SECONDS findings=0 execs=0 rc f name finding sw expect sig tmp art tmo
    local -a failed=() all=() others=()
    local notes=""
    tmo=$(unit_timeout test "$fz" "$san")
    read -r -a all <<< "$(known_env)"
    tmp=$(mktemp)
    # libFuzzer writes a crash file for each input that fails. These inputs are known, thus the
    # crash files go to a temporary directory and not to the current directory.
    art=$(mktemp -d)
    rc=0
    env "${all[@]}" GGML_NO_BACKTRACE=1 FUZZ_DATA_DIR="$DATA" FUZZ_ARTIFACT_DIR="$art" timeout -s KILL 600 "$dir/$fz" -runs=0 -rss_limit_mb=4096 \
        -timeout="$tmo" -artifact_prefix="$art/" -max_len="$(max_len "$fz")" "$HERE/seeds/$fz" >> "$log" 2>&1 || rc=$?
    execs=$(( execs + $(find "$HERE/seeds/$fz" -type f | wc -l) ))
    if [[ $rc != 0 ]]; then
        findings=$(( findings + 1 ))
        failed+=("$HERE/seeds/$fz")
        notes+=" seeds:fail($(signature < "$log" | tr ' ' _))"
    fi
    for f in "$HERE"/regress/"$fz"--*; do
        [[ -f $f ]] || continue
        name=$(basename "$f")
        finding=${name#*--}
        sw=${SWITCH[$finding]:-}
        expect=${EXPECT[$finding]:-}
        if [[ -z $sw ]]; then
            # no switch: the code has the fix of the defect, and the input must pass with all the switches
            rc=0
            env "${all[@]}" GGML_NO_BACKTRACE=1 FUZZ_DATA_DIR="$DATA" FUZZ_ARTIFACT_DIR="$art" timeout -s KILL 600 "$dir/$fz" \
                -rss_limit_mb=4096 -timeout="$tmo" -artifact_prefix="$art/" "$f" > "$tmp" 2>&1 || rc=$?
            cat "$tmp" >> "$log"
            execs=$(( execs + 1 ))
            sig=$(signature < "$tmp")
            echo "run.sh: $name (no switch) gives the code $rc: $sig" >> "$log"
            if [[ $rc == 0 ]]; then
                notes+=" $finding:passes"
            else
                findings=$(( findings + 1 ))
                failed+=("$f (a reproducer with no switch fails)")
                notes+=" $finding:fails(${sig// /_})"
            fi
            continue
        fi
        [[ -n $expect ]] || fuzz_die "the finding $finding of $name has a switch but no expected report in run.sh"
        # A: all the switches on, less the switch of this finding. The input must show its finding,
        # or pass when this configuration cannot show it.
        read -r -a others <<< "$(known_env "$sw")"
        rc=0
        env "${others[@]}" GGML_NO_BACKTRACE=1 FUZZ_DATA_DIR="$DATA" FUZZ_ARTIFACT_DIR="$art" timeout -s KILL 600 "$dir/$fz" -rss_limit_mb=4096 \
            -timeout="$tmo" -artifact_prefix="$art/" "$f" > "$tmp" 2>&1 || rc=$?
        cat "$tmp" >> "$log"
        execs=$(( execs + 1 ))
        sig=$(signature < "$tmp")
        echo "run.sh: $name without $sw gives the code $rc: $sig" >> "$log"
        if [[ $rc == 0 ]]; then
            notes+=" $finding:not-shown"
        elif grep -qE "$expect" "$tmp"; then
            findings=$(( findings + 1 ))
            failed+=("$f")
            notes+=" $finding:reproduced"
        else
            findings=$(( findings + 1 ))
            failed+=("$f (a different report)")
            notes+=" $finding:other(${sig// /_})"
        fi
        # B: all the switches on. The switch of the finding must keep it off (rule R13).
        rc=0
        env "${all[@]}" GGML_NO_BACKTRACE=1 FUZZ_DATA_DIR="$DATA" FUZZ_ARTIFACT_DIR="$art" timeout -s KILL 600 "$dir/$fz" -rss_limit_mb=4096 \
            -timeout="$tmo" -artifact_prefix="$art/" "$f" > "$tmp" 2>&1 || rc=$?
        cat "$tmp" >> "$log"
        execs=$(( execs + 1 ))
        sig=$(signature < "$tmp")
        echo "run.sh: $name with all switches gives the code $rc: $sig" >> "$log"
        if [[ $rc == 0 ]]; then
            notes+="/switch-holds"
        elif grep -qE "$expect" "$tmp"; then
            findings=$(( findings + 1 ))
            failed+=("$f (the switch $sw does not hold)")
            notes+="/switch-leaks"
        else
            findings=$(( findings + 1 ))
            failed+=("$f (a different report with all switches)")
            notes+="/switch-holds-then(${sig// /_})"
        fi
    done
    rm -rf "$tmp" "$art"
    fuzz_result_line "$dir/results.jsonl" "$AREA" "$fz" "$profile" "$san" test "$(( SECONDS - start ))" \
        "$execs" "$findings" "${failed[@]}"
    echo "core-$profile-$san${TREE_TAG:+-$TREE_TAG} $fz: $execs runs, $findings findings |$notes"
}

# Run the mode $1 (test or fuzz) with the profile $2 and the configuration $3 on the targets that
# follow, JOBS at a time. Returns 1 when the test mode has a finding, and when a job stops with an
# error before its result line.
run_mode() {
    local mode=$1 profile=$2 san=$3
    shift 3
    local targets="$*"
    if [[ -z $targets ]]; then
        targets=$ALL_TARGETS
        [[ $san == tsan ]] && targets=$TSAN_TARGETS
    fi
    [[ -f "$DATA/tiny-qwen35-f32.gguf" && -f "$DATA/qwen35-vocab.gguf" ]] || fuzz_die "no data in $DATA. Run 'tests/fuzz/core/run.sh data' first."
    build_tree "$profile" "$san" "$targets"
    local dir
    dir=$(tree_dir "$profile" "$san")
    local summary="$dir/$mode-summary.txt"
    : > "$summary"
    local fz jobs_ok=1
    for fz in $targets; do
        fuzz_start_job "$JOBS" "$fz" "${mode}_one" "$profile" "$san" "$fz" >> "$summary"
    done
    fuzz_wait_jobs >> "$summary" || jobs_ok=0
    cat "$summary"
    echo "run.sh: the JSON lines are in $dir/results.jsonl"
    if [[ $jobs_ok == 0 ]] || { [[ $mode == test ]] && grep -qE ', [1-9][0-9]* findings' "$summary"; }; then
        return 1
    fi
    return 0
}

# Write the data files. The tiny models come from make_tiny_model of the release none build.
make_data() {
    mkdir -p "$DATA"
    local src=${FUZZ_MODEL_SRC:-$REPO/weights/gguf/Qwen3.5-2B-Q8_0.gguf}
    (cd "$REPO" && uv run --frozen python "$HERE/make_data.py" --model "$src" --data-dir "$DATA" --no-seeds)
    build_tree release none make_tiny_model
    "$(tree_dir release none)/make_tiny_model" "$DATA/tiny-qwen35-f32.gguf" f32 > /dev/null
    "$(tree_dir release none)/make_tiny_model" "$DATA/tiny-qwen35-q8_0.gguf" q8_0 > /dev/null
    ls -l "$DATA"
}

# Build the device targets for arm64 Android with the profile $1 and the configuration $2 in the
# Snapdragon container. The release profile has the shipped flags: the -march of the preset here,
# the rest from CMakeLists.txt.
phone_build() {
    local profile=$1 san=$2
    fuzz_check_profile "$profile"
    fuzz_check_config "$san" phone
    # shellcheck source=../../../scripts/lib.sh
    source "$REPO/scripts/lib.sh"
    local src_rel="build/fuzz/core-android-src${TREE_TAG:+-$TREE_TAG}"
    local copy="$REPO/$src_rel"
    local rel
    rel=$(phone_rel "$profile" "$san")
    mkdir -p "$copy" "$REPO/$rel/out"
    # A private copy of the llama.cpp tree: the main session edits the HTP sources of the submodule.
    # The copy does not keep the file times, thus ninja compiles each changed file again. The
    # submodule goes through tests/sanitizers/llama-copy.sh (the shared landing lock, the check
    # against HEAD, and the stamp of the copy). A private tree (FUZZ_LLAMA_DIR) goes through rsync.
    if [[ -n $LLAMA_DIR ]]; then
        rsync -rlc --delete --exclude .git --exclude '/build*/' "$LLAMA_DIR/" "$copy/llama.cpp/"
    else
        fuzz_llama_copy "$copy/llama.cpp"
    fi
    local runtime
    runtime=$(fuzz_phone_runtime "$san")
    container_run "$SNAPDRAGON_IMAGE" bash -euo pipefail -c "
        cmake -S tests/fuzz/core -B $rel/build -G Ninja \
            -DCMAKE_TOOLCHAIN_FILE=\$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
            -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 \
            -DCMAKE_C_FLAGS='$SHIPPED_MARCH' -DCMAKE_CXX_FLAGS='$SHIPPED_MARCH' \
            -DFUZZ_LLAMA_DIR=/workspace/$src_rel/llama.cpp \
            -DFUZZ_PROFILE=$profile -DFUZZ_SANITIZER=$san -DFUZZ_HEXAGON=ON \
            -DFUZZ_TARGETS='${PHONE_TARGETS// /;}' \
            -DHEXAGON_SDK_ROOT=\$HEXAGON_SDK_ROOT -DHEXAGON_TOOLS_ROOT=\$HEXAGON_TOOLS_ROOT \
            -DPREBUILT_LIB_DIR=android_aarch64 -DGGML_OPENCL=OFF > $rel/configure.log
        cmake --build $rel/build -j$BUILD_JOBS --target $PHONE_TARGETS htp-v79 > $rel/build.log
        # the phone gets copies without debug sections. The build directory keeps the full
        # files for the offline symbolization of a report
        for f in $PHONE_TARGETS; do
            \$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip --strip-debug -o $rel/out/\$f $rel/build/\$f
        done
        install -m 0644 $rel/build/llama/ggml/src/ggml-hexagon/libggml-htp-v79.so $rel/out/
        if [[ -n '$runtime' ]]; then
            install -m 0644 \$(ls \$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/*/lib/linux/$runtime) $rel/out/
        fi
    " || fuzz_die "the phone build $rel failed. Read $REPO/$rel/configure.log and $REPO/$rel/build.log."
    if [[ $san == asan ]]; then
        # The ASan runtime of NDK r29 traps in each new thread on the phone: bionic resets the PAC key
        # through prctl, and the prctl interceptor then fails its own AUTIASP. The runtime of
        # compiler-rt 22.1.8 has the upstream correction of the interceptor, thus it replaces the
        # runtime of the NDK.
        install -m 0644 "$(fuzz_asan_runtime_path)" "$REPO/$rel/out/"
    fi
    mkdir -p "$REPO/$rel/out/seeds"
    local fz
    for fz in $PHONE_TARGETS; do
        cp -r "$HERE/seeds/$fz" "$REPO/$rel/out/seeds/"
    done
    # the shared suppression file of this sanitizer, or an empty file (no suppression)
    if [[ -f "$SHARED/$san.supp" ]]; then
        cp "$SHARED/$san.supp" "$REPO/$rel/out/$san.supp"
    else
        : > "$REPO/$rel/out/$san.supp"
    fi
    ls -l "$REPO/$rel/out"
}

# Print the adb commands of the phone run of the build $1 (profile) $2 (configuration). Each command
# has a time limit of 100 s or less, and each run has a thermal status line before it and a process
# check with the thermal status after it. The logs go to build/fuzz/core-android-<profile>-<config>/phone-logs.
#
# The device runs use FUZZ_NPU_CALIBRATE=1: the target prints each difference to the CPU and does not
# stop on it (P4), because the tolerance of the HTP is not known before the first run. P2, P3, P5 and
# P6 (return codes, finite logits, rollback answers, log errors) still stop the run.
phone_commands() {
    local profile=$1 san=$2
    fuzz_check_profile "$profile"
    fuzz_check_config "$san" phone
    local a="adb -s $FUZZ_PHONE_SERIAL"
    local rel
    rel=$(phone_rel "$profile" "$san")
    local out="$REPO/$rel/out"
    local logs="$REPO/$rel/phone-logs"
    local pd="$PHONE_BASE/$profile-$san${TREE_TAG:+-$TREE_TAG}"
    local runtime sopt
    runtime=$(fuzz_phone_runtime "$san")
    # the options of the one sanitizer of this build, as tests/sanitizers/fuzz-lib.sh gives them
    sopt=$(fuzz_phone_options "$san" "$pd/$san.supp")
    [[ -n $sopt ]] || sopt="FUZZ_NO_SANITIZER=1"
    local env="cd $pd && LD_LIBRARY_PATH=$pd ADSP_LIBRARY_PATH=$pd GGML_NO_BACKTRACE=1 FUZZ_DATA_DIR=$PHONE_BASE/data FUZZ_ARTIFACT_DIR=$pd/art $sopt"
    # each adb command has a hard limit: a short command 30 s, a push, a pull or a run 100 s
    local q="timeout -s KILL 30 $a"
    local thermal check
    thermal=$(fuzz_phone_thermal_cmd)
    check=$(fuzz_phone_check_cmd '[f]uzz_')
    local push_files="$out/fuzz_npu_decode $out/fuzz_recurrent $out/libggml-htp-v79.so $out/$san.supp"
    [[ -n $runtime ]] && push_files+=" $out/$runtime"
    # Step 3: the 2B model on HTP0 against the CPU. Release none runs 8 inputs. A release build with a
    # sanitizer runs the 2 seeds only: one input takes up to 18 s there (release ubsan), and 8 inputs
    # do not complete in the run budget of 90 s. The debug CPU reference is slower: debug none runs
    # one seed with prompts of at most 4 tokens, and debug hwasan does not load the two copies of the
    # 2B model in 90 s.
    local step3 runs2b=8
    [[ $san == none ]] || runs2b=2
    local npu2b_env="$env FUZZ_DEVICE=HTP0 FUZZ_THREADS=6 FUZZ_MODEL=/data/local/tmp/qwen/models/Qwen3.5-2B-Q8_0.gguf FUZZ_NPU_CALIBRATE=1"
    if [[ $profile == release ]]; then
        step3="# 3. The 2B Q8_0 model of the app on HTP0 against the CPU, $runs2b inputs. The two copies of the model
#    need more than 4 GB, thus this run has -rss_limit_mb=8192.
$thermal
timeout -s KILL 100 $a shell \"$npu2b_env timeout -s KILL 90 ./fuzz_npu_decode -runs=$runs2b -seed=2 -rss_limit_mb=8192 -artifact_prefix=art/npu2b- seeds/fuzz_npu_decode\" > $logs/npu-2b.txt 2>&1; tail -n 4 $logs/npu-2b.txt
$check"
    elif [[ $san == hwasan ]]; then
        step3="# 3. No 2B step: the debug hwasan build does not load the two copies of the 2B model in the run budget of 90 s."
    else
        step3="# 3. The 2B Q8_0 model of the app on HTP0 against the CPU, one seed, prompts of at most 4 tokens (the
#    debug CPU reference is slow). The two copies of the model need more than 4 GB, thus -rss_limit_mb=8192.
$thermal
timeout -s KILL 100 $a shell \"$npu2b_env FUZZ_NPU_MAX_TOKENS=4 timeout -s KILL 90 ./fuzz_npu_decode -rss_limit_mb=8192 -artifact_prefix=art/npu2b- seeds/fuzz_npu_decode/rand-00\" > $logs/npu-2b.txt 2>&1; tail -n 4 $logs/npu-2b.txt
$check"
    fi
    cat <<EOF
# ---- phone run of the $profile-$san build ----
mkdir -p $logs

# 1. Push the build (2 targets, the DSP library, ${runtime:-no runtime library}), the tiny models and the seeds.
$thermal
$q shell mkdir -p $PHONE_BASE/data $pd/corpus_npu $pd/corpus_rec $pd/art
timeout -s KILL 100 $a push $push_files $pd/
timeout -s KILL 100 $a push $DATA/tiny-qwen35-q8_0.gguf $DATA/tiny-qwen35-f32.gguf $PHONE_BASE/data/
timeout -s KILL 100 $a push $out/seeds $pd/
$q shell chmod 755 $pd/fuzz_npu_decode $pd/fuzz_recurrent

# 2. HTP0 against the CPU of the phone, tiny Q8_0 model, coverage-guided for 80 s.
$thermal
timeout -s KILL 100 $a shell "$env FUZZ_DEVICE=HTP0 FUZZ_NPU_CALIBRATE=1 timeout -s KILL 90 ./fuzz_npu_decode corpus_npu seeds/fuzz_npu_decode -max_total_time=80 -timeout=30 -rss_limit_mb=4096 -max_len=256 -artifact_prefix=art/npu- -print_final_stats=1" > $logs/npu-tiny.txt 2>&1; tail -n 6 $logs/npu-tiny.txt
$check

$step3

# 4. The recurrent memory target on the CPU of the phone (arm64 kernels), 80 s.
$thermal
timeout -s KILL 100 $a shell "$env FUZZ_THREADS=2 $(known_env)timeout -s KILL 90 ./fuzz_recurrent corpus_rec seeds/fuzz_recurrent -max_total_time=80 -timeout=30 -rss_limit_mb=4096 -max_len=1024 -artifact_prefix=art/rec- -print_final_stats=1" > $logs/rec.txt 2>&1; tail -n 6 $logs/rec.txt
$check

# 5. Pull the crash inputs, if any.
$q shell ls -l $pd/art
timeout -s KILL 100 $a pull $pd/art $logs/art
EOF
}

# The arguments: the mode, the configuration, the options, and the targets.
mode=${1:-}
shift || true
case $mode in
    cpu-asan) mode=fuzz; set -- asan "$@" ;;
    cpu-tsan) mode=fuzz; set -- tsan "$@" ;;
    regress)  mode=test; set -- asan "$@" ;;
esac
case $mode in
    test|fuzz|phone-build|phone-commands)
        san=${1:-}
        shift || true
        [[ -z $LLAMA_DIR || -n $TREE_TAG ]] || fuzz_die "FUZZ_LLAMA_DIR needs FUZZ_TREE_TAG (the suffix of the build directory)"
        [[ -z $LLAMA_DIR || -f $LLAMA_DIR/include/llama.h ]] || fuzz_die "FUZZ_LLAMA_DIR=$LLAMA_DIR holds no include/llama.h"
        profiles="debug release"
        targets=()
        while (( $# > 0 )); do
            case $1 in
                --profile)        profiles=${2:?--profile needs debug or release}; fuzz_check_profile "$profiles"; shift 2 ;;
                --budget-seconds) BUDGET=${2:?--budget-seconds needs a number}; shift 2 ;;
                --jobs)           JOBS=${2:?--jobs needs a number}; shift 2 ;;
                -*)               fuzz_die "the option $1 is not known" ;;
                *)                targets+=("$1"); shift ;;
            esac
        done
        status=0
        for profile in $profiles; do
            case $mode in
                test|fuzz)
                    fuzz_check_config "$san" host
                    run_mode "$mode" "$profile" "$san" "${targets[@]}" || status=1
                    ;;
                phone-build)    phone_build "$profile" "$san" ;;
                phone-commands) phone_commands "$profile" "$san" ;;
            esac
        done
        exit $status
        ;;
    data) make_data ;;
    -h|--help|help) usage 0 ;;
    *)    usage ;;
esac
