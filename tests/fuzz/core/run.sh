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
# none, asan, ubsan, tsan, msan on the host, and none, asan, hwasan, ubsan, tsan
# on the phone. NDK r29 has the TSan runtime. MSan is not available on Android.
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
#   phone-build     Copy the submodule tree into build/fuzz/core-android-src, and
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
#   FUZZ_LLAMA_DIR      A private llama.cpp tree for the test and fuzz modes (the
#                       test of a fix before it lands). The default is the submodule.
#   FUZZ_TREE_TAG       The name of that tree. The build directory gets it as a
#                       suffix: build/fuzz/core-<profile>-<config>-<tag>. It is
#                       necessary with FUZZ_LLAMA_DIR.
#   ADB_SERIAL          The phone of phone-commands (default 192.168.14.130:5555).
#
# Each fuzz run uses nice 10, -rss_limit_mb=4096 and -timeout=30. libFuzzer stops
# at the first crash, thus the script starts it again on the same corpus until
# the budget is spent (at most 200 starts). A start that does not stop 120 s
# after its budget gets SIGKILL: a sanitizer report can hang in the death
# callback of libFuzzer (seen with TSan). The logs, the corpus and the crash
# inputs of a target are in build/fuzz/core-<profile>-<config>/runs/<target>. Each
# run writes one JSON line for each target to build/fuzz/core-<profile>-<config>/results.jsonl:
# {area, target, sanitizer, profile, mode, seconds, executions, findings, crash_files}.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
SHARED="$REPO/tests/sanitizers"
DATA="$REPO/build/fuzz/core/data"
BUDGET=${FUZZ_BUDGET:-600}
JOBS=${FUZZ_JOBS:-4}
BUILD_JOBS=${FUZZ_BUILD_JOBS:-8}
KNOWN=${FUZZ_KNOWN:-1}
LLAMA_DIR=${FUZZ_LLAMA_DIR:-}
TREE_TAG=${FUZZ_TREE_TAG:-}
ADB_SERIAL=${ADB_SERIAL:-192.168.14.130:5555}
PHONE_BASE=/data/local/tmp/qwen/fuzz/core
SHIPPED_MARCH="-march=armv8.7a+fp16+dotprod+i8mm"

ALL_TARGETS="fuzz_gguf fuzz_model_load fuzz_tokenizer fuzz_chat fuzz_sampler fuzz_recurrent fuzz_npu_decode tsan_threadpool tsan_decode tsan_topset"
TSAN_TARGETS="fuzz_recurrent fuzz_npu_decode tsan_threadpool tsan_decode tsan_topset"
PHONE_TARGETS="fuzz_npu_decode fuzz_recurrent"

# The switch of each known finding (rule R8 and R13).
declare -A SWITCH=(
    [embd-host-mrope]=FUZZ_RECURRENT_KNOWN_EMBD_HOST
    [ubatch-tail]=FUZZ_RECURRENT_KNOWN_UBATCH_TAIL
    [kv-keep-streams]=FUZZ_RECURRENT_KNOWN_KV_KEEP
    [recurrent-shared-rollback]=FUZZ_RECURRENT_KNOWN_SHARED_ROLLBACK
    [copy-into-used-sequence]=FUZZ_RECURRENT_CLEAR_BEFORE_COPY
    [dist-inf]=FUZZ_SAMPLER_KNOWN_DIST_INF
    [sampler-bucket-cast]=FUZZ_SAMPLER_KNOWN_BUCKET_CAST
    [tsan-topset-counters]=FUZZ_TOPSET_KNOWN_COUNTERS
)

# The report of each known finding, as an extended regular expression on the run log. A
# regression input "reproduces" its finding when its log matches this expression.
declare -A EXPECT=(
    [dist-inf]="P4: candidate [0-9]+ has the logit \+Inf|Assertion .found. failed"
    [sampler-bucket-cast]='P3: the top set|llama-sampler\.cpp:[0-9]+:[0-9]+: runtime error: .*outside the range'
    [embd-host-mrope]='llama_batch_allocr::ubatch_add|P2: after decode'
    [kv-keep-streams]='P2: after'
    [recurrent-shared-rollback]='cannot write shared recurrent state|P2: after'
    [copy-into-used-sequence]='P2: after'
    [ubatch-tail]='GGML_ASSERT\(n_ubatch > n_keep_tail\)'
    [tsan-topset-counters]='ThreadSanitizer: data race'
)

# Print the header comment of this file (from line 2 to the first empty line) as the usage text, then
# stop with the status $1 (2 when it is not given: a wrong command line).
usage() {
    local line first=1 status=${1:-2}
    while IFS= read -r line; do
        if [[ $first == 1 ]]; then
            first=0
            continue
        fi
        [[ -z $line ]] && break
        line=${line#\#}
        echo "${line# }"
    done < "${BASH_SOURCE[0]}"
    exit "$status"
}

# Write a message to stderr and stop with the code 1.
die() {
    echo "run.sh: $*" >&2
    exit 1
}

# The switches of all known findings, as NAME=1 words. $1, if given, is a switch to leave out.
known_env() {
    local v
    for v in $(printf '%s\n' "${SWITCH[@]}" | sort -u); do
        [[ $v == "${1:-}" ]] || printf '%s=1 ' "$v"
    done
}

# Check the name of a configuration. $1 is the name, $2 is "host" or "phone".
check_san() {
    case "$2:$1" in
        host:none|host:asan|host:ubsan|host:tsan|host:msan) ;;
        phone:none|phone:asan|phone:hwasan|phone:ubsan|phone:tsan) ;;
        *) die "the configuration '$1' is not known for the $2. Host: none asan ubsan tsan msan. Phone: none asan hwasan ubsan tsan." ;;
    esac
}

# Check the name of a profile.
check_profile() {
    [[ $1 == debug || $1 == release ]] || die "the profile '$1' is not known. Use debug or release."
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

# The first error line of a run log on stdin: the signature of a finding.
signature() {
    grep -oE "FUZZ FAILURE: (P[0-9]+: )?[a-zA-Z ,'-]+|[a-z_./-]+:[0-9]+: fatal error|runtime error: [a-z0-9.+ -]*is outside the range|runtime error: [a-z -]+|ERROR: [A-Za-z]+Sanitizer: [a-z-]+|WARNING: ThreadSanitizer: [a-z ]+|WARNING: MemorySanitizer: [a-z-]+|[a-z_./-]+:[0-9]+: GGML_ASSERT\([^)]*\)|GGML_ABORT|what\(\): .*|Assertion .*|libFuzzer: [a-z-]+|cannot write shared[a-z ]+" \
        | head -n 1 || true
}

# The build directory of the profile $1 and the configuration $2 (with the suffix of a private tree).
tree_dir() {
    echo "$REPO/build/fuzz/core-$1-$2${TREE_TAG:+-$TREE_TAG}"
}

# Configure and build the host tree of the profile $1 and the configuration $2 with the targets $3.
build_tree() {
    local profile=$1 san=$2 targets=$3
    local dir
    dir=$(tree_dir "$profile" "$san")
    [[ -f "$SHARED/profile-$profile.cmake" && -f "$SHARED/$san.cmake" ]] \
        || die "the shared files $SHARED/profile-$profile.cmake and $SHARED/$san.cmake are necessary"
    local -a src=()
    [[ -n $LLAMA_DIR ]] && src=(-DFUZZ_LLAMA_DIR="$LLAMA_DIR")
    cmake -G Ninja -S "$HERE" -B "$dir" "${src[@]}" \
        -DFUZZ_TARGETS="${targets// /;}" \
        -DCMAKE_AR="$(command -v llvm-ar)" -DCMAKE_RANLIB="$(command -v llvm-ranlib)" \
        -C "$SHARED/profile-$profile.cmake" -C "$SHARED/$san.cmake" > "$dir.configure.log" 2>&1 \
        || die "the configure of $dir failed. Read $dir.configure.log."
    # shellcheck disable=SC2086
    nice -n 10 cmake --build "$dir" -j"$BUILD_JOBS" --target $targets > "$dir.build.log" 2>&1 \
        || die "the build of $dir failed. Read $dir.build.log."
}

# Write one JSON line for a target to results.jsonl. The crash files are the remaining arguments.
result_line() {
    local dir=$1 target=$2 san=$3 profile=$4 mode=$5 seconds=$6 execs=$7 findings=$8
    shift 8
    local files
    files=$(printf '%s\n' "$@" | jq -R . | jq -sc 'map(select(length > 0))')
    jq -nc --arg area core --arg target "$target" --arg sanitizer "$san" --arg profile "$profile" --arg mode "$mode" \
        --argjson seconds "$seconds" --argjson executions "$execs" --argjson findings "$findings" \
        --argjson crash_files "$files" \
        '{area: $area, target: $target, sanitizer: $sanitizer, profile: $profile, mode: $mode, seconds: $seconds,
          executions: $executions, findings: $findings, crash_files: $crash_files}' >> "$dir/results.jsonl"
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
    # shellcheck source=../../sanitizers/env.sh
    source "$SHARED/env.sh"
    sanitizer_env "$san"
    local -a known=()
    [[ "$KNOWN" == 1 ]] && read -r -a known <<< "$(known_env)"
    local start=$SECONDS left starts=0 rc
    while :; do
        left=$(( BUDGET - (SECONDS - start) ))
        (( left > 5 && starts < 200 )) || break
        starts=$(( starts + 1 ))
        rc=0
        env "${known[@]}" GGML_NO_BACKTRACE=1 FUZZ_DATA_DIR="$DATA" FUZZ_ARTIFACT_DIR="$out/artifacts" \
            timeout -s KILL $(( left + 120 )) nice -n 10 "$dir/$fz" "$out/corpus" "$HERE/seeds/$fz" \
                -max_total_time="$left" -rss_limit_mb=4096 -timeout=30 \
                -max_len="$(max_len "$fz")" -artifact_prefix="$out/artifacts/" -print_final_stats=1 \
                >> "$log" 2>&1 || rc=$?
        echo "run.sh: start $starts ended with code $rc after $(( SECONDS - start )) s" >> "$log"
        [[ $rc == 0 ]] && break
    done
    # the executions: the sum of the last progress count of each start
    local execs=0 prev=0 n
    while read -r n; do
        if (( n < prev )); then
            execs=$(( execs + prev ))
        fi
        prev=$n
    done < <(grep -oE '^#[0-9]+' "$log" | tr -d '#')
    execs=$(( execs + prev ))
    local cov
    cov=$(grep -oE 'cov: [0-9]+ ft: [0-9]+' "$log" | tail -n 1 || true)
    local -a arts=()
    mapfile -t arts < <(find "$out/artifacts" -type f | sort)
    result_line "$dir" "$fz" "$san" "$profile" fuzz "$(( SECONDS - start ))" "$execs" "${#arts[@]}" "${arts[@]}"
    echo "core-$profile-$san${TREE_TAG:+-$TREE_TAG} $fz: $(( SECONDS - start )) s, $starts starts, $execs executions, $cov, corpus $(find "$out/corpus" -type f | wc -l), crash files ${#arts[@]}"
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
    # shellcheck source=../../sanitizers/env.sh
    source "$SHARED/env.sh"
    sanitizer_env "$san"
    local start=$SECONDS findings=0 execs=0 rc f name finding sw expect sig tmp art
    local -a failed=() all=() others=()
    local notes=""
    read -r -a all <<< "$(known_env)"
    tmp=$(mktemp)
    # libFuzzer writes a crash file for each input that fails. These inputs are known, thus the
    # crash files go to a temporary directory and not to the current directory.
    art=$(mktemp -d)
    rc=0
    env "${all[@]}" GGML_NO_BACKTRACE=1 FUZZ_DATA_DIR="$DATA" FUZZ_ARTIFACT_DIR="$art" timeout -s KILL 600 "$dir/$fz" -runs=0 -rss_limit_mb=4096 \
        -timeout=60 -artifact_prefix="$art/" -max_len="$(max_len "$fz")" "$HERE/seeds/$fz" >> "$log" 2>&1 || rc=$?
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
                -rss_limit_mb=4096 -timeout=60 -artifact_prefix="$art/" "$f" > "$tmp" 2>&1 || rc=$?
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
        [[ -n $expect ]] || die "the finding $finding of $name has a switch but no expected report in run.sh"
        # A: all the switches on, less the switch of this finding. The input must show its finding,
        # or pass when this configuration cannot show it.
        read -r -a others <<< "$(known_env "$sw")"
        rc=0
        env "${others[@]}" GGML_NO_BACKTRACE=1 FUZZ_DATA_DIR="$DATA" FUZZ_ARTIFACT_DIR="$art" timeout -s KILL 600 "$dir/$fz" -rss_limit_mb=4096 \
            -timeout=60 -artifact_prefix="$art/" "$f" > "$tmp" 2>&1 || rc=$?
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
            -timeout=60 -artifact_prefix="$art/" "$f" > "$tmp" 2>&1 || rc=$?
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
    result_line "$dir" "$fz" "$san" "$profile" test "$(( SECONDS - start ))" "$execs" "$findings" "${failed[@]}"
    echo "core-$profile-$san${TREE_TAG:+-$TREE_TAG} $fz: $execs runs, $findings findings |$notes"
}

# Run the mode $1 (test or fuzz) with the profile $2 and the configuration $3 on the targets that
# follow, JOBS at a time. Returns 1 when the test mode has a finding.
run_mode() {
    local mode=$1 profile=$2 san=$3
    shift 3
    local targets="$*"
    if [[ -z $targets ]]; then
        targets=$ALL_TARGETS
        [[ $san == tsan ]] && targets=$TSAN_TARGETS
    fi
    [[ -f "$DATA/tiny-qwen35-f32.gguf" && -f "$DATA/qwen35-vocab.gguf" ]] || die "no data in $DATA. Run 'tests/fuzz/core/run.sh data' first."
    build_tree "$profile" "$san" "$targets"
    local dir
    dir=$(tree_dir "$profile" "$san")
    local summary="$dir/$mode-summary.txt"
    : > "$summary"
    local fz
    for fz in $targets; do
        while (( $(jobs -rp | wc -l) >= JOBS )); do
            sleep 5
        done
        if [[ $mode == fuzz ]]; then
            fuzz_one "$profile" "$san" "$fz" >> "$summary" &
        else
            test_one "$profile" "$san" "$fz" >> "$summary" &
        fi
    done
    wait
    cat "$summary"
    echo "run.sh: the JSON lines are in $dir/results.jsonl"
    if [[ $mode == test ]] && grep -qE ', [1-9][0-9]* findings' "$summary"; then
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

# The runtime library that a phone build needs next to its executables, or nothing.
phone_runtime() {
    case $1 in
        asan)   echo libclang_rt.asan-aarch64-android.so ;;
        hwasan) echo libclang_rt.hwasan-aarch64-android.so ;;
        tsan)   echo libclang_rt.tsan-aarch64-android.so ;;
        ubsan)  echo libclang_rt.ubsan_standalone-aarch64-android.so ;;
        *)      echo "" ;;
    esac
}

# Build the device targets for arm64 Android with the profile $1 and the configuration $2 in the
# Snapdragon container. The release profile has the shipped flags: the -march of the preset here,
# the rest from CMakeLists.txt.
phone_build() {
    local profile=$1 san=$2
    check_profile "$profile"
    check_san "$san" phone
    # shellcheck source=../../../scripts/lib.sh
    source "$REPO/scripts/lib.sh"
    local src="$REPO/build/fuzz/core-android-src"
    local rel="build/fuzz/core-android-$profile-$san"
    mkdir -p "$src" "$REPO/$rel/out"
    # A private copy of the submodule: the main session edits the HTP sources.
    rsync -a --delete --exclude .git --exclude '/build*/' "$REPO/third_party/llama.cpp/" "$src/llama.cpp/"
    local runtime
    runtime=$(phone_runtime "$san")
    container_run "$SNAPDRAGON_IMAGE" bash -euo pipefail -c "
        cmake -S tests/fuzz/core -B $rel/build -G Ninja \
            -DCMAKE_TOOLCHAIN_FILE=\$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
            -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 \
            -DCMAKE_C_FLAGS='$SHIPPED_MARCH' -DCMAKE_CXX_FLAGS='$SHIPPED_MARCH' \
            -DFUZZ_LLAMA_DIR=/workspace/build/fuzz/core-android-src/llama.cpp \
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
    " || die "the phone build $rel failed. Read $REPO/$rel/configure.log and $REPO/$rel/build.log."
    if [[ $san == asan ]]; then
        # The ASan runtime of NDK r29 traps in each new thread on the phone: bionic resets the PAC key
        # through prctl, and the prctl interceptor then fails its own AUTIASP. The runtime of
        # compiler-rt 22.1.8 (tests/sanitizers/build-asan-android-runtime.sh) has the upstream
        # correction of the interceptor, thus it replaces the runtime of the NDK.
        local rt22="$REPO/build/fuzz/ops/phone/asan-rt22/libclang_rt.asan-aarch64-android.so"
        [[ -f $rt22 ]] || die "the ASan runtime $rt22 of compiler-rt 22.1.8 is missing (tests/sanitizers/build-asan-android-runtime.sh builds it)"
        [[ $(sha256sum "$rt22" | cut -c1-8) == 546f2a86 ]] || die "the ASan runtime $rt22 does not have the hash 546f2a86"
        install -m 0644 "$rt22" "$REPO/$rel/out/"
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
    check_profile "$profile"
    check_san "$san" phone
    local a="adb -s $ADB_SERIAL"
    local out="$REPO/build/fuzz/core-android-$profile-$san/out"
    local logs="$REPO/build/fuzz/core-android-$profile-$san/phone-logs"
    local pd="$PHONE_BASE/$profile-$san"
    local runtime
    runtime=$(phone_runtime "$san")
    # the options of the one sanitizer of this build, as tests/sanitizers/env.sh gives them on the host
    local sopt="FUZZ_NO_SANITIZER=1"
    case $san in
        asan)   sopt="ASAN_OPTIONS=halt_on_error=1:allocator_may_return_null=1:detect_leaks=0:suppressions=$pd/asan.supp" ;;
        hwasan) sopt="HWASAN_OPTIONS=halt_on_error=1:allocator_may_return_null=1" ;;
        ubsan)  sopt="UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1:report_error_type=1:suppressions=$pd/ubsan.supp" ;;
        tsan)   sopt="TSAN_OPTIONS=halt_on_error=1:second_deadlock_stack=1:suppressions=$pd/tsan.supp" ;;
    esac
    local env="cd $pd && LD_LIBRARY_PATH=$pd ADSP_LIBRARY_PATH=$pd GGML_NO_BACKTRACE=1 FUZZ_DATA_DIR=$PHONE_BASE/data FUZZ_ARTIFACT_DIR=$pd/art $sopt"
    local thermal="$a shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
    local check="$a shell 'pgrep -a fuzz_; dumpsys thermalservice | grep \"Thermal Status\"'"
    local push_files="$out/fuzz_npu_decode $out/fuzz_recurrent $out/libggml-htp-v79.so $out/$san.supp"
    [[ -n $runtime ]] && push_files+=" $out/$runtime"
    cat <<EOF
# ---- phone run of the $profile-$san build ----
mkdir -p $logs

# 1. Push the build (2 targets, the DSP library, ${runtime:-no runtime library}), the tiny models and the seeds.
$thermal
$a shell mkdir -p $PHONE_BASE/data $pd/corpus_npu $pd/corpus_rec $pd/art
timeout -s KILL 100 $a push $push_files $pd/
timeout -s KILL 100 $a push $DATA/tiny-qwen35-q8_0.gguf $DATA/tiny-qwen35-f32.gguf $PHONE_BASE/data/
timeout -s KILL 100 $a push $out/seeds $pd/
$a shell chmod 755 $pd/fuzz_npu_decode $pd/fuzz_recurrent

# 2. HTP0 against the CPU of the phone, tiny Q8_0 model, coverage-guided for 80 s.
$thermal
timeout -s KILL 100 $a shell "$env FUZZ_DEVICE=HTP0 FUZZ_NPU_CALIBRATE=1 timeout -s KILL 90 ./fuzz_npu_decode corpus_npu seeds/fuzz_npu_decode -max_total_time=80 -timeout=30 -rss_limit_mb=4096 -max_len=256 -artifact_prefix=art/npu- -print_final_stats=1" > $logs/npu-tiny.txt 2>&1; tail -n 6 $logs/npu-tiny.txt
$check

# 3. The 2B Q8_0 model of the app on HTP0 against the CPU, 8 inputs. The two copies of the model
#    need more than 4 GB, thus this run has -rss_limit_mb=8192.
$thermal
timeout -s KILL 100 $a shell "$env FUZZ_DEVICE=HTP0 FUZZ_THREADS=6 FUZZ_MODEL=/data/local/tmp/qwen/models/Qwen3.5-2B-Q8_0.gguf FUZZ_NPU_CALIBRATE=1 timeout -s KILL 90 ./fuzz_npu_decode -runs=8 -seed=2 -rss_limit_mb=8192 -artifact_prefix=art/npu2b- seeds/fuzz_npu_decode" > $logs/npu-2b.txt 2>&1; tail -n 4 $logs/npu-2b.txt
$check

# 4. The recurrent memory target on the CPU of the phone (arm64 kernels), 80 s.
$thermal
timeout -s KILL 100 $a shell "$env FUZZ_THREADS=2 $(known_env)timeout -s KILL 90 ./fuzz_recurrent corpus_rec seeds/fuzz_recurrent -max_total_time=80 -timeout=30 -rss_limit_mb=4096 -max_len=1024 -artifact_prefix=art/rec- -print_final_stats=1" > $logs/rec.txt 2>&1; tail -n 6 $logs/rec.txt
$check

# 5. Pull the crash inputs, if any.
$a shell ls -l $pd/art
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
        [[ -z $LLAMA_DIR || -n $TREE_TAG ]] || die "FUZZ_LLAMA_DIR needs FUZZ_TREE_TAG (the suffix of the build directory)"
        [[ -z $LLAMA_DIR || -f $LLAMA_DIR/include/llama.h ]] || die "FUZZ_LLAMA_DIR=$LLAMA_DIR holds no include/llama.h"
        profiles="debug release"
        targets=()
        while (( $# > 0 )); do
            case $1 in
                --profile)        profiles=${2:?--profile needs debug or release}; check_profile "$profiles"; shift 2 ;;
                --budget-seconds) BUDGET=${2:?--budget-seconds needs a number}; shift 2 ;;
                --jobs)           JOBS=${2:?--jobs needs a number}; shift 2 ;;
                -*)               die "the option $1 is not known" ;;
                *)                targets+=("$1"); shift ;;
            esac
        done
        status=0
        for profile in $profiles; do
            case $mode in
                test|fuzz)
                    check_san "$san" host
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
