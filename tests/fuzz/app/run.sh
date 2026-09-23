#!/usr/bin/env bash
# Build and run the fuzzers of the native layer of the app and the JVM fuzz
# tests of its persistence.
#
#   tests/fuzz/app/run.sh test <none|asan|ubsan|tsan|msan> [--profile debug|release] [--budget-seconds N] [--jobs N]
#   tests/fuzz/app/run.sh fuzz <none|asan|ubsan|tsan|msan> [--profile debug|release] [--budget-seconds N] [--jobs N]
#   tests/fuzz/app/run.sh build <none|asan|ubsan|tsan|msan> [--profile debug|release]
#   tests/fuzz/app/run.sh jvm
#   tests/fuzz/app/run.sh phone-build <none|asan|hwasan|ubsan> [--profile debug|release]
#   tests/fuzz/app/run.sh phone-commands <none|asan|hwasan|ubsan> [--profile debug|release]
#
# The modes:
#   test            Build build/fuzz/app-<profile>-<sanitizer> and run each
#                   fuzzer one time on each of its seeds and regression inputs,
#                   with no mutation, then the scenario of each finding. Stop
#                   with the code 1 when one of them reports a finding.
#   fuzz            Build build/fuzz/app-<profile>-<sanitizer> and mutate each
#                   target for the budget (the default is 600 seconds):
#                   fuzz_jni_api, fuzz_jni_threads, fuzz_jni_threads_free
#                   (frees from the second thread), fuzz_caches, fuzz_spec_policy.
#   build           Only build.
#   jvm             Run the JVM fuzz tests (*FuzzTest) with Gradle in the APK
#                   container, on a staged copy of android/. The environment
#                   variables QWEN_FUZZ_ITERATIONS, QWEN_FUZZ_SEED and
#                   QWEN_FUZZ_FINDINGS=1 go to the tests.
#   phone-build     Build the fuzzers and the driver for arm64 Android in the
#                   Snapdragon container into
#                   build/fuzz/app-android-<profile>-<sanitizer>, and stage the
#                   files for the phone into build/fuzz/app/phone-<profile>-<sanitizer>.
#                   An ASan build uses the ASan runtime of compiler-rt 22.1.8
#                   from build/fuzz/ops/phone/asan-rt22 (the runtime of NDK r29
#                   traps in each new thread on the phone).
#   phone-commands  Print the adb commands that push and run a phone build.
#                   This script never runs adb.
#
# The profiles: debug (-O1 -g, no NDEBUG, no LTO) and release (the flags that
# ship: -O3 -DNDEBUG, and -flto with the floating-point flags of the preset
# for llama.cpp; refer to tests/fuzz/app/CMakeLists.txt). Without --profile,
# test, fuzz and build run debug, then release.
#
# The older names stay: cpu-asan [S] is "fuzz asan --budget-seconds S",
# cpu-tsan [S] is "fuzz tsan --budget-seconds S", and regress and scenarios
# are "test asan".
#
# One sanitizer for each build and each run. Each run of test and fuzz
# writes one JSON line for each target to
# build/fuzz/app-<profile>-<sanitizer>/results.jsonl: {area, profile, target,
# sanitizer, mode, seconds, executions, findings, crash_files}.
#
# The environment:
#   FUZZ_BUILD_JOBS The jobs of a build (the default is 8).
#   FAKEJNI_RELAX   The checks of the fake VM that only report in the fuzz mode
#                   (the default is "pending": finding jni-pending-exception has
#                   its own scenario and regression). The test mode relaxes none.
#   FUZZ_APP_SKIP_KNOWN  1 (the default of the fuzz mode) keeps the fuzzers away
#                   from the findings that a scenario reproduces already. The
#                   test mode uses 0.
#
# The llama.cpp tree is a private snapshot in build/fuzz/app/llama-snap: the
# first run copies it from the submodule, thus a later edit of the submodule
# does not change a fuzz build. Remove the snapshot to take a new one.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
OUT="$REPO/build/fuzz/app"
SNAP="$OUT/llama-snap"
MODELS="$OUT/models"
BUILD_JOBS=${FUZZ_BUILD_JOBS:-8}
PROFILE=debug

# The build directory of the sanitizer $1 in the current profile.
bdir() {
    echo "$REPO/build/fuzz/app-$PROFILE-$1"
}
TARGETS="fuzz_jni_api fuzz_jni_threads fuzz_jni_threads_free fuzz_caches fuzz_spec_policy"
SCENARIOS="image-shape image-twice jni-pending spec-disable sampler-nan priority"

export FUZZ_APP_MODEL_DIR="$MODELS"
export FUZZ_APP_WORK="$OUT/work"

# Print the usage text (the comment block at the top of this file) and stop with the code $1
# (the default is 2). --help prints it to stdout and stops with 0.
usage() {
    local line first=1 code=${1:-2}
    while IFS= read -r line; do
        if ((first)); then
            first=0
            continue
        fi
        [[ -z "$line" ]] && break
        line=${line#\#}
        if ((code == 0)); then
            echo "${line# }"
        else
            echo "${line# }" >&2
        fi
    done < "${BASH_SOURCE[0]}"
    exit "$code"
}

# Write a message to stderr and stop with the code 1.
die() {
    echo "run.sh: $*" >&2
    exit 1
}

# Copy the submodule into the private snapshot, one time.
snapshot() {
    if [[ -f "$SNAP/CMakeLists.txt" ]]; then
        return
    fi
    echo "run.sh: copy third_party/llama.cpp into $SNAP"
    mkdir -p "$SNAP"
    rsync -a --exclude=.git --exclude='build*/' "$REPO/third_party/llama.cpp/" "$SNAP/"
}

# Write the tiny models, one time.
models() {
    if [[ -f "$MODELS/tiny-qwen35-f32.gguf" && -f "$MODELS/tiny-qwen35-mmproj.gguf" ]]; then
        return
    fi
    snapshot
    uv run --no-project --with numpy --with pyyaml python "$HERE/tools/make_tiny_model.py" \
        --gguf-py "$SNAP/gguf-py" --template "$SNAP/models/templates/Qwen3.5-4B.jinja" --out "$MODELS"
}

# The one suppression file of UBSan of the campaign (rule R5). This area has no file of its own.
UBSAN_SUPP="$REPO/tests/sanitizers/ubsan.supp"

# Set the runtime options of the sanitizer $1: the shared ones of the campaign
# (tests/sanitizers/env.sh) when they exist, else the same values here.
runtime_options() {
    case $1 in
        none | asan | ubsan | tsan | msan) ;;
        *) die "unknown sanitizer '$1': none, asan, ubsan, tsan or msan" ;;
    esac
    if [[ -f "$REPO/tests/sanitizers/env.sh" ]]; then
        # shellcheck source=/dev/null
        source "$REPO/tests/sanitizers/env.sh"
        sanitizer_env "$1"
        return
    fi
    unset ASAN_OPTIONS LSAN_OPTIONS UBSAN_OPTIONS TSAN_OPTIONS MSAN_OPTIONS
    local common="halt_on_error=1:allocator_may_return_null=1"
    case $1 in
        asan) export ASAN_OPTIONS="$common:detect_leaks=1:detect_stack_use_after_return=1" ;;
        ubsan) export UBSAN_OPTIONS="$common:print_stacktrace=1${UBSAN_SUPP:+:suppressions=$UBSAN_SUPP}" ;;
        tsan) export TSAN_OPTIONS="$common:second_deadlock_stack=1" ;;
        msan) export MSAN_OPTIONS="$common" ;;
    esac
}

# Configure and build the host fuzzers with the sanitizer $1 into build/fuzz/app-$1. The flags
# come from the initial cache tests/sanitizers/$1.cmake of the campaign when it exists.
build_host() {
    local san=$1 dir
    dir=$(bdir "$1")
    local -a init=()
    snapshot
    models
    # The shared initial caches of the campaign give the flags: the profile
    # first, then the sanitizer. Without them tests/fuzz/app/CMakeLists.txt
    # gives the same flags itself.
    if [[ -f "$REPO/tests/sanitizers/profile-$PROFILE.cmake" && -f "$REPO/tests/sanitizers/$san.cmake" ]]; then
        init=(-C "$REPO/tests/sanitizers/profile-$PROFILE.cmake" -C "$REPO/tests/sanitizers/$san.cmake")
        # A build directory of an earlier configuration keeps its flags in its cache: it goes.
        if [[ -f "$dir/CMakeCache.txt" && ! -f "$dir/.initial-cache" ]]; then
            rm -rf "$dir"
        fi
    fi
    mkdir -p "$dir/logs"
    cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DFUZZ_SANITIZER="$san" -DFUZZ_PROFILE="$PROFILE" \
        "${init[@]}" -S "$HERE" -B "$dir" -G Ninja \
        -DLLAMA_CPP_DIR="$SNAP" -DFUZZ_APP_MODEL_DIR="$MODELS" > "$dir/logs/configure.log" 2>&1 \
        || die "the configuration of $dir failed, see $dir/logs/configure.log"
    if ((${#init[@]} > 0)); then
        : > "$dir/.initial-cache"
    fi
    nice -n 10 cmake --build "$dir" -j"$BUILD_JOBS" \
        --target fuzz_jni_api fuzz_jni_threads fuzz_caches fuzz_spec_policy app_fuzz_driver > "$dir/logs/build.log" 2>&1 \
        || die "the build of $dir failed, see $dir/logs/build.log"
}

# The binary of a target: fuzz_jni_threads_free is fuzz_jni_threads with frees from the second thread.
binary_of() {
    case $1 in
        fuzz_jni_threads_free) echo fuzz_jni_threads ;;
        *) echo "$1" ;;
    esac
}

# The seed directory of a target.
seeds_of() {
    case $1 in
        fuzz_jni_api) echo "$HERE/seeds/jni_api" ;;
        fuzz_jni_threads | fuzz_jni_threads_free) echo "$HERE/seeds/jni_threads" ;;
        fuzz_caches) echo "$HERE/seeds/caches" ;;
        fuzz_spec_policy) echo "$HERE/seeds/spec_policy" ;;
    esac
}

# The environment assignments of a target under the sanitizer $2.
env_of() {
    local target=$1 san=$2
    case $target in
        fuzz_jni_api | fuzz_jni_threads) echo "FUZZ_APP_MAX_OPS=24" ;;
        fuzz_jni_threads_free) echo "FUZZ_APP_MAX_OPS=24 FUZZ_APP_CROSS_FREE=1" ;;
    esac
    # The exactness oracle runs in the other builds: under ThreadSanitizer the time goes to the threads.
    if [[ $san == tsan && $target == fuzz_jni_* ]]; then
        echo "FUZZ_APP_ORACLE=0"
    fi
}

# The maximum input length of a target.
max_len_of() {
    case $1 in
        fuzz_spec_policy) echo 8192 ;;
        *) echo 1024 ;;
    esac
}

# The task of a known report outside of this area that stops a run, or nothing: $1 the log.
# Task #127: the type traits of ggml-cpu are called through a function pointer of another
# type (UBSan function-type-mismatch in get_rows of a Q8_0 model). fuzz-ops owns the fix.
blocker_of() {
    if grep -q 'through pointer to incorrect function type' "$1" 2> /dev/null; then
        echo "#127"
    fi
}

# Append one result line: target, sanitizer, mode, seconds, executions, then the crash files.
# With BLOCKED=<task> and BLOCKED_LOG=<log> set, the line records a run that a known
# report outside of this area stopped: no finding of this area, and not a pass.
result() {
    local target=$1 san=$2 mode=$3 seconds=$4 executions=$5
    shift 5
    local files
    files=$(printf '%s\n' "$@" | jq -R . | jq -sc 'map(select(length > 0))')
    jq -nc --arg profile "$PROFILE" --arg target "$target" --arg san "$san" --arg mode "$mode" \
        --argjson seconds "$seconds" --argjson executions "$executions" --argjson files "$files" \
        --arg blocked "${BLOCKED:-}" --arg blocked_log "${BLOCKED_LOG:-}" \
        '{area: "app", profile: $profile, target: $target, sanitizer: $san, mode: $mode, seconds: $seconds,
          executions: $executions, findings: ($files | length), crash_files: $files}
         + (if $blocked == "" then {} else {blocked_by: $blocked, blocked_log: $blocked_log} end)' \
        >> "$(bdir "$san")/results.jsonl"
}

# Mutate one target for the budget: $1 the target, $2 the sanitizer, $3 the budget in seconds.
fuzz_one() {
    local target=$1 san=$2 budget=$3 dir
    dir=$(bdir "$2")
    local bin corpus art log t0 t1 runs rc
    bin="$dir/$(binary_of "$target")"
    # A corpus for each build, from the seeds: libFuzzer runs the whole corpus before it
    # checks -max_total_time, and a corpus of a faster build can take longer than the budget.
    corpus="$dir/corpus/$target"
    art="$dir/artifacts/$target"
    log="$dir/logs/fuzz-$target.log"
    rm -rf "$corpus"
    mkdir -p "$corpus" "$art"
    cp -n "$(seeds_of "$target")"/* "$corpus/" 2> /dev/null || true
    local -A before=()
    local f
    for f in "$art"/*; do
        [[ -e "$f" ]] && before[$f]=1
    done
    t0=$(date +%s)
    # shellcheck disable=SC2046
    env FAKEJNI_RELAX="${FAKEJNI_RELAX-pending}" FUZZ_APP_SKIP_KNOWN="${FUZZ_APP_SKIP_KNOWN:-1}" \
        FUZZ_ARTIFACT_DIR="$art" $(env_of "$target" "$san") \
        nice -n 10 timeout -s KILL $((budget + 300)) "$bin" -max_total_time="$budget" -rss_limit_mb=4096 \
        -max_len="$(max_len_of "$target")" -timeout=180 -print_final_stats=1 -close_fd_mask=1 \
        -artifact_prefix="$art/" "$corpus" > "$log" 2>&1 && rc=0 || rc=$?
    t1=$(date +%s)
    runs=$(grep -o 'stat::number_of_executed_units: *[0-9]*' "$log" | grep -o '[0-9]*$' || true)
    local -a crashes=()
    # The outer timeout killed a run that did not stop by itself: a hang (for
    # example ThreadSanitizer in the death callback of libFuzzer). It is a finding.
    if ((rc == 137)); then
        echo "==HARNESS== hang: the run did not stop within $((budget + 300)) s and was killed" >> "$log"
        crashes+=("$log")
    fi
    for f in "$art"/*; do
        # A slow-unit file of libFuzzer is an input that took more than 10 s, not a finding.
        case ${f##*/} in
            crash-* | leak-* | timeout-* | oom-*) [[ -z "${before[$f]:-}" ]] && crashes+=("$f") ;;
        esac
    done
    if ((${#crashes[@]} == 0)) && grep -qE 'ERROR: (AddressSanitizer|ThreadSanitizer|MemorySanitizer|LeakSanitizer|libFuzzer)|WARNING: (ThreadSanitizer|MemorySanitizer)|runtime error:|==FAKEJNI== [^r]|==FUZZ-' "$log"; then
        crashes=("$log")
    fi
    local blocked
    blocked=$(blocker_of "$log")
    if [[ -n "$blocked" ]]; then
        BLOCKED=$blocked BLOCKED_LOG=$log result "$target" "$san" fuzz $((t1 - t0)) "${runs:-0}"
        echo "run.sh: fuzz $PROFILE-$san $target: ${runs:-0} executions in $((t1 - t0)) s, blocked by $blocked"
        return
    fi
    result "$target" "$san" fuzz $((t1 - t0)) "${runs:-0}" "${crashes[@]}"
    echo "run.sh: fuzz $PROFILE-$san $target: ${runs:-0} executions in $((t1 - t0)) s, ${#crashes[@]} findings"
}

# Run each seed and regression input of one target one time: $1 the target, $2 the sanitizer.
test_one() {
    local target=$1 san=$2 dir
    dir=$(bdir "$2")
    local bin inputs=() crashes=() t0 t1 f n=0 log blocked="" blocked_log=""
    bin="$dir/$(binary_of "$target")"
    mkdir -p "$dir/logs/test" "$dir/artifacts/test-$target"
    for f in "$(seeds_of "$target")"/* "$HERE/regress/$(binary_of "$target")"/*; do
        [[ -f "$f" ]] && inputs+=("$f")
    done
    t0=$(date +%s)
    for f in "${inputs[@]}"; do
        n=$((n + 1))
        log="$dir/logs/test/$target-$(basename "$f").log"
        # shellcheck disable=SC2046
        if ! env FAKEJNI_RELAX='' FUZZ_APP_SKIP_KNOWN=0 FUZZ_ARTIFACT_DIR="$dir/artifacts/test-$target" \
            $(env_of "$target" "$san") nice -n 10 timeout -s KILL 900 \
            "$bin" -runs=1 -rss_limit_mb=4096 -artifact_prefix="$dir/artifacts/test-$target/" "$f" > "$log" 2>&1; then
            if [[ -n "$(blocker_of "$log")" ]]; then
                blocked=$(blocker_of "$log")
                blocked_log=$log
            else
                crashes+=("$f")
            fi
        fi
    done
    t1=$(date +%s)
    BLOCKED=$blocked BLOCKED_LOG=$blocked_log result "$target" "$san" test $((t1 - t0)) "$n" "${crashes[@]}"
    echo "run.sh: test $PROFILE-$san $target: $n inputs, ${#crashes[@]} findings${blocked:+, blocked by $blocked}"
    ((${#crashes[@]} == 0)) && [[ -z "$blocked" ]]
}

# Run the scenario of each finding with the driver of the build: $1 the sanitizer.
test_scenarios() {
    local san=$1 dir s rc t0 t1 status=0
    dir=$(bdir "$1")
    mkdir -p "$dir/logs/test"
    for s in $SCENARIOS; do
        t0=$(date +%s)
        FAKEJNI_RELAX='' nice -n 10 timeout -s KILL 300 "$dir/app_fuzz_driver" --scenario "$s" \
            > "$dir/logs/test/scenario-$s.log" 2>&1 && rc=0 || rc=$?
        t1=$(date +%s)
        if ((rc == 0)); then
            result "scenario:$s" "$san" test $((t1 - t0)) 1
        elif [[ -n "$(blocker_of "$dir/logs/test/scenario-$s.log")" ]]; then
            BLOCKED=$(blocker_of "$dir/logs/test/scenario-$s.log") BLOCKED_LOG="$dir/logs/test/scenario-$s.log" \
                result "scenario:$s" "$san" test $((t1 - t0)) 1
            status=1
        else
            result "scenario:$s" "$san" test $((t1 - t0)) 1 "$dir/logs/test/scenario-$s.log"
            status=1
        fi
        local what
        what=$(grep -m1 -oE 'ERROR: [A-Za-z]+Sanitizer: [a-z-]+|runtime error: .{0,100}|Assertion .{0,80}|==FAKEJNI== .{0,160}|check-priority: (PASS|FAIL).*|scenario [a-z-]+: .*' \
            "$dir/logs/test/scenario-$s.log" || true)
        echo "run.sh: test $PROFILE-$san scenario:$s: exit $rc ${what:-(no message, refer to the log)}"
    done
    return $status
}

mode_test() {
    local san=$1 target status=0
    runtime_options "$san"
    build_host "$san"
    for target in $TARGETS; do
        test_one "$target" "$san" || status=1
    done
    test_scenarios "$san" || status=1
    return $status
}

mode_fuzz() {
    local san=$1 budget=$2 jobs=$3 target results lines=0 found
    runtime_options "$san"
    build_host "$san"
    results="$(bdir "$san")/results.jsonl"
    [[ -f "$results" ]] && lines=$(wc -l < "$results")
    for target in $TARGETS; do
        fuzz_one "$target" "$san" "$budget" &
        while (($(jobs -rp | wc -l) >= jobs)); do
            wait -n || true
        done
    done
    wait
    # The exit code is not 0 when one target of this run reported a finding.
    found=$(tail -n +$((lines + 1)) "$results" | jq -s 'map(.findings) | add // 0')
    ((found == 0))
}

jvm() {
    # shellcheck source=../../../scripts/lib.sh
    source "$REPO/scripts/lib.sh"
    local stage="$OUT/jvm/src"
    rm -rf "$stage"
    mkdir -p "$stage" "$OUT/logs"
    (
        cd "$REPO"
        while IFS= read -r -d '' file; do
            [[ -e "$file" ]] || continue
            cp --parents -- "$file" "$stage/"
        done < <(git ls-files -z --cached --others --exclude-standard -- android)
    )
    printf 'sdk.dir=/opt/android-sdk\nllama.dir=/workspace/third_party/llama.cpp\n' > "$stage/android/local.properties"
    local image
    image=$(ensure_apk_image)
    mkdir -p "$REPO/build/cache/gradle"
    # The command runs in the container, thus its variables expand there.
    # shellcheck disable=SC2016
    container_run \
        -v "$stage/android:/workspace/android" -w /workspace/android \
        -e GRADLE_USER_HOME=/workspace/build/cache/gradle \
        -e ANDROID_USER_HOME=/workspace/build/cache/home/.android \
        -e QWEN_FUZZ_ITERATIONS="${QWEN_FUZZ_ITERATIONS:-}" -e QWEN_FUZZ_SEED="${QWEN_FUZZ_SEED:-}" \
        -e QWEN_FUZZ_FINDINGS="${QWEN_FUZZ_FINDINGS:-}" \
        "$image" bash -euo pipefail -c '
./gradlew --no-daemon --no-build-cache --console=plain -Pprebuilt=true -Pandroid.builder.sdkDownload=false \
    :app:testDebugUnitTest --tests "ai.airi.qwenmobile.*FuzzTest" || status=$?
rm -rf /workspace/build/fuzz/app/jvm/results
cp -r app/build/test-results/testDebugUnitTest /workspace/build/fuzz/app/jvm/results
exit ${status:-0}
' 2>&1 | tee "$OUT/logs/jvm.log" | grep -E "FuzzTest|BUILD|tests completed|FAILED" || true
    echo "run.sh: the JUnit results are in $OUT/jvm/results"
}

# The ASan runtime of NDK r29 traps in each new thread on the phone: bionic resets
# the PAC key through prctl, and the prctl interceptor then fails its own AUTIASP.
# The runtime of compiler-rt 22.1.8 has the upstream fix. fuzz-ops builds it, and
# each Android ASan run uses it, first in LD_LIBRARY_PATH.
ASAN_RT22="$REPO/build/fuzz/ops/phone/asan-rt22/libclang_rt.asan-aarch64-android.so"
ASAN_RT22_SHA256=546f2a868184b0a818f7704d1c42990526db878858f38eacf4f1bc75fa93ede1

# Stop when the fixed ASan runtime is missing or has a different hash.
check_asan_rt22() {
    [[ -f "$ASAN_RT22" ]] || die "no $ASAN_RT22: fuzz-ops builds the fixed ASan runtime (compiler-rt 22.1.8)"
    [[ $(sha256sum "$ASAN_RT22" | cut -d' ' -f1) == "$ASAN_RT22_SHA256" ]] \
        || die "$ASAN_RT22 does not have the sha256 $ASAN_RT22_SHA256"
}

# The runtime library of the sanitizer $1 on the phone, or nothing.
android_runtime() {
    case $1 in
        asan) echo libclang_rt.asan-aarch64-android.so ;;
        hwasan) echo libclang_rt.hwasan-aarch64-android.so ;;
        ubsan) echo libclang_rt.ubsan_standalone-aarch64-android.so ;;
        none) ;;
        *) die "unknown phone sanitizer '$1': none, asan, hwasan or ubsan (MSan does not exist on Android)" ;;
    esac
}

phone_build() {
    local san=$1
    android_runtime "$san" > /dev/null
    # shellcheck source=../../../scripts/lib.sh
    source "$REPO/scripts/lib.sh"
    snapshot
    models
    local dsp="$REPO/build/native/llama/ggml/src/ggml-hexagon/libggml-htp-v79.so"
    [[ -f "$dsp" ]] || die "no DSP library at $dsp: the main build makes it"
    local rel_snap=${SNAP#"$REPO"/} runtime name="$PROFILE-$san" prebuilt=""
    local jnilibs="$REPO/android/snapdragon/jniLibs/arm64-v8a"
    runtime=$(android_runtime "$san")
    if [[ $PROFILE == release && $san == none ]]; then
        # The release none run links the shipped libraries: they must be the ones of the last native build.
        diff -q "$REPO/build/hashes-native.txt" <(sha256_table "$jnilibs"/*.so) > /dev/null \
            || die "$jnilibs does not match build/hashes-native.txt: the shipped libraries are not the last build"
        prebuilt=/workspace/android/snapdragon/jniLibs/arm64-v8a
    fi
    mkdir -p "$OUT/logs"
    # The command runs in the container, thus its variables expand there.
    # shellcheck disable=SC2016
    container_run -e JOBS="$BUILD_JOBS" -e SNAP="/workspace/$rel_snap" -e SAN="$san" -e PROFILE="$PROFILE" \
        -e RUNTIME="$runtime" -e PREBUILT="$prebuilt" "$SNAPDRAGON_IMAGE" bash -euo pipefail -c '
cmake -S tests/fuzz/app -B build/fuzz/app-android-$PROFILE-$SAN -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 \
    -DFUZZ_SANITIZER="$SAN" -DFUZZ_PROFILE="$PROFILE" -DLLAMA_CPP_DIR="$SNAP" -DFUZZ_LLAMA_PREBUILT_DIR="$PREBUILT" \
    -DFUZZ_APP_MODEL_DIR=/data/local/tmp/qwen/fuzz/app/models \
    -DHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" -DHEXAGON_TOOLS_ROOT="$HEXAGON_TOOLS_ROOT" -DPREBUILT_LIB_DIR=android_aarch64
nice -n 10 cmake --build build/fuzz/app-android-$PROFILE-$SAN -j"$JOBS" \
    --target fuzz_jni_api fuzz_jni_threads fuzz_caches fuzz_spec_policy app_fuzz_driver
if [ -n "$RUNTIME" ]; then
    cp "$(find "$ANDROID_NDK_ROOT" -name "$RUNTIME" | head -1)" build/fuzz/app-android-$PROFILE-$SAN/
fi
' > "$OUT/logs/phone-build-$name.log" 2>&1 || die "the phone build failed, see $OUT/logs/phone-build-$name.log"
    local stage="$OUT/phone-$name" b="$REPO/build/fuzz/app-android-$name"
    rm -rf "$stage"
    mkdir -p "$stage/bin" "$stage/lib" "$stage/models"
    cp "$b"/{fuzz_jni_api,fuzz_jni_threads,fuzz_caches,fuzz_spec_policy,app_fuzz_driver} "$stage/bin/"
    cp "$dsp" "$stage/lib/"
    if [[ $san == asan ]]; then
        # The fixed runtime goes in its own directory, which is first in LD_LIBRARY_PATH.
        check_asan_rt22
        mkdir -p "$stage/asan-rt"
        cp "$ASAN_RT22" "$stage/asan-rt/"
    elif [[ -n "$runtime" ]]; then
        cp "$b/$runtime" "$stage/lib/"
    fi
    if [[ -n "$prebuilt" ]]; then
        cp "$jnilibs"/*.so "$stage/lib/"
    fi
    cp "$MODELS"/tiny-qwen35-*.gguf "$stage/models/"
    [[ -f "$UBSAN_SUPP" ]] || die "no $UBSAN_SUPP: the sanitizer-matrix agent writes it"
    cp "$UBSAN_SUPP" "$stage/ubsan.supp"
    echo "run.sh: the phone files are in $stage ($(du -sh "$stage" | cut -f1))"
}

phone_commands() {
    local san=$1
    android_runtime "$san" > /dev/null
    local s="$OUT/phone-$PROFILE-$san" d="/data/local/tmp/qwen/fuzz/app/$PROFILE-$san" a="adb -s 192.168.14.130:5555"
    local rt libs="$d/lib" push="$s/bin $s/lib $s/models $s/ubsan.supp"
    case $san in
        asan)
            rt="ASAN_OPTIONS=halt_on_error=1:allocator_may_return_null=1:detect_leaks=0"
            libs="$d/asan-rt:$d/lib"
            push="$s/asan-rt $push"
            ;;
        hwasan) rt="HWASAN_OPTIONS=halt_on_error=1:allocator_may_return_null=1" ;;
        ubsan) rt="UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1:suppressions=$d/ubsan.supp" ;;
        none) rt="" ;;
    esac
    local envs="LD_LIBRARY_PATH=$libs ADSP_LIBRARY_PATH=$d/lib FUZZ_APP_LIBDIR=$d/lib FUZZ_APP_MODEL_DIR=$d/models FUZZ_APP_WORK=$d/work FUZZ_ARTIFACT_DIR=$d/logs FUZZ_APP_SKIP_KNOWN=1 FAKEJNI_RELAX=pending $rt"
    local real="FUZZ_APP_REAL_MODEL=/data/local/tmp/qwen/models/Qwen3.5-2B-Q8_0.gguf FUZZ_APP_REAL_ONLY=1 FUZZ_APP_MAX_OPS=8 FUZZ_APP_MAX_GEN=16 FUZZ_APP_ORACLE=0"
    local thermal="$a shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
    # One run: the thermal status before and after it, and the processes that stay.
    one() {
        echo "# $1"
        echo "$thermal"
        echo "$a shell \"$2; echo rc=\\\$?\""
        echo "$thermal"
        echo "$a shell pgrep -a -f 'fuzz_|app_fuzz_driver'"
    }
    echo "# The phone build of $PROFILE-$san: $s (run \"tests/fuzz/app/run.sh phone-build $san --profile $PROFILE\" first)."
    echo "$a shell mkdir -p $d/work $d/logs"
    echo "$a push $push $d/"
    if [[ $san == asan ]]; then
        echo "$a shell sha256sum $d/asan-rt/libclang_rt.asan-aarch64-android.so  # must be $ASAN_RT22_SHA256"
    fi
    echo "$a shell chmod 755 $d/bin/fuzz_jni_api $d/bin/fuzz_jni_threads $d/bin/fuzz_caches $d/bin/fuzz_spec_policy $d/bin/app_fuzz_driver"
    echo "$a shell ls -la /data/local/tmp/qwen/models/"
    one "1. The draft length policy, CPU only, 80 s." \
        "cd $d && timeout -s KILL 100 env $envs bin/fuzz_spec_policy -max_total_time=80 -rss_limit_mb=2048 -artifact_prefix=logs/ > logs/spec_policy.log 2>&1"
    one "2. The caches, CPU only, 80 s." \
        "cd $d && timeout -s KILL 100 env $envs bin/fuzz_caches -max_total_time=80 -rss_limit_mb=2048 -artifact_prefix=logs/ > logs/caches.log 2>&1"
    one "3. The JNI API with the tiny model on the CPU backend." \
        "cd $d && timeout -s KILL 100 env $envs bin/app_fuzz_driver --seconds 80 --seed 1000 > logs/driver-cpu-tiny.log 2>&1"
    one "4. The JNI API with the tiny model, most loads on HTP0." \
        "cd $d && timeout -s KILL 100 env $envs FUZZ_APP_DEVICE=HTP0 bin/app_fuzz_driver --seconds 80 --seed 2000 > logs/driver-htp0-tiny.log 2>&1"
    one "5. Short programs with the real 2B Q8_0 model on HTP0." \
        "cd $d && timeout -s KILL 100 env $envs FUZZ_APP_DEVICE=HTP0 $real bin/app_fuzz_driver --seconds 80 --seed 3000 > logs/driver-htp0-real.log 2>&1"
    one "6. Short programs with the real 2B Q8_0 model on the CPU." \
        "cd $d && timeout -s KILL 100 env $envs $real bin/app_fuzz_driver --seconds 80 --seed 4000 > logs/driver-cpu-real.log 2>&1"
    one "7. Stop requests and frees from a second thread, tiny model on HTP0." \
        "cd $d && timeout -s KILL 100 env $envs FUZZ_APP_DEVICE=HTP0 bin/app_fuzz_driver --threads --cross-free --seconds 80 --seed 5000 > logs/driver-threads-htp0.log 2>&1"
    one "8. libFuzzer on the JNI API, tiny model, most loads on HTP0." \
        "cd $d && timeout -s KILL 100 env $envs FUZZ_APP_DEVICE=HTP0 FUZZ_APP_MAX_OPS=24 bin/fuzz_jni_api -max_total_time=80 -rss_limit_mb=3072 -max_len=1024 -artifact_prefix=logs/ > logs/fuzz_jni_api-htp0.log 2>&1"
    local sc
    for sc in image-shape spec-disable sampler-nan priority; do
        one "9. The scenario $sc (its log ends with its report, or with a line that has \"no\")." \
            "cd $d && timeout -s KILL 60 env $envs FAKEJNI_RELAX= bin/app_fuzz_driver --scenario $sc > logs/scenario-$sc.log 2>&1"
    done
    echo "$a pull $d/logs $OUT/phone-logs-$PROFILE-$san"
}

# Parse the options: sets PROFILES, BUDGET and JOBS.
parse_options() {
    PROFILES="debug release"
    BUDGET=600
    JOBS=4
    while (($# > 0)); do
        case $1 in
            --profile)
                PROFILES=${2:?"--profile needs a value: debug or release"}
                [[ $PROFILES == debug || $PROFILES == release ]] || die "--profile must be debug or release, not '$PROFILES'"
                shift 2
                ;;
            --budget-seconds) BUDGET=${2:?"--budget-seconds needs a value"}; shift 2 ;;
            --jobs) JOBS=${2:?"--jobs needs a value"}; shift 2 ;;
            *) die "unknown option '$1'" ;;
        esac
    done
}

mkdir -p "$OUT/logs" "$OUT/work"
mode=${1:-}
shift || true
status=0
case $mode in
    test | fuzz | build)
        san=${1:?"$mode needs a sanitizer: none, asan, ubsan, tsan or msan"}
        shift
        parse_options "$@"
        for PROFILE in $PROFILES; do
            case $mode in
                test) mode_test "$san" || status=1 ;;
                fuzz) mode_fuzz "$san" "$BUDGET" "$JOBS" || status=1 ;;
                build) runtime_options "$san"; build_host "$san" ;;
            esac
        done
        ;;
    cpu-asan | cpu-tsan)
        for PROFILE in debug release; do
            mode_fuzz "${mode#cpu-}" "${1:-600}" 4 || status=1
        done
        ;;
    regress | scenarios)
        for PROFILE in debug release; do
            mode_test asan || status=1
        done
        ;;
    jvm) jvm ;;
    phone-build | phone-commands)
        san=${1:?"$mode needs a sanitizer: none, asan, hwasan or ubsan"}
        shift
        parse_options "$@"
        for PROFILE in $PROFILES; do
            if [[ $mode == phone-build ]]; then phone_build "$san"; else phone_commands "$san"; fi
        done
        ;;
    -h | --help) usage 0 ;;
    *) usage ;;
esac
exit $status
