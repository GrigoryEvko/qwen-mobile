#!/usr/bin/env bash
# Build and run the llama.cpp test suite (ctest) in one profile and one
# sanitizer configuration.
#
# Usage:
#   tests/suite/llama-ctest.sh <none|asan|ubsan|tsan|msan> --profile <debug|release> [options]
#
# Options:
#   --profile P             debug or release (rule R11). Necessary.
#   --phase build|run|all   The preset value is all.
#   --jobs N                The parallel compile jobs. The preset value is 8.
#   --test-jobs N           The parallel ctest tests. The preset value is 4.
#   --timeout-scale F       Multiply each test timeout by F. The preset value is 1.
#
# The script:
#   1. Configures an out-of-tree build of the live submodule
#      third_party/llama.cpp (read only) in
#      build/fuzz/matrix-llama-<profile>-<config>, with
#      tests/sanitizers/profile-<profile>.cmake and <config>.cmake. The build
#      has the CPU backend only, no OpenSSL and no network. A change of the
#      options or of the sanitizer files starts a clean build.
#   2. Builds the executable of each ctest test.
#   3. Writes the tiny model with a real tokenizer (make-vocab-model.py) to
#      build/fuzz/matrix/models/tiny-llama-spm.gguf, if it is not there.
#   4. Runs ctest in three steps. First test-generate-models, which writes
#      the generated models of each architecture to <build>/tests/test-models.
#      Then each other test in parallel, less the tests of
#      tests/suite/llama-exclude.tsv and less test-opt and test-barrier. Then
#      test-opt and test-barrier one at a time: they start their own thread
#      pools. The tests with the label "model" get LLAMACPP_TEST_MODELFILE of
#      the tiny model. Without it, they skip and give a false pass. Each
#      test that reads LLAMA_ARG_THREADS gets 8 threads (SUITE_TEST_THREADS).
#   5. The local-model step runs the tests that need the model download of
#      upstream (test-thread-safety, test-state-restore-fragmented) and each
#      sub-test of test-backend-sampler, with the tiny model.
#   6. Writes one record for each test to
#      build/fuzz/matrix-llama-<profile>-<config>/results.jsonl (the schema
#      of tests/suite/lib.sh).
#
# The sanitizer runtime options come from tests/sanitizers/env.sh.
#
# Requirements: cmake, ninja, clang, ld.lld, jq, yq, rg, python3 with numpy.
# No container. RAM and time: refer to the report of tests/run-suite.sh.
#
# Exit status: 0 if each test that runs passes with no sanitizer report,
# 1 if a test fails or gives a report, 2 if the step cannot run.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
source "$SUITE_REPO_ROOT/tests/sanitizers/env.sh"

readonly LLAMA_SRC="$SUITE_REPO_ROOT/third_party/llama.cpp"
readonly EXCLUDE_FILE="$SUITE_REPO_ROOT/tests/suite/llama-exclude.tsv"
readonly MODEL_DIR="$SUITE_REPO_ROOT/build/fuzz/matrix/models"
readonly TINY_MODEL="$MODEL_DIR/tiny-llama-spm.gguf"
# The sub-tests of test-backend-sampler that need a trained model, with the
# reason. The local-model step records them as excluded.
declare -A SAMPLER_EXCLUDE=(
    [logit_bias]="Needs a trained model: the test asserts that a +10 logit bias wins a dist sample. With random weights the top logits are higher than the biased logit. The upstream fixture downloads stories15M."
)

# The threads of each test that reads LLAMA_ARG_THREADS.
readonly TEST_THREADS="${SUITE_TEST_THREADS:-8}"
# The tests that choose their thread count themselves (test-opt uses
# hardware_concurrency() / 2) or measure the thread pool (test-barrier). They
# run one at a time after the parallel step.
readonly SERIAL_REGEX='^(test-opt|test-barrier)$'
CONFIG=""
PROFILE=""
PHASE="all"
JOBS=8
TEST_JOBS=4
TIMEOUT_SCALE=1

# The llama.cpp options of each configuration. The profile file and the
# sanitizer file add the compiler, the flags, GGML_OPENMP=OFF and GGML_NATIVE.
readonly LLAMA_OPTIONS=(
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_COMMON=ON
    # test-chat links server-context, and the mtmd tests link mtmd. Both
    # come from tools/, thus the tools and the server are ON.
    -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_APP=OFF
    -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_OPENSSL=OFF -DLLAMA_LLGUIDANCE=OFF -DLLAMA_TESTS_INSTALL=OFF
    -DGGML_BACKEND_DL=OFF -DGGML_CCACHE=OFF -DGGML_LLAMAFILE=OFF
    -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_OPENCL=OFF -DGGML_METAL=OFF -DGGML_HEXAGON=OFF
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)

# Print the header comment of this script as the usage text.
print_usage() {
    local line
    while IFS= read -r line; do
        [[ "$line" == "#!"* ]] && continue
        [[ "$line" != "#"* ]] && break
        line="${line#\#}"
        echo "${line# }"
    done < "${BASH_SOURCE[0]}"
}

# Read the command-line options into the global variables.
parse_args() {
    [[ $# -ge 1 ]] || { print_usage >&2; exit 2; }
    case "$1" in -h|--help) print_usage; exit 0 ;; esac
    CONFIG="$1"
    shift
    suite_is_config "$CONFIG" || suite_die "The configuration '$CONFIG' is not known. Use one of: $SUITE_CONFIGS."
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --profile) PROFILE="$2"; shift 2 ;;
            --phase) PHASE="$2"; shift 2 ;;
            --jobs) JOBS="$2"; shift 2 ;;
            --test-jobs) TEST_JOBS="$2"; shift 2 ;;
            --timeout-scale) TIMEOUT_SCALE="$2"; shift 2 ;;
            *) suite_die "The option '$1' is not known. Use --help for the usage." ;;
        esac
    done
    suite_is_profile "$PROFILE" || suite_die "--profile must be debug or release, not '$PROFILE'."
    [[ "$PHASE" =~ ^(build|run|all)$ ]] || suite_die "--phase must be build, run or all, not '$PHASE'."
}

# Print the per-test timeout in seconds of the configuration. The slowest
# test is test-save-load-state (112 generated models). It takes about 3
# minutes with no sanitizer, 8 minutes under ASan and 10 minutes under TSan on
# a 28-core host. Each limit is about four times that, with a margin for a
# host under load.
test_timeout() {
    local base
    case "$CONFIG" in
        none) base=900 ;;
        asan|ubsan) base=1800 ;;
        tsan|msan) base=3600 ;;
    esac
    jq -n --argjson b "$base" --argjson s "$TIMEOUT_SCALE" '($b * $s) | ceil'
}

# Configure the build directory. A different fingerprint starts from an
# empty directory, because a CMake cache keeps the old flags.
configure() {
    local fp stamp="$BUILD/.matrix-fingerprint" init_args
    fp="$(suite_fingerprint "$PROFILE" "$CONFIG" "${LLAMA_OPTIONS[@]}")"
    if [[ -f "$BUILD/CMakeCache.txt" && -f "$stamp" && "$(cat "$stamp")" == "$fp" ]]; then
        suite_log "$PROFILE-$CONFIG: the configure step is current."
        return 0
    fi
    suite_log "$PROFILE-$CONFIG: configure $BUILD from an empty directory."
    rm -rf "$BUILD"
    mkdir -p "$BUILD"
    mapfile -t init_args < <(suite_cmake_init_args "$PROFILE" "$CONFIG")
    cmake -G Ninja -S "$LLAMA_SRC" -B "$BUILD" "${init_args[@]}" \
        "${LLAMA_OPTIONS[@]}" > "$BUILD/configure.log" 2>&1 \
        || { tail -30 "$BUILD/configure.log" >&2; suite_die "$PROFILE-$CONFIG: the configure step failed. Refer to $BUILD/configure.log."; }
    echo "$fp" > "$stamp"
}

# Build the executable of each ctest test that the build directory makes.
# Before the build, ctest --show-only gives no command for a test, because
# the executable does not exist. Thus the target names come from the
# resolved paths in tests/CTestTestfile.cmake. CMake 4.3 writes the test
# name as [=[name]=], CMake 4.4 as "name", thus the pattern takes the two.
build_tests() {
    local targets
    mapfile -t targets < <(rg -o --no-filename -r '$2' \
        "add_test\\((\\[=\\[[^]]+\\]=\\]|\"[^\"]+\") \"${BUILD}/bin/([^\"/]+)\"" \
        "$BUILD/tests/CTestTestfile.cmake" | sort -u)
    [[ ${#targets[@]} -gt 0 ]] || suite_die "$PROFILE-$CONFIG: ctest lists no test. Refer to $BUILD/configure.log."
    suite_log "$PROFILE-$CONFIG: build ${#targets[@]} test executables with $JOBS jobs."
    local t0=$SECONDS
    nice -n 10 cmake --build "$BUILD" -j "$JOBS" --target "${targets[@]}" > "$BUILD/build.log" 2>&1 \
        || { rg -n -e 'error:' -e 'FAILED' "$BUILD/build.log" | head -30 >&2; suite_die "$PROFILE-$CONFIG: the build failed. Refer to $BUILD/build.log."; }
    echo "$((SECONDS - t0))" > "$BUILD/.build-seconds"
    suite_log "$PROFILE-$CONFIG: the build took $((SECONDS - t0)) s."
}

# Write the tiny model with a tokenizer, if it is not there. The model does
# not depend on the configuration, thus all builds share one file.
# Return status: 1 if python3, numpy or gguf-py is missing.
ensure_tiny_model() {
    [[ -f "$TINY_MODEL" ]] && return 0
    mkdir -p "$MODEL_DIR"
    PYTHONPATH="$LLAMA_SRC/gguf-py${PYTHONPATH:+:$PYTHONPATH}" \
        python3 "$SUITE_REPO_ROOT/tests/suite/make-vocab-model.py" --out "$TINY_MODEL" >&2
}

# Print the excluded test names of this configuration, one on each line,
# as "name<TAB>reason".
excluded_tests() {
    local name configs reason
    while IFS=$'\t' read -r name configs reason; do
        [[ -z "$name" || "$name" == \#* ]] && continue
        if [[ "$configs" == "all" || ",$configs," == *",$CONFIG,"* || ",$configs," == *",$PROFILE-$CONFIG,"* ]]; then
            printf '%s\t%s\n' "$name" "$reason"
        fi
    done < "$EXCLUDE_FILE"
}

# Convert a ctest JUnit file to result records.
# Arguments: the JUnit file, the directory for the log of each test.
junit_to_records() {
    local junit="$1" logdir="$2" name status secs out_file findings reason line
    mkdir -p "$logdir"
    # yq converts the XML to JSON. A single testcase is an object, not a
    # list, thus the filter makes a list in each case.
    yq -p xml -o json '.' "$junit" \
        | jq -c '.testsuite.testcase | if type == "array" then .[] else . end
                 | {name: ."+@name", time: (."+@time" // "0" | tonumber),
                    status: (."+@status" // "run"),
                    failure: (.failure // null | if . == null then null else (."+@message" // "failed") end),
                    out: (."system-out" // "")}' \
        > "$logdir/.cases.jsonl"
    while IFS= read -r line; do
        name="$(jq -r '.name' <<< "$line")"
        secs="$(jq -r '.time' <<< "$line")"
        out_file="$logdir/$name.log"
        jq -r '.out' <<< "$line" > "$out_file"
        findings="$(suite_count_findings "$out_file")"
        reason=""
        case "$(jq -r '.status' <<< "$line")" in
            run) status="pass" ;;
            fail)
                status="fail"
                reason="$(jq -r '.failure // "failed"' <<< "$line")"
                [[ "$reason" == *Timeout* ]] && status="timeout"
                ;;
            disabled|notrun) status="excluded"; reason="ctest did not run the test" ;;
            *) status="fail"; reason="ctest status $(jq -r '.status' <<< "$line")" ;;
        esac
        if [[ "$status" == "pass" && "$findings" -gt 0 ]]; then
            status="fail"
            reason="the test passed, but its output has $findings sanitizer report(s)"
        fi
        suite_record "$RESULTS" llama-ctest "$name" "$PROFILE" "$CONFIG" test "$secs" 1 "$findings" \
            "$status" "$reason" "$out_file"
    done < "$logdir/.cases.jsonl"
}

# Run one ctest invocation and convert its JUnit file to records.
# Arguments: a name for the files, then the ctest arguments.
run_ctest() {
    local tag="$1" rc=0
    shift
    local junit="$OUT/junit-$tag.xml"
    rm -f "$junit"
    ctest --test-dir "$BUILD" --timeout "$(test_timeout)" \
        --output-junit "$junit" --output-log "$OUT/ctest-$tag.log" \
        --test-output-size-passed 1048576 --test-output-size-failed 16777216 \
        --test-output-truncation middle --no-tests=error "$@" \
        > "$OUT/ctest-$tag.stdout" 2>&1 || rc=$?
    if [[ ! -s "$junit" ]]; then
        tail -20 "$OUT/ctest-$tag.stdout" >&2
        suite_die "$PROFILE-$CONFIG: ctest ($tag) wrote no JUnit file. Refer to $OUT/ctest-$tag.stdout."
    fi
    junit_to_records "$junit" "$OUT/logs"
    return $rc
}

# Print the names of the sub-tests of test-backend-sampler, from the table
# BACKEND_TESTS of its source. A new upstream sub-test thus runs with no
# change here.
sampler_subtests() {
    rg -o -r '$1' '^\s*\{\s*"([a-z0-9_]+)",\s*test_backend_' "$LLAMA_SRC/tests/test-backend-sampler.cpp"
}

# The local-model step: the tests that need a model with a tokenizer.
# Return status: 1 if one of them fails.
run_local_model_tests() {
    local failed=0 limit sub area=llama-ctest
    limit="$(test_timeout)"
    if ! ensure_tiny_model; then
        suite_record "$RESULTS" "$area" "local-model-step" "$PROFILE" "$CONFIG" test 0 0 0 missing-prerequisite \
            "python3 with numpy is necessary to write $TINY_MODEL (pip install numpy)" ""
        return 1
    fi
    suite_run_record "$RESULTS" "$area" "test-thread-safety.local-model" "$PROFILE" "$CONFIG" "$limit" \
        "$OUT/logs/test-thread-safety.local-model.log" \
        "$BUILD/bin/test-thread-safety" -m "$TINY_MODEL" -ngl 99 \
        -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2 || failed=1
    suite_run_record "$RESULTS" "$area" "test-state-restore-fragmented.local-model" "$PROFILE" "$CONFIG" "$limit" \
        "$OUT/logs/test-state-restore-fragmented.local-model.log" \
        "$BUILD/bin/test-state-restore-fragmented" -m "$TINY_MODEL" || failed=1
    # tsan and msan: test-save-load-state with 8 of the 112 generated
    # architectures (llama-exclude.tsv gives the reason). The set has the
    # architectures of the app (qwen35 dense and MoE) and one of each kind
    # of memory: attention, recurrent (mamba2), hybrid (nemotron_h), MLA
    # (deepseek2) and sliding window (gemma2).
    if [[ "$CONFIG" == tsan || "$CONFIG" == msan ]]; then
        local subset="$OUT/models-subset" m
        mkdir -p "$subset"
        for m in qwen35-dense qwen35moe-moe qwen3-dense llama-dense mamba2-dense nemotron_h-dense \
                 gemma2-dense deepseek2-moe; do
            ln -sf "$BUILD/tests/test-models/$m.gguf" "$subset/$m.gguf"
        done
        suite_run_record "$RESULTS" "$area" "test-save-load-state.subset" "$PROFILE" "$CONFIG" "$limit" \
            "$OUT/logs/test-save-load-state.subset.log" \
            "$BUILD/bin/test-save-load-state" --models "$subset" || failed=1
    fi
    while IFS= read -r sub; do
        if [[ -n "${SAMPLER_EXCLUDE[$sub]:-}" ]]; then
            suite_record "$RESULTS" "$area" "test-backend-sampler.$sub" "$PROFILE" "$CONFIG" test 0 0 0 excluded \
                "${SAMPLER_EXCLUDE[$sub]}" ""
            continue
        fi
        suite_run_record "$RESULTS" "$area" "test-backend-sampler.$sub" "$PROFILE" "$CONFIG" "$limit" \
            "$OUT/logs/test-backend-sampler.$sub.log" \
            "$BUILD/bin/test-backend-sampler" --model "$TINY_MODEL" --test "$sub" || failed=1
    done < <(sampler_subtests)
    return $failed
}

# Run the whole test phase and write results.jsonl.
run_tests() {
    [[ -f "$BUILD/CMakeCache.txt" ]] || suite_die "$PROFILE-$CONFIG: $BUILD is not configured. Run the build phase first."
    suite_require ctest jq yq rg python3
    OUT="$BUILD/suite"
    RESULTS="$BUILD/results.jsonl"
    rm -rf "$OUT"
    rm -f "$RESULTS"
    mkdir -p "$OUT/logs"
    sanitizer_env "$CONFIG"
    # The tests that take their parameters from common_params use
    # cpu_get_num_math() threads, which is the count of the physical cores
    # (192 on a 2-socket build server). ggml then spins at each barrier with
    # that many threads, and under a loaded host a test that takes seconds
    # on 8 threads runs for more than 15 minutes. A fixed count also makes
    # the runs the same on each host.
    export LLAMA_ARG_THREADS="$TEST_THREADS" LLAMA_ARG_THREADS_BATCH="$TEST_THREADS"

    local exclude_names=() exclude_regex failed=0 name reason
    while IFS=$'\t' read -r name reason; do
        exclude_names+=("$name")
        suite_record "$RESULTS" llama-ctest "$name" "$PROFILE" "$CONFIG" test 0 0 0 excluded "$reason" ""
    done < <(excluded_tests)
    if ! python3 -c 'import jinja2' 2> /dev/null; then
        exclude_names+=("test-jinja-py")
        suite_record "$RESULTS" llama-ctest test-jinja-py "$PROFILE" "$CONFIG" test 0 0 0 missing-prerequisite \
            "python3 with the package jinja2 is necessary (pip install jinja2)" ""
        suite_log "$PROFILE-$CONFIG: python3 has no jinja2. test-jinja-py does not run."
    fi
    exclude_regex="^($(IFS='|'; echo "${exclude_names[*]}"))\$"

    ensure_tiny_model || suite_log "$PROFILE-$CONFIG: WARNING: no tiny model. The model tests fail."
    suite_log "$PROFILE-$CONFIG: ctest step 1, test-generate-models."
    run_ctest setup -R '^test-generate-models$' || failed=1

    suite_log "$PROFILE-$CONFIG: ctest step 2, the other tests, $TEST_JOBS in parallel, timeout $(test_timeout) s each."
    LLAMACPP_TEST_MODELFILE="$TINY_MODEL" run_ctest main \
        -E "$exclude_regex|^test-generate-models\$|$SERIAL_REGEX" -FS generate-models -j "$TEST_JOBS" || failed=1

    suite_log "$PROFILE-$CONFIG: ctest step 3, the tests that start their own thread pools, one at a time."
    LLAMACPP_TEST_MODELFILE="$TINY_MODEL" run_ctest serial \
        -R "$SERIAL_REGEX" -E "$exclude_regex" -FS generate-models -j 1 || failed=1

    suite_log "$PROFILE-$CONFIG: the local-model step."
    run_local_model_tests || failed=1

    local pass fail total
    pass="$(jq -s '[.[] | select(.status == "pass")] | length' "$RESULTS")"
    fail="$(jq -s '[.[] | select(.status == "fail" or .status == "timeout")] | length' "$RESULTS")"
    total="$(jq -s 'length' "$RESULTS")"
    suite_log "$PROFILE-$CONFIG: $pass pass, $fail fail or timeout, $total records. Refer to $RESULTS."
    [[ "$fail" -eq 0 && $failed -eq 0 ]]
}

main() {
    parse_args "$@"
    suite_require cmake ninja clang clang++ ld.lld jq rg
    # A ccache in PATH (for example /usr/lib64/ccache/clang) must not keep
    # the sanitizer objects: ten builds can push the other builds of the
    # user out of the cache.
    export CCACHE_DISABLE=1
    BUILD="$SUITE_REPO_ROOT/build/fuzz/matrix-llama-$PROFILE-$CONFIG"
    if [[ "$PHASE" == "build" || "$PHASE" == "all" ]]; then
        configure
        build_tests
    fi
    if [[ "$PHASE" == "run" || "$PHASE" == "all" ]]; then
        run_tests || exit 1
    fi
}

main "$@"
