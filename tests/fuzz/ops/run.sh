#!/usr/bin/env bash
# The op fuzzer of the Qwen3.5 path. It builds one profile and one configuration (one sanitizer) at
# a time, runs the test suite (the seeds and the regression inputs, once) or the fuzz suite
# (libFuzzer) on the CPU, prepares the phone runs on the CPU and on HTP0, and compares the phone
# results with the oracle. The usage text gives the modes.

set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly HERE
# shellcheck source=../../../scripts/lib.sh
source "$HERE/../../../scripts/lib.sh"

readonly AREA=ops
readonly SNAP="$REPO_ROOT/build/fuzz/ops-src"
readonly SRC="$SNAP/ggml"
readonly MISC="$REPO_ROOT/build/fuzz/ops"          # the oracle, fixes, phone stage, temporary files
readonly B_ORACLE_X86="$REPO_ROOT/build/oracle-x86"   # the naive llama.cpp build, read by other areas
readonly SHARED_SAN="$REPO_ROOT/tests/sanitizers"
readonly SHIPPED_LIBS="$REPO_ROOT/android/snapdragon/jniLibs/arm64-v8a"
readonly SYMBOLIZER="$REPO_ROOT/build/fuzz/android-symbolizer/llvm-symbolizer"   # of LLVM 22.1.8
readonly ASAN_RUNTIME="$REPO_ROOT/build/fuzz/asan-android-runtime/libclang_rt.asan-aarch64-android.so"   # of compiler-rt 22.1.8
# FUZZ_OPS_SRC: a different ggml tree for the host builds, for example the private copy of a fix
# (never the shared submodule). FUZZ_OPS_TAG: the suffix of the build directories of that tree.
readonly HOST_SRC=${FUZZ_OPS_SRC:-$SRC}
readonly TAG=${FUZZ_OPS_TAG:-}
if [[ $HOST_SRC != "$SRC" && -z $TAG ]]; then
    echo "fuzz-ops: FUZZ_OPS_SRC needs FUZZ_OPS_TAG, thus the builds of the tree stay apart" >&2
    exit 2
fi
# The oracle of a private tree comes from that tree, thus the harness and the oracle run the same
# ggml code.
readonly B_ORACLE="$MISC/oracle${TAG:+-$TAG}"
readonly ALL_GROUPS="matmul gdn attn norm elem data"
readonly HOST_CONFIGS="none asan ubsan tsan msan"
readonly PHONE_CONFIGS_ALL="none asan hwasan ubsan"
readonly ALL_PROFILES="debug release"

BUDGET=${FUZZ_BUDGET:-600}
JOBS=${FUZZ_JOBS:-4}
BUILD_JOBS=${BUILD_JOBS:-8}
RSS_MB=${RSS_MB:-4096}
PHONE=${PHONE:-192.168.14.130:5555}
PHONE_DIR=${PHONE_DIR:-/data/local/tmp/qwen/fuzz/ops}
PHONE_BUILDS=${PHONE_BUILDS:-}
DSP_LIB=${DSP_LIB:-}
ADSP_DIR=${ADSP_DIR:-/data/local/tmp/qwen/q8ref/lib}
PACK_N=${PACK_N:-20}
PACK_CORPUS=${PACK_CORPUS:-40}
PHONE_SECONDS=${PHONE_SECONDS:-85}
DIAG_BUILD=${DIAG_BUILD:-release-none}

# Print the usage text.
usage() {
    cat << 'EOF'
usage: tests/fuzz/ops/run.sh test CONFIG [--profile debug|release] [--budget-seconds N] [--jobs N]
       tests/fuzz/ops/run.sh fuzz CONFIG [--profile debug|release] [--budget-seconds N] [--jobs N] [GROUP...]
       tests/fuzz/ops/run.sh MODE [ARGS]

CONFIG is one sanitizer: none, asan, ubsan, tsan or msan. A build never mixes two sanitizers.
--profile is debug (-O1, assert on, no LTO) or release (the shipped flags: -O3 -DNDEBUG -flto
-ffp-model=fast); without it the suite runs both, debug first. The build directory is
build/fuzz/ops-PROFILE-CONFIG, and each run adds one JSON line for each target (each group of
ops) to build/fuzz/ops-PROFILE-CONFIG/results.jsonl:
  {area, target, sanitizer, profile, mode, seconds, executions, findings, crash_files}

  test CONFIG   Run each seed input and each regression input once, with no mutation. A finding
                gives a nonzero exit code: a sanitizer report, a crash or an assert, a write
                outside a tensor or into an input, a decode mismatch between the harness and the
                oracle, or a result above the loose bound for an input without special values.
                The suite also runs ops_f16_check (target f16_check): the f32-to-f16 conversions
                of ggml with the flags of the profile, against a table and the F16C hardware.
  fuzz CONFIG  Run libFuzzer on each group for --budget-seconds (default 600), --jobs groups at
                a time (default 4). The default is every group. The inputs include the special
                values (Inf, NaN, subnormal, huge) and fully masked rows in the two profiles: the
                value asserts of the ggml CPU ops run only with GGML_CPU_VALUE_ASSERTS.

Other modes:
  phone-build [PROFILE-CONFIG...]  Build ops_replay for arm64 Android: one build for each profile
                (debug, release) and each configuration (none, asan, hwasan, ubsan); the default is
                all eight. The release none build links the shipped libraries of
                android/snapdragon/jniLibs/arm64-v8a (hash-checked against build/hashes-native.txt).
                Then make the case pack, and stage the phone files in build/fuzz/ops/phone.
  phone-commands [probe|full|diag]   Print the adb commands. "diag" runs the diagnosis pack of
                the HTP0 guard writes and run-to-run differences with repeats on HTP0. "probe" runs
                the first 3 cases of each
                build and prints every UBSan report (the evidence run). "full" runs the pack with
                the builds of PHONE_BUILDS (default: all staged builds). This script never runs adb.
  compare [RESULT...]      Compare the pulled phone results with the oracle, print the per-op
                error table, and write the findings to build/fuzz/ops/findings-phone. The
                column nf_values counts the non-finite mismatches of all runs, and nf_f16
                counts the part of them with a finite reference above 65504 (the f16 range).
  summary PROFILE CONFIG   Print the merged statistics of the fuzz runs of one build.
  minimize FILE GROUP KIND VERDICT [PROFILE CONFIG]
                           Minimize a numeric finding with libFuzzer, and write the result to
                           tests/fuzz/ops/regress/KIND/. VERDICT is above-strict, above-loose or nonfinite.
  snapshot                 Refresh the private copy of ggml (build/fuzz/ops-src/ggml): the llama.cpp
                           commit of HEAD plus patches/series, made by tests/sanitizers/llama-copy.sh.
                           Each run that builds from the copy refreshes it first.
  symbolizer               Build llvm-symbolizer for arm64 Android with
                           tests/sanitizers/build-android-symbolizer.sh (phone-build does it when it is
                           missing). Without it, a sanitizer report on the phone has no function
                           names, and the function-level entries of tests/sanitizers/ubsan.supp do not
                           match.
  asan-runtime             Build the ASan runtime of the phone (compiler-rt 22.1.8) with
                           tests/sanitizers/build-asan-android-runtime.sh, and check it.
  oracle-x86               Build the naive x86 oracle of the full llama.cpp tree in build/oracle-x86:
                           gcc, Release, no SIMD option, -ffp-contract=off -fno-fast-math. Other
                           areas read its llama-perplexity (the KL bases).
  phone-libs               With FUZZ_OPS_SRC and FUZZ_OPS_TAG: build the app libraries and llama-bench
                           for arm64 Android from the llama.cpp tree above FUZZ_OPS_SRC, with the
                           recipe of scripts/build-native.sh, into build/fuzz/ops/phone/libs-TAG.
                           The libraries are byte-identical to a build of scripts/build-native.sh
                           only with three conditions: (1) the same source path
                           (/workspace/third_party/llama.cpp), (2) the same build path
                           (/workspace/build/native/llama), and (3) a git tree at the pinned commit
                           with the series (the libraries hold the ggml commit, "c6824a9-dirty").
                           FUZZ_OPS_REPRO_PATHS=1 gives (1) and (2) with bind mounts in the container
                           only; a "git clone --shared" of the submodule gives (3).
  phone-check-commands     With FUZZ_OPS_TAG: print the phone check of a fix before it lands:
                           the op pack and llama-bench, the shipped libraries against libs-TAG.
  bounds                   Print the bound rule of each kind.
  help                     Print this text.
  Aliases: cpu-asan = fuzz asan, cpu-tsan = fuzz tsan, regress = test asan.

Groups (the targets): matmul gdn attn norm elem data

Flags and runtime options: the shared files tests/sanitizers/profile-<profile>.cmake,
tests/sanitizers/<config>.cmake and tests/sanitizers/env.sh. Suppressions: only the shared files
tests/sanitizers/<config>.supp. The area has no entry there: the fixes of its UBSan reports (the
null pointer of ggml_graph_nbytes, the casts of the type traits) are patches/fuzz-ops/0001 and 0002.

Special cases: a case is special (refer to src/case.h) when an input holds Inf, NaN, a subnormal
or a huge value, or when a conversion of the op makes an input infinite. The known properties of
the CPU backend below are not defects, and their cases are special:
  - A matmul activation past the f16 range (an F16 weight) or past 127 x 65520 (the Q8_0 scale of
    a Q8_0 or Q4_0 weight) gives inf or NaN, and which one depends on the order of the operations.
  - FLASH_ATTN_EXT with an F16 V (P3): the one-row path of the CPU adds the V rows in an f16
    accumulator, which overflows past 65504. The tiled path adds them in f32.
  - ROPE with a freq factor that makes an angle past 2^64: sin and cos are not defined by the inputs.

Environment (defaults in parentheses):
  FUZZ_BUDGET, FUZZ_JOBS  the defaults of --budget-seconds and --jobs (600, 4)
  BUILD_JOBS     the parallel build jobs (8)
  RSS_MB         the memory limit of a fuzzer process in MB (4096)
  PHONE          the adb serial (192.168.14.130:5555)
  PHONE_DIR      the work directory on the phone (/data/local/tmp/qwen/fuzz/ops)
  PHONE_BUILDS   the builds of phone-commands full, for example "release-none debug-asan"
  DSP_LIB        a DSP library to push as libggml-htp-v79.so; empty uses ADSP_DIR as it is
  ADSP_DIR       the ADSP_LIBRARY_PATH on the phone without DSP_LIB (/data/local/tmp/qwen/q8ref/lib)
  PACK_N         the random cases of each kind in the phone pack (20)
  PACK_CORPUS    the most inputs from each CPU corpus in the phone pack (40). The pack takes the
                 corpora of the fuzz suites of the none builds (debug and release) only
  PHONE_SECONDS  the seconds of each phone command before its deadline (85)
  FUZZ_OPS_SRC, FUZZ_OPS_TAG  a different ggml tree for the host builds and the oracle (the private
                 copy of a fix), and the suffix of their build directories:
                 build/fuzz/ops-PROFILE-CONFIG-TAG and build/fuzz/ops/oracle-TAG
  FUZZ_OPS_UBSAN_SUPP  a UBSan suppression file in place of tests/sanitizers/ubsan.supp (the check
                 of a fix without the entries of its task)
EOF
}

# Refresh the private copy of ggml with the shared helper tests/sanitizers/llama-copy.sh: the
# llama.cpp commit of HEAD plus patches/series, without the file times of the source (refer to that
# script). Then record the date and hashes. A lock keeps two runs of this script from one copy at
# the same time.
take_snapshot() {
    mkdir -p "$SNAP"
    (
        flock 8
        "$REPO_ROOT/tests/sanitizers/llama-copy.sh" --ggml "$SNAP" > /dev/null \
            || die "tests/sanitizers/llama-copy.sh --ggml $SNAP failed"
        date > "$SNAP/SNAPSHOT-DATE"
        sha256sum "$SRC/src/ggml-hexagon/ggml-hexagon.cpp" "$SRC/src/ggml-hexagon/htp/htp-ops.h" \
            "$SRC/src/ggml-hexagon/htp-opnode.h" > "$SNAP/SNAPSHOT-HASHES"
    ) 8> "$SNAP.lock"
    echo "fuzz-ops: the snapshot in $SNAP is the tree of HEAD"
}

# Refresh the snapshot one time for each run of this script, before the first build from it.
SNAPSHOT_DONE=0
need_snapshot() {
    [[ $SNAPSHOT_DONE == 1 ]] && return 0
    take_snapshot
    SNAPSHOT_DONE=1
}

# Stop when the argument is not a host configuration.
check_host_config() {
    [[ " $HOST_CONFIGS " == *" $1 "* ]] || die "the configuration '$1' is not one of: $HOST_CONFIGS"
}

# Stop when the argument is not a profile.
check_profile() {
    [[ " $ALL_PROFILES " == *" $1 "* ]] || die "the profile '$1' is not one of: $ALL_PROFILES"
}

# Build the oracle with gcc, like build/oracle-x86. It is the reference of every profile and every
# configuration, and it is not a configuration.
build_oracle() {
    [[ $HOST_SRC == "$SRC" ]] && need_snapshot
    mkdir -p "$B_ORACLE"
    if [[ ! -f "$B_ORACLE/build.ninja" ]]; then
        CC=gcc CXX=g++ nice -n 10 cmake -S "$HERE" -B "$B_ORACLE" -G Ninja -DCMAKE_BUILD_TYPE=Release \
            -DFUZZ_OPS_VARIANT=oracle -DFUZZ_SANITIZER=none -DFUZZ_OPS_GGML_SRC="$HOST_SRC" > "$B_ORACLE/cmake.log" 2>&1 \
            || die "the configure of the oracle failed, refer to $B_ORACLE/cmake.log"
    fi
    nice -n 10 cmake --build "$B_ORACLE" -j"$BUILD_JOBS" --target ops_oracle > "$B_ORACLE/build.log" 2>&1 \
        || die "the build of the oracle failed, refer to $B_ORACLE/build.log"
}

# Build the naive x86 oracle of the full llama.cpp tree in build/oracle-x86 (llama-perplexity and the
# other tools): gcc, Release (-O3 -DNDEBUG), GGML_NATIVE and each x86 SIMD option off,
# -ffp-contract=off -fno-fast-math. OpenMP and llamafile stay on (the llama.cpp defaults). Thus the
# CPU backend uses its scalar paths, for example quantize_row_q8_0_ref and the scalar Q8_0 dot. The
# source is the submodule with the series. BUILD-INFO records the commit and the hash of its diff,
# because another session can change the tree.
build_oracle_x86() {
    local -a opts=(-DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
        -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF
        "-DCMAKE_C_FLAGS=-ffp-contract=off -fno-fast-math" "-DCMAKE_CXX_FLAGS=-ffp-contract=off -fno-fast-math")
    local opt
    for opt in GGML_NATIVE GGML_AVX GGML_AVX2 GGML_AVX512 GGML_AVX512_VBMI GGML_AVX512_VNNI GGML_AVX512_BF16 \
               GGML_AVX_VNNI GGML_FMA GGML_F16C GGML_SSE42 GGML_BMI2 GGML_AMX_TILE GGML_AMX_INT8 GGML_AMX_BF16; do
        opts+=("-D$opt=OFF")
    done
    mkdir -p "$B_ORACLE_X86"
    CC=gcc CXX=g++ nice -n 10 cmake -S "$LLAMA_SUBMODULE" -B "$B_ORACLE_X86" -G Ninja "${opts[@]}" \
        > "$B_ORACLE_X86/cmake.log" 2>&1 \
        || die "the configure of the x86 oracle failed, refer to $B_ORACLE_X86/cmake.log"
    nice -n 10 cmake --build "$B_ORACLE_X86" -j"$BUILD_JOBS" > "$B_ORACLE_X86/build.log" 2>&1 \
        || die "the build of the x86 oracle failed, refer to $B_ORACLE_X86/build.log"
    {
        echo "date: $(date -Iseconds)"
        echo "llama.cpp commit: $(git -C "$LLAMA_SUBMODULE" rev-parse HEAD)"
        echo "sha256 of the diff of the tree: $(git -C "$LLAMA_SUBMODULE" diff HEAD | sha256sum | cut -d' ' -f1)"
        echo "compiler: $(gcc --version | head -n 1)"
        echo "options: ${opts[*]}"
    } > "$B_ORACLE_X86/BUILD-INFO"
    echo "fuzz-ops: the x86 oracle is in $B_ORACLE_X86/bin (refer to $B_ORACLE_X86/BUILD-INFO)"
}

# Configure (once) and build the host fuzzer of one profile and one configuration. The flags come
# from the shared initial caches.
build_config() {
    local profile=$1 config=$2 dir
    dir=$(host_dir "$1" "$2")
    [[ $HOST_SRC == "$SRC" ]] && need_snapshot
    [[ -d "$HOST_SRC/src" ]] || die "FUZZ_OPS_SRC=$HOST_SRC is not a ggml tree"
    [[ -f "$SHARED_SAN/profile-$profile.cmake" ]] || die "the shared file $SHARED_SAN/profile-$profile.cmake is missing"
    [[ -f "$SHARED_SAN/$config.cmake" ]] || die "the shared file $SHARED_SAN/$config.cmake is missing"
    mkdir -p "$dir"
    if [[ ! -f "$dir/build.ninja" ]]; then
        nice -n 10 cmake -S "$HERE" -B "$dir" -G Ninja -DFUZZ_OPS_VARIANT=fuzz -DFUZZ_OPS_GGML_SRC="$HOST_SRC" \
            -C "$SHARED_SAN/profile-$profile.cmake" -C "$SHARED_SAN/$config.cmake" > "$dir/cmake.log" 2>&1 \
            || die "the configure of $dir failed, refer to $dir/cmake.log"
    fi
    nice -n 10 cmake --build "$dir" -j"$BUILD_JOBS" --target fuzz_ops ops_replay ops_f16_check > "$dir/build.log" 2>&1 \
        || die "the build of $dir failed, refer to $dir/build.log"
}

# Run a command with the runtime options of one configuration, from the shared file
# tests/sanitizers/env.sh (the same options in each area). The subshell keeps the
# options away from the other runs.
with_san() {
    local config=$1
    shift
    (
        # shellcheck source=../../sanitizers/env.sh
        source "$SHARED_SAN/env.sh"
        sanitizer_env "$config" || exit 2
        # FUZZ_OPS_UBSAN_SUPP replaces the shared UBSan suppression file for the check of a fix: the
        # file without the entries of a defect shows that its fix removes the reports before the
        # landing removes the entries from the shared file.
        if [[ $config == ubsan && -n ${FUZZ_OPS_UBSAN_SUPP:-} ]]; then
            UBSAN_OPTIONS="${UBSAN_OPTIONS%%:suppressions=*}:suppressions=$FUZZ_OPS_UBSAN_SUPP"
        fi
        "$@"
    )
}

# The host build directory of a profile and a configuration. A private tree (FUZZ_OPS_SRC) adds
# its tag (FUZZ_OPS_TAG) to the name, thus its builds and results stay apart.
host_dir() {
    echo "$REPO_ROOT/build/fuzz/ops-$1-$2${TAG:+-$TAG}"
}

# The kinds of a group.
group_kinds_of() {
    case $1 in
        matmul) echo "mul_mat mul_mat_add mul_mat_multi mul_mat_id" ;;
        gdn)    echo "gated_delta_net gdn_state_chain gdn_conv_chain ssm_conv" ;;
        attn)   echo "rope flash_attn_ext soft_max" ;;
        norm)   echo "rms_norm l2_norm" ;;
        elem)   echo "binary scale unary swiglu gate_chain cumsum" ;;
        data)   echo "cpy set_rows get_rows concat" ;;
    esac
}

# Add one JSON line to the results file of a build. The arguments: profile, config, target, mode,
# seconds, executions, findings, then the crash files.
json_line() {
    local profile=$1 config=$2 target=$3 mode=$4 seconds=$5 executions=$6 findings=$7
    shift 7
    jq -cn --arg area "$AREA" --arg target "$target" --arg sanitizer "$config" --arg profile "$profile" \
        --arg mode "$mode" --argjson seconds "$seconds" --argjson executions "$executions" \
        --argjson findings "$findings" \
        '{area: $area, target: $target, sanitizer: $sanitizer, profile: $profile, mode: $mode, seconds: $seconds,
          executions: $executions, findings: $findings, crash_files: $ARGS.positional}' --args "$@" \
        >> "$(host_dir "$profile" "$config")/results.jsonl"
}

# The test suite of one group: each seed and each regression input of its kinds, once. Print the
# number of inputs and the number of failed inputs; the failed inputs go to the file of the last
# argument.
test_group() {
    local profile=$1 config=$2 g=$3 failed_list=$4
    local dir
    dir=$(host_dir "$profile" "$config")
    local w="$dir/test/$g"
    rm -rf "$w"
    mkdir -p "$w"
    local n=0 bad=0 f k force item
    local -a inputs=()
    for f in "$HERE/corpus/$g"/*; do
        [[ -f $f ]] && inputs+=("$g|$f")
    done
    for k in $(group_kinds_of "$g"); do
        for f in "$HERE/regress/$k"/*.bin; do
            [[ -f $f ]] && inputs+=("$k|$f")
        done
    done
    mkdir -p "$w/artifacts"
    for item in "${inputs[@]}"; do
        force=${item%%|*}
        f=${item#*|}
        n=$((n + 1))
        # A regression input with the prefix "tame-" decodes with FUZZ_OPS_TAME=1 (refer to
        # src/case.cpp): its case exists only with that decode.
        local tame=0
        [[ $(basename "$f") == tame-* ]] && tame=1
        if ! with_san "$config" env FUZZ_OPS_ORACLE="$B_ORACLE/ops_oracle" FUZZ_OPS_GROUP="$force" \
            FUZZ_OPS_FINDINGS="$w/findings" FUZZ_OPS_ABORT="verdict=above-loose,special=0" FUZZ_OPS_TAME=$tame \
            FUZZ_ARTIFACT_DIR="$w/artifacts" \
            timeout -s KILL 300 "$dir/fuzz_ops" -rss_limit_mb="$RSS_MB" -timeout=120 \
            -artifact_prefix="$w/artifacts/" "$f" >> "$w/log.txt" 2>&1; then
            bad=$((bad + 1))
            echo "$f" >> "$failed_list"
            echo "fuzz-ops: test $profile-$config $g: FINDING on $f (log $w/log.txt)" >&2
        fi
    done
    echo "$n $bad"
}

run_test() {
    local profile=$1 config=$2
    build_oracle
    build_config "$profile" "$config"
    local total_bad=0 g dir t0 t1 fbad=0
    dir=$(host_dir "$profile" "$config")
    mkdir -p "$dir/test"
    # The f32-to-f16 conversions of ggml with the flags of the profile (src/f16_check.cpp).
    t0=$(date +%s)
    if ! with_san "$config" timeout -s KILL 300 "$dir/ops_f16_check" > "$dir/test/f16_check.log" 2>&1; then
        fbad=1
        echo "fuzz-ops: test $profile-$config f16_check: FINDING (log $dir/test/f16_check.log)" >&2
    fi
    t1=$(date +%s)
    json_line "$profile" "$config" f16_check test $((t1 - t0)) 1 "$fbad"
    echo "fuzz-ops: test $profile-$config f16_check: 1 check, $fbad findings"
    total_bad=$((total_bad + fbad))
    for g in $ALL_GROUPS; do
        local t0 t1 res n bad failed
        failed="$(host_dir "$profile" "$config")/test/$g.failed"
        mkdir -p "$(dirname "$failed")"
        : > "$failed"
        t0=$(date +%s)
        res=$(test_group "$profile" "$config" "$g" "$failed")
        t1=$(date +%s)
        n=${res% *}
        bad=${res#* }
        local -a crash_files=()
        mapfile -t crash_files < "$failed"
        json_line "$profile" "$config" "$g" test $((t1 - t0)) "$n" "$bad" "${crash_files[@]}"
        echo "fuzz-ops: test $profile-$config $g: $n inputs, $bad findings"
        total_bad=$((total_bad + bad))
    done
    if [[ $total_bad -ne 0 ]]; then
        echo "fuzz-ops: test $profile-$config: $total_bad findings" >&2
        return 1
    fi
    echo "fuzz-ops: test $profile-$config: no finding"
}

# Run one group with libFuzzer for the budget. libFuzzer stops at a crash, thus a loop starts it
# again with the rest of the time. (The fork mode of the libFuzzer of clang 22 reads a null
# pointer in GlobalEnv::secondsSinceProcessStartUp at the start, thus it is not used.) Each
# process writes its own statistics file.
fuzz_group() {
    local profile=$1 config=$2 g=$3 budget=$4
    local dir
    dir=$(host_dir "$profile" "$config")
    local w="$dir/work/$g" f="$dir/findings/$g" log="$dir/logs/fuzz-$g.log"
    mkdir -p "$w/corpus" "$w/artifacts" "$f" "$dir/logs"
    rm -f "$f"/stats-*.tsv
    : > "$log"
    local marker="$w/.start"
    : > "$marker"
    echo "fuzz-ops: fuzz $profile-$config $g for $budget s, log $log"
    local t0 end round=0 left r0 last="" summary_line
    t0=$(date +%s)
    end=$((t0 + budget))
    while left=$((end - $(date +%s))); [[ $left -gt 5 && $round -lt 100 ]]; do
        echo "fuzz-ops: round $round, $left s left" >> "$log"
        r0=$(date +%s)
        # The outer timeout stops a process that hangs (for example the TSan report path under
        # libFuzzer can deadlock). libFuzzer stops by itself within -timeout after -max_total_time,
        # thus a kill by the outer timeout is a hang, and the hang is a finding.
        local rc=0
        with_san "$config" env FUZZ_OPS_ORACLE="$B_ORACLE/ops_oracle" FUZZ_OPS_GROUP="$g" FUZZ_OPS_FINDINGS="$f" \
            FUZZ_ARTIFACT_DIR="$w/artifacts" \
            timeout -s KILL $((left + 180)) nice -n 10 "$dir/fuzz_ops" -max_total_time="$left" -rss_limit_mb="$RSS_MB" \
            -max_len=512 -timeout=120 -artifact_prefix="$w/artifacts/" -print_final_stats=1 \
            "$w/corpus" "$HERE/corpus/$g" >> "$log" 2>&1 || rc=$?
        if [[ $rc -eq 137 ]]; then
            {
                echo "hang: the outer timeout stopped round $round of $profile-$config $g after $(($(date +%s) - r0)) s"
                tail -n 40 "$log"
            } > "$w/artifacts/hang-round-$round.txt"
            echo "fuzz-ops: HANG in round $round of $profile-$config $g (refer to $w/artifacts/hang-round-$round.txt)" | tee -a "$log"
        fi
        round=$((round + 1))
        # A report that ends two short rounds in a row stops the rest of every round too: the
        # defect blocks this group, thus the run stops and reports it.
        summary_line=$(rg '^SUMMARY: ' "$log" | tail -n 1 || true)
        if [[ -n $summary_line && $summary_line == "$last" && $(($(date +%s) - r0)) -lt 20 ]]; then
            echo "fuzz-ops: the same report stops each round, the run of $g stops: $summary_line" | tee -a "$log"
            break
        fi
        last=$summary_line
    done
    # executions: the sum of the K rows; findings: the crash files of this run and the numeric
    # findings above the loose bound for inputs without special values
    local execs=0 x numeric
    for x in $(cat "$f"/stats-*.tsv 2> /dev/null | rg '^K' | cut -f3); do
        execs=$((execs + x))
    done
    # rg gives the status 1 when nothing matches; with pipefail that must not stop the group
    numeric=$({ cat "$f"/stats-*.tsv 2> /dev/null || true; } | { rg '^A' || true; } | cut -f2-5 \
        | { rg '\t(above-loose|nonfinite)\t0$' || true; } | sort -u | wc -l)
    local -a crash_files=()
    mapfile -t crash_files < <(find "$w/artifacts" -type f -newer "$marker" | sort)
    json_line "$profile" "$config" "$g" fuzz $(($(date +%s) - t0)) "$execs" $((${#crash_files[@]} + numeric)) "${crash_files[@]}"
    echo "fuzz-ops: fuzz $profile-$config $g done: $round rounds, $execs executions, ${#crash_files[@]} crash files, $numeric numeric findings"
}

run_fuzz() {
    local profile=$1 config=$2 budget=$3 jobs=$4
    shift 4
    local groups=("$@")
    [[ ${#groups[@]} -gt 0 ]] || read -r -a groups <<< "$ALL_GROUPS"
    build_oracle
    build_config "$profile" "$config"
    local g
    for g in "${groups[@]}"; do
        while [[ $(jobs -rp | wc -l) -ge $jobs ]]; do
            wait -n || true
        done
        fuzz_group "$profile" "$config" "$g" "$budget" &
    done
    wait
    summary "$profile" "$config"
}

# Print the merged statistics of the fuzz runs of one build.
summary() {
    local profile=$1 config=$2 dir
    dir=$(host_dir "$1" "$2")
    local -a files
    mapfile -t files < <(find "$dir/findings" -name 'stats-*.tsv' 2> /dev/null | sort)
    [[ ${#files[@]} -gt 0 ]] || { echo "no statistics in $dir/findings"; return 0; }
    "$B_ORACLE/ops_oracle" stats "${files[@]}"
    echo "crash files:"
    find "$dir/work" -path '*/artifacts/*' -type f 2> /dev/null | sort || true
}

# Check the shipped libraries against build/hashes-native.txt.
check_shipped() {
    local lib want got
    for lib in libggml.so libggml-base.so libggml-cpu.so libggml-hexagon.so libggml-opencl.so; do
        want=$(rg -F "  $lib" "$REPO_ROOT/build/hashes-native.txt" | cut -d' ' -f1)
        got=$(sha256sum "$SHIPPED_LIBS/$lib" | cut -d' ' -f1)
        [[ -n $want && $want == "$got" ]] || die "$SHIPPED_LIBS/$lib does not match build/hashes-native.txt ($got, want $want)"
    done
}

# Build ops_replay for the phone (one build for each profile and configuration), make the pack,
# and stage.
phone_build() {
    local -a builds=("$@")
    local p c
    if [[ ${#builds[@]} -eq 0 ]]; then
        for p in $ALL_PROFILES; do
            for c in $PHONE_CONFIGS_ALL; do
                builds+=("$p-$c")
            done
        done
    fi
    build_oracle
    local stage="$MISC/phone"
    mkdir -p "$stage"
    # the pack: seeds, a subset of each CPU corpus, the regression inputs, and random cases.
    # PHONE_KEEP_PACKS=1 keeps cases.pack and san.pack: a phone run must compare with its own pack.
    local g k d
    if [[ ${PHONE_KEEP_PACKS:-0} != 1 || ! -f $stage/cases.pack || ! -f $stage/san.pack ]]; then
    local -a args=(gen --out "$stage/cases.pack" --n "$PACK_N" --seed 20260923 --max-per-dir "$PACK_CORPUS")
    for g in $ALL_GROUPS; do
        args+=(--corpus "$g:$HERE/corpus/$g")
        for d in "$REPO_ROOT"/build/fuzz/ops-{debug,release}-none/work/"$g"/corpus; do
            [[ -d $d ]] && args+=(--corpus "$g:$d")
        done
    done
    if [[ -d "$HERE/regress" ]]; then
        for k in "$HERE"/regress/*/; do
            [[ -d $k ]] && args+=(--cases "$(basename "$k"):$k")
        done
    fi
    "$B_ORACLE/ops_oracle" "${args[@]}"
    # The pack of the sanitizer builds: the seeds and the regression inputs only. The host fuzz
    # suites give the random coverage, thus the phone runs the full pack in the none builds only.
    local -a sargs=(gen --out "$stage/san.pack" --n 0)
    for g in $ALL_GROUPS; do
        sargs+=(--corpus "$g:$HERE/corpus/$g")
    done
    if [[ -d "$HERE/regress" ]]; then
        for k in "$HERE"/regress/*/; do
            [[ -d $k ]] && sargs+=(--cases "$(basename "$k"):$k")
        done
    fi
    "$B_ORACLE/ops_oracle" "${sargs[@]}"
    fi
    # The diagnosis pack of the HTP0 guard writes and run-to-run differences: the regression
    # inputs of MUL_MAT_ID and of the
    # shared-input MUL_MAT (the HTP0 guard write and the HTP0 run-to-run differences), the Q8_0
    # MUL_MAT of the model shapes at n = 1 to 8 (mm_model entries 0 to 63), and the F32 (2048, 16)
    # MUL_MAT of ssm_alpha and ssm_beta of the 2B model, alone and as a pair with one input
    # (entries 128 to 143). The Q4_0 entries 64 to 127 are not in the pack.
    "$B_ORACLE/ops_oracle" gen --out "$stage/diag.pack" --n 0 \
        --cases "mul_mat_id:$HERE/regress/mul_mat_id" --cases "mul_mat_multi:$HERE/regress/mul_mat_multi" \
        --enumerate mm_model:64 --enumerate-from mm_model:128:16
    local b
    for b in "${builds[@]}"; do
        p=${b%%-*}
        c=${b#*-}
        check_profile "$p"
        [[ " $PHONE_CONFIGS_ALL " == *" $c "* ]] || die "the phone configuration '$c' is not one of: $PHONE_CONFIGS_ALL"
        local rt="" prebuilt=""
        # the runtime library of the sanitizer; the ubsan build links its runtime statically
        # (-static-libsan, refer to the vptr text in CMakeLists.txt). The asan builds use the runtime
        # of compiler-rt 22.1.8 in the stage directory asan-rt, not the runtime of the NDK.
        case $c in
            hwasan) rt=libclang_rt.hwasan-aarch64-android.so ;;
        esac
        if [[ $b == release-none ]]; then
            # the release none run uses the shipped libraries themselves
            check_shipped
            prebuilt="-DFUZZ_OPS_GGML_PREBUILT=/workspace/android/snapdragon/jniLibs/arm64-v8a"
        fi
        echo "fuzz-ops: build ops_replay for arm64 Android, $b"
        local bdir="build/fuzz/ops-$b/android"
        mkdir -p "$REPO_ROOT/$bdir"
        # The container sees the repository at /workspace, thus the paths are relative to it.
        container_run "$SNAPDRAGON_IMAGE" bash -euo pipefail -c "
cmake -S tests/fuzz/ops -B $bdir -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE=\$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 \
    -DFUZZ_OPS_VARIANT=android -DFUZZ_PROFILE=$p -DFUZZ_SANITIZER=$c $prebuilt \
    -DFUZZ_OPS_GGML_SRC=/workspace/build/fuzz/ops-src/ggml \
    -DHEXAGON_SDK_ROOT=\$HEXAGON_SDK_ROOT -DHEXAGON_TOOLS_ROOT=\$HEXAGON_TOOLS_ROOT -DPREBUILT_LIB_DIR=android_aarch64
cmake --build $bdir -j$BUILD_JOBS --target ops_replay
if [ '$b' = debug-none ] && [ '${FUZZ_OPS_BUILD_DSP:-0}' = 1 ]; then
    cmake --build $bdir -j$BUILD_JOBS --target htp-v79
fi
if [ -n '$rt' ]; then
    rt=\$(find \$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt -name '$rt' | head -n 1)
    cp -f \"\$rt\" $bdir/
fi
" > "$REPO_ROOT/$bdir/build.log" 2>&1 || die "the Android build of $b failed, refer to $REPO_ROOT/$bdir/build.log"
        rm -rf "${stage:?}/$b"
        mkdir -p "$stage/$b"
        # the NDK runtime is read only, thus -f replaces an earlier copy
        cp -f "$REPO_ROOT/$bdir/ops_replay" "$stage/$b/"
        [[ -n $rt ]] && cp -f "$REPO_ROOT/$bdir/$rt" "$stage/$b/"
        if [[ $b == release-none ]]; then
            cp -f "$SHIPPED_LIBS"/libggml.so "$SHIPPED_LIBS"/libggml-base.so "$SHIPPED_LIBS"/libggml-cpu.so \
                "$SHIPPED_LIBS"/libggml-hexagon.so "$SHIPPED_LIBS"/libggml-opencl.so "$stage/$b/"
        fi
    done
    cp -f "$HERE/phone_run.sh" "$stage/"
    # The ASan runtime of compiler-rt 22.1.8: phone_run.sh puts asan-rt first in LD_LIBRARY_PATH.
    build_asan_runtime
    rm -rf "$stage/asan-rt"
    mkdir -p "$stage/asan-rt"
    cp -f "$ASAN_RUNTIME" "$stage/asan-rt/"
    # The sanitizer runtimes on the phone give function names to the reports only with a symbolizer.
    build_symbolizer
    cp -f "$SYMBOLIZER" "$stage/llvm-symbolizer"
    rm -f "$stage/ubsan.supp"
    [[ -f "$SHARED_SAN/ubsan.supp" ]] && cp -f "$SHARED_SAN/ubsan.supp" "$stage/"
    # FUZZ_OPS_BUILD_DSP=1: the v79 DSP library of the snapshot (it pairs with the host code of the
    # snapshot in the builds other than release-none, which has the shipped host libraries)
    if [[ ${FUZZ_OPS_BUILD_DSP:-0} == 1 && -z $DSP_LIB ]]; then
        DSP_LIB=$(find "$REPO_ROOT/build/fuzz/ops-debug-none/android" -name libggml-htp-v79.so | head -n 1)
        [[ -f $DSP_LIB ]] || die "no libggml-htp-v79.so in build/fuzz/ops-debug-none/android"
    fi
    rm -f "$stage/libggml-htp-v79.so"
    if [[ -n $DSP_LIB ]]; then
        [[ -f $DSP_LIB ]] || die "DSP_LIB=$DSP_LIB is not a file"
        cp -f "$DSP_LIB" "$stage/libggml-htp-v79.so"
    fi
    (cd "$stage" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
    cat "$stage/SHA256SUMS"
}

# Print one phone run: the thermal status, the run inside timeout, the thermal status again, and
# the check that no process is left.
phone_run_cmd() {
    local build=$1 tag=$2 backends=$3 suffix=$4 fusion=$5 adsp=$6 count=$7 halt=$8 trace=${9:-}
    local d=$PHONE_DIR
    echo "adb -s $PHONE shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
    local penv=""
    [[ -n ${FUZZ_OPS_CMD_PACK:-} && ${FUZZ_OPS_CMD_PACK} != cases.pack ]] && penv="env FUZZ_OPS_PACK=$FUZZ_OPS_CMD_PACK "
    echo "adb -s $PHONE shell 'timeout -s KILL 100 ${penv}sh $d/phone_run.sh $build $tag $backends $suffix $fusion $adsp $PHONE_SECONDS $count $halt $trace'"
    echo "adb -s $PHONE shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
    echo "adb -s $PHONE shell 'pgrep -a ops_replay; tail -n 4 $d/out/log-$build-$tag.txt'"
}

phone_commands() {
    local what=${1:-probe}
    local stage="$MISC/phone" d=$PHONE_DIR adsp=$ADSP_DIR b
    [[ -f "$stage/cases.pack" && -f "$stage/phone_run.sh" ]] || die "run tests/fuzz/ops/run.sh phone-build first"
    local -a staged=()
    for b in "$stage"/{debug,release}-{none,asan,hwasan,ubsan}/; do
        [[ -f "$b/ops_replay" ]] && staged+=("$(basename "$b")")
    done
    if [[ $what == diag ]]; then
        echo "# 1. Push the files of the diagnosis. The work directory stays as it is (the full commands"
        echo "#    empty it later, thus pull the diagnosis results before them)."
        echo "adb -s $PHONE shell 'mkdir -p $d/out'"
        echo "adb -s $PHONE push $stage/phone_run.sh $stage/diag.pack $d/"
        echo "adb -s $PHONE push $stage/$DIAG_BUILD $d/"
        if [[ -f "$stage/libggml-htp-v79.so" ]]; then
            adsp="$d/dsp"
            echo "adb -s $PHONE shell 'mkdir -p $d/dsp'"
            echo "adb -s $PHONE push $stage/libggml-htp-v79.so $d/dsp/"
        fi
        echo "adb -s $PHONE shell 'chmod 755 $d/$DIAG_BUILD/ops_replay; sha256sum $d/diag.pack $d/$DIAG_BUILD/ops_replay $adsp/libggml-htp-v79.so | cut -c1-16'"
    else
    echo "# 1. Push the files into an empty work directory."
    echo "adb -s $PHONE shell 'rm -rf $d && mkdir -p $d/out $d/dsp'"
    echo "adb -s $PHONE push $stage/phone_run.sh $stage/cases.pack $stage/san.pack $stage/diag.pack $d/"
    for b in "${staged[@]}"; do
        echo "adb -s $PHONE push $stage/$b $d/"
    done
    [[ -f "$stage/ubsan.supp" ]] && echo "adb -s $PHONE push $stage/ubsan.supp $d/"
    [[ -f "$stage/llvm-symbolizer" ]] && echo "adb -s $PHONE push $stage/llvm-symbolizer $d/"
    [[ -d "$stage/asan-rt" ]] && echo "adb -s $PHONE push $stage/asan-rt $d/"
    echo "adb -s $PHONE shell 'chmod 755 $d/*/ops_replay $d/llvm-symbolizer; sha256sum $d/llvm-symbolizer $d/*/ops_replay | cut -c1-16'"
    if [[ -f "$stage/libggml-htp-v79.so" ]]; then
        adsp="$d/dsp"
        echo "adb -s $PHONE push $stage/libggml-htp-v79.so $d/dsp/"
    fi
    fi
    echo
    if [[ $what == diag ]]; then
        # diag.pack: the regression inputs of mul_mat_id and mul_mat_multi, 64 Q8_0 cases, then the
        # 16 F32 cases. Each regression input decodes to one case.
        local DIAG_F32_FIRST
        DIAG_F32_FIRST=$(( $(find "$HERE/regress/mul_mat_id" "$HERE/regress/mul_mat_multi" -name '*.bin' | wc -l) + 64 ))
        echo "# 2. The diagnosis of the HTP0 guard writes and run-to-run differences on HTP0 (the"
        echo "#    $DIAG_BUILD build): diag.pack, each case 20 times with new buffers (--repeat 20). A run with"
        echo "#    different outputs gets NONDET, a write outside a tensor gets GUARD. Three variants:"
        echo "#    the default, one HVX thread (GGML_HEXAGON_NHVX=1), and no graph and batch caches."
        echo "#    Each command resumes the last one; run each command until its log shows 'all runs done'."
        echo "#    A fourth variant runs the F32 (2048, 16) cases of ssm_alpha and ssm_beta (the last 16"
        echo "#    cases of diag.pack, from index $DIAG_F32_FIRST) without the fusions."
        local variant tag xenv n fusion extra parts
        for variant in default nhvx1 nocache nofuse-f32; do
            fusion=1
            extra="--repeat 20"
            parts=4
            case $variant in
                default)    xenv="" ;;
                nhvx1)      xenv="GGML_HEXAGON_NHVX=1" ;;
                nocache)    xenv="GGML_HEXAGON_GRAPHCACHE=0 GGML_HEXAGON_BATCHCACHE=0" ;;
                nofuse-f32) xenv=""; fusion=0; extra="--repeat 20 --first $DIAG_F32_FIRST"; parts=1 ;;
            esac
            tag="diag-$variant"
            for n in $(seq 1 "$parts"); do
                echo "# $variant, part $n"
                echo "adb -s $PHONE shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
                echo "adb -s $PHONE shell 'timeout -s KILL 100 env FUZZ_OPS_PACK=diag.pack FUZZ_OPS_EXTRA=\"$extra\" FUZZ_OPS_ENV=\"$xenv\" sh $d/phone_run.sh $DIAG_BUILD $tag HTP0 -$variant $fusion $adsp $PHONE_SECONDS all 1'"
                echo "adb -s $PHONE shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
                echo "adb -s $PHONE shell 'pgrep -a ops_replay; tail -n 2 $d/out/log-$DIAG_BUILD-$tag.txt'"
            done
        done
        echo
        echo "# 3. Pull the logs and results."
        echo "mkdir -p $stage/pulled-diag"
        echo "adb -s $PHONE pull $d/out $stage/pulled-diag/"
        return 0
    fi
    if [[ $what == probe ]]; then
        echo "# 2. The probe: the first 3 cases on CPU and HTP0 with each build (one profile, one sanitizer)."
        echo "#    The ASan builds run under the ptrace tracer of ops_replay (--trace): it prints the pc, the"
        echo "#    module, the instruction, si_code, BTYPE and a backtrace of each fatal signal of each"
        echo "#    thread, also in a thread that blocks the signal. phone_run.sh starts each ASan run with"
        echo "#    the thread self-test and the ASan runtime of compiler-rt 22.1.8 (asan-rt). The crash buffer of"
        echo "#    logcat shows a tombstone if debuggerd saw a signal."
        echo "#    The ubsan builds run two times: UBSAN_HALT=0 prints every report (the evidence), then"
        echo "#    UBSAN_HALT=1 shows that the entries of ubsan.supp match on the phone (a stop is a"
        echo "#    report that no entry matches)."
        for b in "${staged[@]}"; do
            echo "# probe of $b"
            if [[ $b == *-asan ]]; then
                phone_run_cmd "$b" probe CPU,HTP0 - 1 "$adsp" 3 0 trace
                echo "adb -s $PHONE shell 'timeout -s KILL 20 logcat -d -b crash -t 60'"
            elif [[ $b == *-ubsan ]]; then
                phone_run_cmd "$b" probe CPU,HTP0 - 1 "$adsp" 3 0
                phone_run_cmd "$b" probehalt CPU,HTP0 -halt 1 "$adsp" 3 1
            else
                phone_run_cmd "$b" probe CPU,HTP0 - 1 "$adsp" 3 0
            fi
        done
        echo
        echo "# 3. Pull the probe logs and results."
        echo "mkdir -p $stage/pulled"
        echo "adb -s $PHONE pull $d/out $stage/pulled/"
        return 0
    fi
    local -a runs=("${staged[@]}")
    [[ -n $PHONE_BUILDS ]] && read -r -a runs <<< "$PHONE_BUILDS"
    echo "# 2. The full runs, in parts of $PHONE_SECONDS s (each command stops at its deadline and the next"
    echo "#    part resumes). The none builds run cases.pack (1019 cases) on CPU and HTP0 with the"
    echo "#    fusions of the app, and on HTP0 without the fusions. The sanitizer builds run san.pack"
    echo "#    (the seeds and the regression inputs) on CPU and HTP0 with the fusions: the sanitizers"
    echo "#    instrument the host code, and the host fuzz suites give the random coverage. When the log"
    echo "#    of a run shows 'all runs done', skip its other parts. The part counts are estimates."
    local parts n pack
    for b in "${runs[@]}"; do
        case ${b#*-} in
            none) parts=3; pack=cases.pack ;; *) parts=1; pack=san.pack ;;
        esac
        [[ ${b%%-*} == debug ]] && parts=$((parts + 1))
        for n in $(seq 1 "$parts"); do
            echo "# $b: app, part $n of $parts ($pack)"
            # release-none links the shipped host libraries, thus it pairs with the shipped DSP
            # library (ADSP_DIR); the other builds have the host code of the snapshot
            local badsp=$adsp
            [[ $b == release-none ]] && badsp=$ADSP_DIR
            FUZZ_OPS_CMD_PACK=$pack phone_run_cmd "$b" app CPU,HTP0 - 1 "$badsp" all 1
        done
        if [[ ${b#*-} == none ]]; then
            for n in 1 2; do
                echo "# $b: nofuse, part $n of 2"
                phone_run_cmd "$b" nofuse HTP0 -nofuse 0 "$badsp" all 1
            done
        fi
    done
    echo
    echo "# 3. Pull the results, then run: tests/fuzz/ops/run.sh compare"
    echo "mkdir -p $stage/pulled"
    echo "adb -s $PHONE pull $d/out $stage/pulled/"
    echo
    echo "# 4. Clean the phone work directory when all runs are complete."
    echo "adb -s $PHONE shell 'rm -rf $d'"
}

# Compare the pulled phone results with the oracle. Each result file goes with its pack: the
# diagnosis runs (tag diag-*) with diag.pack, the none builds with cases.pack, and the sanitizer
# builds with san.pack. One table and one findings directory for each pack.
compare() {
    build_oracle
    local stage="$MISC/phone"
    local -a results=("$@")
    if [[ ${#results[@]} -eq 0 ]]; then
        mapfile -t results < <(find "$stage"/pulled* -name 'res-*.bin' | sort)
    fi
    [[ ${#results[@]} -gt 0 ]] || die "no result files: pull them first (run.sh phone-commands)"
    local -a cases_r=() san_r=() diag_r=()
    local r base
    for r in "${results[@]}"; do
        base=$(basename "$r")
        if [[ $base == *-diag-* ]]; then
            diag_r+=(--results "$r")
        elif [[ $base == res-debug-none-* || $base == res-release-none-* ]]; then
            cases_r+=(--results "$r")
        else
            san_r+=(--results "$r")
        fi
    done
    mkdir -p "$MISC/logs"
    : > "$MISC/logs/phone-compare.txt"
    local pack
    for pack in cases san diag; do
        local -n list="${pack}_r"
        [[ ${#list[@]} -gt 0 && -f "$stage/$pack.pack" ]] || continue
        echo "== $pack.pack" | tee -a "$MISC/logs/phone-compare.txt"
        "$B_ORACLE/ops_oracle" compare --pack "$stage/$pack.pack" --findings "$MISC/findings-phone-$pack" \
            "${list[@]}" | tee -a "$MISC/logs/phone-compare.txt"
    done
}

minimize() {
    [[ $# -ge 4 ]] || die "minimize needs FILE GROUP KIND VERDICT [PROFILE CONFIG]"
    local file=$1 g=$2 kind=$3 verdict=$4 profile=${5:-release} config=${6:-none}
    check_profile "$profile"
    check_host_config "$config"
    build_oracle
    build_config "$profile" "$config"
    local w="$MISC/minimize" dst="$HERE/regress/$kind"
    rm -rf "$w"
    mkdir -p "$w" "$dst"
    with_san "$config" env FUZZ_OPS_ORACLE="$B_ORACLE/ops_oracle" FUZZ_OPS_GROUP="$g" FUZZ_OPS_FINDINGS="$w/findings" \
        FUZZ_OPS_ABORT="kind=$kind,verdict=$verdict" FUZZ_ARTIFACT_DIR="$w" timeout -s KILL 900 nice -n 10 \
        "$(host_dir "$profile" "$config")/fuzz_ops" \
        -minimize_crash=1 -max_total_time=300 -runs=100000 -artifact_prefix="$w/" -exact_artifact_path="$w/min.bin" \
        "$file" > "$w/log.txt" 2>&1 || true
    [[ -f "$w/min.bin" ]] || die "no minimized input, refer to $w/log.txt"
    local name
    name="$(basename "${file%.bin}")-min.bin"
    cp "$w/min.bin" "$dst/$name"
    echo "fuzz-ops: $dst/$name ($(stat -c %s "$dst/$name") bytes)"
    "$B_ORACLE/ops_oracle" show "$dst/$name" --kind "$kind"
}

# Parse the options of test and fuzz, then run each profile in sequence.
suite() {
    local mode=$1 config=${2:-}
    [[ -n $config ]] || die "$mode needs a configuration: $HOST_CONFIGS"
    check_host_config "$config"
    shift 2
    local budget=$BUDGET jobs=$JOBS profiles=$ALL_PROFILES
    local -a groups=()
    while [[ $# -gt 0 ]]; do
        case $1 in
            --budget-seconds) budget=$2; shift 2 ;;
            --jobs)           jobs=$2; shift 2 ;;
            --profile)        check_profile "$2"; profiles=$2; shift 2 ;;
            -*)               die "unknown option $1" ;;
            *)                groups+=("$1"); shift ;;
        esac
    done
    local p rc=0
    for p in $profiles; do
        if [[ $mode == test ]]; then
            run_test "$p" "$config" || rc=1
        else
            run_fuzz "$p" "$config" "$budget" "$jobs" "${groups[@]}"
        fi
    done
    return $rc
}

# The llvm-symbolizer of the phone: tests/sanitizers/build-android-symbolizer.sh builds it when it is
# missing. The sanitizer runtimes on the phone need it: without it a report frame has no function
# name, thus no function-level entry of tests/sanitizers/ubsan.supp can match it.
build_symbolizer() {
    if [[ ! -x $SYMBOLIZER ]]; then
        "$SHARED_SAN/build-android-symbolizer.sh" --jobs "$BUILD_JOBS" \
            || die "llvm-symbolizer for Android did not build (tests/sanitizers/build-android-symbolizer.sh)"
    fi
    echo "fuzz-ops: $SYMBOLIZER ($(sha256sum "$SYMBOLIZER" | cut -c1-16))"
}

# Build the libraries of the app and llama-bench for arm64 Android from the private llama.cpp tree
# of a fix (the parent of FUZZ_OPS_SRC), with the flags of the preset
# arm64-android-snapdragon-release and the reproducibility flags of scripts/build-native.sh, for
# the phone check of the fix. It never writes build/native or the jniLibs of the
# app. The output: build/fuzz/ops/phone/libs-TAG/.
phone_libs() {
    [[ -n $TAG ]] || die "phone-libs needs FUZZ_OPS_SRC and FUZZ_OPS_TAG (the private tree of a fix)"
    local tree rel bdir out lib
    tree=$(cd "$HOST_SRC/.." && pwd)
    rel=${tree#"$REPO_ROOT"/}
    [[ $rel != "$tree" && -f $tree/CMakeLists.txt ]] || die "the tree $tree must be a llama.cpp tree in the repository"
    cp -f "$REPO_ROOT/android/snapdragon/CMakeUserPresets.json" "$tree/CMakeUserPresets.json"
    bdir="build/fuzz/ops/phone-libs-$TAG"
    out="$MISC/phone/libs-$TAG"
    mkdir -p "$REPO_ROOT/$bdir"
    # FUZZ_OPS_REPRO_PATHS=1: the container sees the tree at third_party/llama.cpp and the build
    # directory at build/native/llama (bind mounts in the container only; the host directories do
    # not change). The libraries hold these paths and the commit of the tree, thus a git tree at
    # the pinned commit with the series gives the bytes of scripts/build-native.sh.
    local -a mounts=()
    local bin_dir="$REPO_ROOT/$bdir/bin"
    if [[ ${FUZZ_OPS_REPRO_PATHS:-0} == 1 ]]; then
        mounts=(-v "$tree:/workspace/third_party/llama.cpp" -v "$REPO_ROOT/$bdir:/workspace/build/native/llama"
            -e SOURCE_DATE_EPOCH="$(source_date_epoch)")
        rel=third_party/llama.cpp
        bdir=build/native/llama
    fi
    echo "fuzz-ops: build the app libraries and llama-bench from $rel into $bdir"
    container_run "${mounts[@]}" "$SNAPDRAGON_IMAGE" bash -euo pipefail -c "
repro='-ffile-prefix-map=/workspace=. -fdebug-prefix-map=/workspace=. -Werror=date-time'
flags=\$(python3 -c 'import json, sys; p = [x for x in json.load(open(sys.argv[1]))[\"configurePresets\"] if x[\"name\"] == \"arm64-android-snapdragon\"][0]; print(p[\"cacheVariables\"][\"CMAKE_C_FLAGS\"])' $rel/CMakeUserPresets.json)
export CFLAGS=\"\$repro\" CXXFLAGS=\"\$repro\"
cmake -S $rel --preset arm64-android-snapdragon-release -B $bdir \
    -DLLAMA_BUILD_NUMBER=$LLAMA_BUILD_NUMBER -DLLAMA_BUILD_COMMIT=${LLAMA_COMMIT:0:7} \
    -DCMAKE_C_FLAGS=\"\$flags \$repro\" -DCMAKE_CXX_FLAGS=\"\$flags \$repro\"
cmake --build $bdir -j$BUILD_JOBS --target $LLAMA_LIBS llama-bench
" > "$REPO_ROOT/build/fuzz/ops/phone-libs-$TAG.log" 2>&1 \
        || die "the build failed, refer to $REPO_ROOT/build/fuzz/ops/phone-libs-$TAG.log"
    # The app libraries go to libs-TAG. llama-bench and its own library libllama-bench-impl.so are
    # tools, not app libraries: they go to libs-TAG/tools.
    rm -rf "$out"
    mkdir -p "$out/tools"
    for lib in $LLAMA_LIBS; do
        cp -f "$bin_dir/lib$lib.so" "$out/"
    done
    cp -f "$bin_dir/llama-bench" "$bin_dir/libllama-bench-impl.so" "$out/tools/"
    (cd "$out" && sha256sum ./*.so tools/* > SHA256SUMS)
    cat "$out/SHA256SUMS"
}

# Stage and print the phone check of a fix before it lands: the shipped libraries (before)
# against the libraries of the fix (after, from phone-libs), in one command set. Build "after"
# with FUZZ_OPS_REPRO_PATHS=1 from a git tree: then each library that the fix does not change is
# byte-identical to the shipped one, and the shipped set is a clean "before".
#   1. The op pack on CPU and HTP0 with the release none replay driver on each library set. The
#      command "ops_oracle same" then shows if the outputs are bit-identical.
#   2. llama-bench, tg on the 4B model and pp on the 2B model, on the CPU, 5 repetitions. For
#      each model, one warmup run that is not measured, then the order before, after, before,
#      after, with the thermal status, the clock caps and the battery between the runs.
#      llama-bench and libllama-bench-impl.so are one tool set (the same files for the two
#      library sets), in a directory after the library set in LD_LIBRARY_PATH.
phone_check_commands() {
    [[ -n $TAG ]] || die "phone-check-commands needs FUZZ_OPS_TAG (the libraries of phone-libs)"
    local stage="$MISC/phone" d=$PHONE_DIR lib side
    local after="$stage/libs-$TAG" tools="$stage/check-$TAG-tools"
    [[ -f $after/tools/llama-bench && -f $stage/release-none/ops_replay && -f $stage/cases.pack ]] \
        || die "run phone-build and phone-libs first"
    # "before": the shipped libraries, or with FUZZ_OPS_CHECK_BEFORE=TAG0 the private build libs-TAG0
    # (the current series without the fix, when the series has changed since the shipped build)
    local before_dir="$SHIPPED_LIBS"
    if [[ -n ${FUZZ_OPS_CHECK_BEFORE:-} ]]; then
        before_dir="$stage/libs-$FUZZ_OPS_CHECK_BEFORE"
        [[ -f $before_dir/libggml-cpu.so ]] || die "no libraries in $before_dir (run phone-libs for that tag)"
    else
        check_shipped
    fi
    for side in before after; do
        rm -rf "$stage/check-$TAG-$side"
        mkdir -p "$stage/check-$TAG-$side"
        cp -f "$stage/release-none/ops_replay" "$stage/check-$TAG-$side/"
        for lib in $LLAMA_LIBS; do
            if [[ $side == before ]]; then
                cp -f "$before_dir/lib$lib.so" "$stage/check-$TAG-$side/"
            else
                cp -f "$after/lib$lib.so" "$stage/check-$TAG-$side/"
            fi
        done
    done
    rm -rf "$tools"
    mkdir -p "$tools"
    cp -f "$after/tools/llama-bench" "$after/tools/libllama-bench-impl.so" "$tools/"
    local status="adb -s $PHONE shell 'dumpsys thermalservice | grep \"Thermal Status\"; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq; dumpsys battery | grep -E \"powered|status|temperature\"'"
    local m4=/data/local/tmp/qwen/models/Qwen3.5-4B-Q8_0.gguf m2=/data/local/tmp/qwen/models/Qwen3.5-2B-Q8_0.gguf
    local bench="$d/check-$TAG-tools/llama-bench -dev none -ngl 0 -t 6 -fa 1 -o md"
    echo "# The phone check of FUZZ_OPS_TAG=$TAG, before = $before_dir. Do not run on the charger."
    echo "adb -s $PHONE shell 'mkdir -p $d/out'"
    echo "adb -s $PHONE push $stage/check-$TAG-before $stage/check-$TAG-after $tools $d/"
    echo "adb -s $PHONE shell 'chmod 755 $d/check-$TAG-*/ops_replay $d/check-$TAG-tools/llama-bench; cd $d && sha256sum check-$TAG-*/*.so check-$TAG-tools/llama-bench | cut -c1-16'"
    echo "# 1. The op pack with each library set (the first 600 cases, CPU and HTP0)."
    for side in before after; do
        phone_run_cmd "check-$TAG-$side" libcheck CPU,HTP0 - 1 "$ADSP_DIR" 600 1
    done
    echo "# 2. llama-bench on the CPU: tg on the 4B model, then pp on the 2B model."
    local what model args n
    for what in tg pp; do
        if [[ $what == tg ]]; then model=$m4; args="-p 0 -n 32"; else model=$m2; args="-p 128 -n 0"; fi
        echo "# $what: the warmup run (1 repetition, not measured)"
        echo "$status"
        echo "adb -s $PHONE shell 'cd $d && timeout -s KILL 100 env LD_LIBRARY_PATH=$d/check-$TAG-before:$d/check-$TAG-tools ADSP_LIBRARY_PATH=$ADSP_DIR $bench -r 1 -m $model $args > /dev/null 2> out/bench-$TAG-$what-warmup.log; echo $what warmup rc=\$?'"
        n=0
        for side in before after before after; do
            n=$((n + 1))
            echo "$status"
            echo "adb -s $PHONE shell 'cd $d && timeout -s KILL 100 env LD_LIBRARY_PATH=$d/check-$TAG-$side:$d/check-$TAG-tools ADSP_LIBRARY_PATH=$ADSP_DIR $bench -r 5 -m $model $args 2> out/bench-$TAG-$what-$side-$n.log; echo $what $side $n rc=\$?'"
        done
    done
    echo "$status"
    echo "adb -s $PHONE shell 'pgrep -a llama; pgrep -a ops_replay'"
    echo "# 3. Pull, then: build/fuzz/ops/oracle/ops_oracle same PULLED/res-check-$TAG-before-libcheck.bin PULLED/res-check-$TAG-after-libcheck.bin"
    echo "mkdir -p $stage/pulled-check-$TAG"
    echo "adb -s $PHONE pull $d/out $stage/pulled-check-$TAG/"
}

# The ASan runtime of the phone: tests/sanitizers/build-asan-android-runtime.sh
# builds it from compiler-rt 22.1.8. The runtime of NDK r29 kills each new thread on the SM8750.
build_asan_runtime() {
    if [[ ! -f $ASAN_RUNTIME ]]; then
        "$SHARED_SAN/build-asan-android-runtime.sh" --jobs "$BUILD_JOBS" \
            || die "the ASan runtime for Android did not build (tests/sanitizers/build-asan-android-runtime.sh)"
    fi
    "$SHARED_SAN/build-asan-android-runtime.sh" --verify-only > /dev/null \
        || die "the ASan runtime $ASAN_RUNTIME does not pass its checks"
}

main() {
    local mode=${1:-help}
    [[ $# -gt 0 ]] && shift
    case $mode in
        test | fuzz)    suite "$mode" "$@" ;;
        cpu-asan)       suite fuzz asan "$@" ;;
        cpu-tsan)       suite fuzz tsan "$@" ;;
        regress)        suite test asan ;;
        phone-build)    phone_build "$@" ;;
        phone-commands) phone_commands "$@" ;;
        compare)        compare "$@" ;;
        summary)        build_oracle; summary "${1:-release}" "${2:-none}" ;;
        minimize)       minimize "$@" ;;
        snapshot)       take_snapshot ;;
        symbolizer)     build_symbolizer ;;
        asan-runtime)   build_asan_runtime ;;
        oracle-x86)     build_oracle_x86 ;;
        phone-libs)     phone_libs ;;
        phone-check-commands) phone_check_commands ;;
        bounds)         build_oracle; "$B_ORACLE/ops_oracle" bounds ;;
        help | -h | --help) usage ;;
        *)              usage; die "unknown mode: $mode" ;;
    esac
}

main "$@"
# Stop here: an edit of this file while a long run reads it must not run the new text.
exit $?
