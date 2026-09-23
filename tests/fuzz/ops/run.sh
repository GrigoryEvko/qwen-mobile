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
readonly B_ORACLE="$MISC/oracle"
readonly SHARED_SAN="$REPO_ROOT/tests/sanitizers"
readonly SHIPPED_LIBS="$REPO_ROOT/android/snapdragon/jniLibs/arm64-v8a"
readonly SYMBOLIZER="$MISC/tools/llvm-symbolizer-android-arm64"   # run.sh symbolizer builds it
readonly ASAN_RUNTIME="$MISC/tools/libclang_rt.asan-aarch64-android.so"   # run.sh asan-runtime (task #176)
# FUZZ_OPS_SRC: a different ggml tree for the host builds, for example the private copy of a fix
# (landing rule L1). FUZZ_OPS_TAG: the suffix of the build directories of that tree.
readonly HOST_SRC=${FUZZ_OPS_SRC:-$SRC}
readonly TAG=${FUZZ_OPS_TAG:-}
if [[ $HOST_SRC != "$SRC" && -z $TAG ]]; then
    echo "fuzz-ops: FUZZ_OPS_SRC needs FUZZ_OPS_TAG, thus the builds of the tree stay apart" >&2
    exit 2
fi
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
PACK_N=${PACK_N:-40}
PACK_CORPUS=${PACK_CORPUS:-150}
PHONE_SECONDS=${PHONE_SECONDS:-85}

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
  fuzz CONFIG   Run libFuzzer on each group for --budget-seconds (default 600), --jobs groups at
                a time (default 4). The default is every group. The fuzz suite sets FUZZ_OPS_TAME=1
                (refer to src/case.cpp): the value asserts of the ggml CPU ops stop every input
                with a special value, and the test suite reports them.

Other modes:
  phone-build [PROFILE-CONFIG...]  Build ops_replay for arm64 Android: one build for each profile
                (debug, release) and each configuration (none, asan, hwasan, ubsan); the default is
                all eight. The release none build links the shipped libraries of
                android/snapdragon/jniLibs/arm64-v8a (hash-checked against build/hashes-native.txt).
                Then make the case pack, and stage the phone files in build/fuzz/ops/phone.
  phone-commands [probe|full]   Print the adb commands. "probe" runs the first 3 cases of each
                build and prints every UBSan report (the evidence run). "full" runs the pack with
                the builds of PHONE_BUILDS (default: all staged builds). This script never runs adb.
  compare [RESULT...]      Compare the pulled phone results with the oracle, print the per-op
                error table, and write the findings to build/fuzz/ops/findings-phone.
  summary PROFILE CONFIG   Print the merged statistics of the fuzz runs of one build.
  minimize FILE GROUP KIND VERDICT [PROFILE CONFIG]
                           Minimize a numeric finding with libFuzzer, and write the result to
                           tests/fuzz/ops/regress/KIND/. VERDICT is above-strict, above-loose or nonfinite.
  snapshot                 Take a new private copy of third_party/llama.cpp/ggml. The phone run needs
                           a host library that pairs with the DSP library, thus take it with care.
  symbolizer               Build llvm-symbolizer for arm64 Android (phone-build does it when it is
                           missing). Without it, a sanitizer report on the phone has no function
                           names, and the function-level entries of tests/sanitizers/ubsan.supp do not
                           match. It needs llvm-tblgen on the host and an NDK (NDK_HOST).
  bounds                   Print the bound rule of each kind.
  help                     Print this text.
  Aliases of the older modes: cpu-asan = fuzz asan, cpu-tsan = fuzz tsan, regress = test asan.

Groups (the targets): matmul gdn attn norm elem data

Flags and runtime options: the shared files tests/sanitizers/profile-<profile>.cmake,
tests/sanitizers/<config>.cmake and tests/sanitizers/env.sh. Suppressions: only the shared files
tests/sanitizers/<config>.supp. The entries of this area in tests/sanitizers/ubsan.supp are for
task #127 (function).

Environment (defaults in parentheses):
  FUZZ_BUDGET, FUZZ_JOBS  the defaults of --budget-seconds and --jobs (600, 4)
  BUILD_JOBS     the parallel build jobs (8)
  RSS_MB         the memory limit of a fuzzer process in MB (4096)
  PHONE          the adb serial (192.168.14.130:5555)
  PHONE_DIR      the work directory on the phone (/data/local/tmp/qwen/fuzz/ops)
  PHONE_BUILDS   the builds of phone-commands full, for example "release-none debug-asan"
  DSP_LIB        a DSP library to push as libggml-htp-v79.so; empty uses ADSP_DIR as it is
  ADSP_DIR       the ADSP_LIBRARY_PATH on the phone without DSP_LIB (/data/local/tmp/qwen/q8ref/lib)
  PACK_N         the random cases of each kind in the phone pack (40)
  PACK_CORPUS    the most inputs from each CPU corpus in the phone pack (150)
  PHONE_SECONDS  the seconds of each phone command before its deadline (85)
  FUZZ_OPS_SRC, FUZZ_OPS_TAG  a different ggml tree for the host builds (the private copy of a
                 fix) and the suffix of its build directories: build/fuzz/ops-PROFILE-CONFIG-TAG
  FUZZ_OPS_UBSAN_SUPP  a UBSan suppression file in place of tests/sanitizers/ubsan.supp (the check
                 of a fix without the entries of its task)
EOF
}

# Copy the ggml tree of the submodule into the private snapshot, and record the date and hashes.
take_snapshot() {
    rm -rf "$SNAP"
    mkdir -p "$SNAP"
    cp -a "$LLAMA_SUBMODULE/ggml" "$SNAP/"
    date > "$SNAP/SNAPSHOT-DATE"
    sha256sum "$SRC/src/ggml-hexagon/ggml-hexagon.cpp" "$SRC/src/ggml-hexagon/htp/htp-ops.h" \
        "$SRC/src/ggml-hexagon/htp-opnode.h" > "$SNAP/SNAPSHOT-HASHES"
    echo "fuzz-ops: new snapshot in $SNAP"
}

# Make sure that the snapshot exists.
need_snapshot() {
    [[ -d "$SRC/src" ]] || take_snapshot
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
    need_snapshot
    mkdir -p "$B_ORACLE"
    if [[ ! -f "$B_ORACLE/build.ninja" ]]; then
        CC=gcc CXX=g++ nice -n 10 cmake -S "$HERE" -B "$B_ORACLE" -G Ninja -DCMAKE_BUILD_TYPE=Release \
            -DFUZZ_OPS_VARIANT=oracle -DFUZZ_SANITIZER=none -DFUZZ_OPS_GGML_SRC="$SRC" > "$B_ORACLE/cmake.log" 2>&1 \
            || die "the configure of the oracle failed, refer to $B_ORACLE/cmake.log"
    fi
    nice -n 10 cmake --build "$B_ORACLE" -j"$BUILD_JOBS" --target ops_oracle > "$B_ORACLE/build.log" 2>&1 \
        || die "the build of the oracle failed, refer to $B_ORACLE/build.log"
}

# Configure (once) and build the host fuzzer of one profile and one configuration. The flags come
# from the shared initial caches (rules R1, R4, R7, R11, R12).
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
    nice -n 10 cmake --build "$dir" -j"$BUILD_JOBS" --target fuzz_ops ops_replay > "$dir/build.log" 2>&1 \
        || die "the build of $dir failed, refer to $dir/build.log"
}

# Run a command with the runtime options of one configuration, from the shared file
# tests/sanitizers/env.sh (the same options in each area, rule R1). The subshell keeps the
# options away from the other runs.
with_san() {
    local config=$1
    shift
    (
        # shellcheck source=../../sanitizers/env.sh
        source "$SHARED_SAN/env.sh"
        sanitizer_env "$config" || exit 2
        # FUZZ_OPS_UBSAN_SUPP replaces the shared UBSan suppression file for the check of a fix: the
        # file without the entries of its task shows that the fix removes the reports (rule L4)
        # before the landing removes the entries from the shared file (rule L8).
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
        # A regression input from the fuzz suite has the prefix "tame-": the fuzz suite decodes
        # with FUZZ_OPS_TAME=1, thus the replay must decode it the same way.
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
    local total_bad=0 g
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
        # FUZZ_OPS_TAME=1: the value asserts of the ggml CPU ops (F-ASSERT-1 to 4) stop each round at
        # the first input with a special value; the test suite reports them (rule R8).
        # The outer timeout stops a process that hangs (for example the TSan report path under
        # libFuzzer can deadlock). libFuzzer stops by itself within -timeout after -max_total_time,
        # thus a kill by the outer timeout is a hang, and the hang is a finding.
        local rc=0
        with_san "$config" env FUZZ_OPS_ORACLE="$B_ORACLE/ops_oracle" FUZZ_OPS_GROUP="$g" FUZZ_OPS_FINDINGS="$f" \
            FUZZ_OPS_TAME=1 FUZZ_ARTIFACT_DIR="$w/artifacts" \
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
    numeric=$(cat "$f"/stats-*.tsv 2> /dev/null | rg '^A' | cut -f2-5 | rg '\t(above-loose|nonfinite)\t0$' | sort -u | wc -l)
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
    # the pack: seeds, a subset of each CPU corpus, the regression inputs, and random cases
    local -a args=(gen --out "$stage/cases.pack" --n "$PACK_N" --seed 20260923 --max-per-dir "$PACK_CORPUS")
    local g k d
    for g in $ALL_GROUPS; do
        args+=(--corpus "$g:$HERE/corpus/$g")
        for d in "$REPO_ROOT"/build/fuzz/ops-*-*/work/"$g"/corpus; do
            # the fuzz suite decodes with FUZZ_OPS_TAME=1, thus its corpus keeps that decode
            [[ -d $d ]] && args+=(--tame-corpus "$g:$d")
        done
    done
    if [[ -d "$HERE/regress" ]]; then
        for k in "$HERE"/regress/*/; do
            [[ -d $k ]] && args+=(--cases "$(basename "$k"):$k")
        done
    fi
    "$B_ORACLE/ops_oracle" "${args[@]}"
    local b
    for b in "${builds[@]}"; do
        p=${b%%-*}
        c=${b#*-}
        check_profile "$p"
        [[ " $PHONE_CONFIGS_ALL " == *" $c "* ]] || die "the phone configuration '$c' is not one of: $PHONE_CONFIGS_ALL"
        local rt="" prebuilt=""
        # the runtime library of the sanitizer; the ubsan build links its runtime statically
        # (-static-libsan, refer to F-UB-3 in CMakeLists.txt)
        case $c in
            asan)   rt=libclang_rt.asan-aarch64-android.so ;;
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
    # The sanitizer runtimes on the phone give function names to the reports only with a symbolizer.
    build_symbolizer
    cp -f "$SYMBOLIZER" "$stage/llvm-symbolizer"
    rm -f "$stage/ubsan.supp"
    [[ -f "$SHARED_SAN/ubsan.supp" ]] && cp -f "$SHARED_SAN/ubsan.supp" "$stage/"
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
    echo "adb -s $PHONE shell 'timeout -s KILL 100 sh $d/phone_run.sh $build $tag $backends $suffix $fusion $adsp $PHONE_SECONDS $count $halt $trace'"
    echo "adb -s $PHONE shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
    echo "adb -s $PHONE shell 'pgrep -a ops_replay; tail -n 4 $d/out/log-$build-$tag.txt'"
}

phone_commands() {
    local what=${1:-probe}
    local stage="$MISC/phone" d=$PHONE_DIR adsp=$ADSP_DIR b
    [[ -f "$stage/cases.pack" && -f "$stage/phone_run.sh" ]] || die "run tests/fuzz/ops/run.sh phone-build first"
    local -a staged=()
    for b in "$stage"/*-*/; do
        [[ -f "$b/ops_replay" ]] && staged+=("$(basename "$b")")
    done
    echo "# 1. Push the files. The old files of the mixed ASan+UBSan build go first."
    echo "adb -s $PHONE shell 'rm -rf $d && mkdir -p $d/out $d/dsp'"
    echo "adb -s $PHONE push $stage/phone_run.sh $stage/cases.pack $d/"
    for b in "${staged[@]}"; do
        echo "adb -s $PHONE push $stage/$b $d/"
    done
    [[ -f "$stage/ubsan.supp" ]] && echo "adb -s $PHONE push $stage/ubsan.supp $d/"
    [[ -f "$stage/llvm-symbolizer" ]] && echo "adb -s $PHONE push $stage/llvm-symbolizer $d/"
    echo "adb -s $PHONE shell 'chmod 755 $d/*/ops_replay $d/llvm-symbolizer; sha256sum $d/llvm-symbolizer $d/*/ops_replay | cut -c1-16'"
    if [[ -f "$stage/libggml-htp-v79.so" ]]; then
        adsp="$d/dsp"
        echo "adb -s $PHONE push $stage/libggml-htp-v79.so $d/dsp/"
    fi
    echo
    if [[ $what == probe ]]; then
        echo "# 2. The probe: the first 3 cases on CPU and HTP0 with each build (one profile, one sanitizer)."
        echo "#    The ASan builds run under the ptrace tracer of ops_replay (--trace): it prints the pc, the"
        echo "#    module, the instruction, si_code, BTYPE and a backtrace of each fatal signal of each"
        echo "#    thread, also in a thread that blocks the signal. Before the cases, a thread self-test"
        echo "#    (one pthread, one std::thread) runs with the default ASan options and with three"
        echo "#    variants. The crash buffer of logcat shows a tombstone if debuggerd saw the signal."
        echo "#    The ubsan builds run two times: UBSAN_HALT=0 prints every report (the evidence), then"
        echo "#    UBSAN_HALT=1 shows that the entries of ubsan.supp match on the phone (no report for"
        echo "#    task #127; a stop is a report that no entry matches)."
        local asan_base="detect_leaks=0:halt_on_error=1:abort_on_error=0:external_symbolizer_path=$d/llvm-symbolizer"
        for b in "${staged[@]}"; do
            echo "# probe of $b"
            if [[ $b == *-asan ]]; then
                local v
                echo "adb -s $PHONE shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
                for v in "" ":detect_stack_use_after_return=0" ":use_sigaltstack=0" ":handle_sigill=1"; do
                    echo "adb -s $PHONE shell 'cd $d && timeout -s KILL 60 env LD_LIBRARY_PATH=$d/$b ASAN_OPTIONS=$asan_base$v ./$b/ops_replay --trace --selftest-threads >> out/selftest-$b.txt 2>&1; echo options=$asan_base$v rc=\$?'"
                done
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
    echo "# 2. The full runs: each build on CPU and HTP0 with the fusions of the app, then HTP0"
    echo "#    without the fusions. Repeat each run until its log shows 'all runs done'."
    for b in "${runs[@]}"; do
        echo "# $b: app"
        phone_run_cmd "$b" app CPU,HTP0 - 1 "$adsp" all 1
        echo "# $b: nofuse"
        phone_run_cmd "$b" nofuse HTP0 -nofuse 0 "$adsp" all 1
    done
    echo
    echo "# 3. Pull the results, then run: tests/fuzz/ops/run.sh compare"
    echo "mkdir -p $stage/pulled"
    echo "adb -s $PHONE pull $d/out $stage/pulled/"
    echo
    echo "# 4. Clean the phone work directory when the campaign is complete."
    echo "adb -s $PHONE shell 'rm -rf $d'"
}

compare() {
    build_oracle
    local stage="$MISC/phone"
    local -a results=("$@")
    if [[ ${#results[@]} -eq 0 ]]; then
        mapfile -t results < <(find "$stage/pulled" -name 'res-*.bin' | sort)
    fi
    [[ ${#results[@]} -gt 0 ]] || die "no result files: pull them first (run.sh phone-commands)"
    local -a args=(compare --pack "$stage/cases.pack" --findings "$MISC/findings-phone")
    local r
    for r in "${results[@]}"; do
        args+=(--results "$r")
    done
    mkdir -p "$MISC/logs"
    "$B_ORACLE/ops_oracle" "${args[@]}" | tee "$MISC/logs/phone-compare.txt"
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

# Build llvm-symbolizer for arm64 Android into $SYMBOLIZER. The sanitizer runtimes on the phone
# need it: without it a report frame has no function name, thus no function-level entry of
# tests/sanitizers/ubsan.supp can match it (rule R5). The source is a sparse, shallow clone of
# llvm-project at the tag of the host tablegen (the versions must agree). The tool is not part of
# the code under test, thus the NDK of the host (NDK_HOST) builds it.
build_symbolizer() {
    [[ -x $SYMBOLIZER ]] && { echo "fuzz-ops: $SYMBOLIZER exists"; return 0; }
    local tools="$MISC/tools" tblgen ver ndk=${NDK_HOST:-}
    tblgen=$(command -v llvm-tblgen) || die "llvm-tblgen is necessary on the host (the package llvm-devel or llvm)"
    ver=$("$tblgen" --version | rg -o '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)
    [[ -n $ver ]] || die "cannot read the version of $tblgen"
    if [[ -z $ndk ]]; then
        ndk=$(find "$HOME/Android/Sdk/ndk" -mindepth 1 -maxdepth 1 -type d 2> /dev/null | sort -V | tail -n 1)
    fi
    [[ -f $ndk/build/cmake/android.toolchain.cmake ]] || die "set NDK_HOST to an Android NDK on the host"
    mkdir -p "$tools"
    if [[ ! -d $tools/llvm-src/llvm/lib ]]; then
        rm -rf "$tools/llvm-src"
        git clone --quiet --depth 1 --branch "llvmorg-$ver" --filter=blob:none --no-checkout \
            https://github.com/llvm/llvm-project.git "$tools/llvm-src"
        git -C "$tools/llvm-src" sparse-checkout set --cone llvm/cmake llvm/include llvm/lib \
            llvm/tools/llvm-symbolizer llvm/tools/llvm-config llvm/utils llvm/bindings llvm/runtimes \
            llvm/projects llvm/resources cmake third-party
        git -C "$tools/llvm-src" checkout --quiet "llvmorg-$ver"
    fi
    nice -n 10 cmake -S "$tools/llvm-src/llvm" -B "$tools/sym-build" -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE="$ndk/build/cmake/android.toolchain.cmake" -DANDROID_ABI=arm64-v8a \
        -DANDROID_PLATFORM=android-34 -DANDROID_STL=c++_static -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_TARGETS_TO_BUILD=AArch64 -DLLVM_HOST_TRIPLE=aarch64-linux-android \
        -DLLVM_TABLEGEN="$tblgen" -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_DOCS=OFF -DLLVM_INCLUDE_UTILS=OFF -DLLVM_BUILD_UTILS=OFF \
        -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_TERMINFO=OFF \
        -DLLVM_ENABLE_LIBEDIT=OFF -DLLVM_ENABLE_LIBPFM=OFF -DLLVM_ENABLE_CURL=OFF -DLLVM_ENABLE_HTTPLIB=OFF \
        -DLLVM_ENABLE_BINDINGS=OFF > "$tools/sym-configure.log" 2>&1 \
        || die "the configuration of llvm-symbolizer failed, refer to $tools/sym-configure.log"
    nice -n 10 ninja -C "$tools/sym-build" -j"$BUILD_JOBS" llvm-symbolizer > "$tools/sym-ninja.log" 2>&1 \
        || die "the build of llvm-symbolizer failed, refer to $tools/sym-ninja.log"
    "$ndk/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" -o "$SYMBOLIZER" "$tools/sym-build/bin/llvm-symbolizer"
    echo "fuzz-ops: $SYMBOLIZER ($(sha256sum "$SYMBOLIZER" | cut -c1-16))"
}

# Build the libraries of the app and llama-bench for arm64 Android from the private llama.cpp tree
# of a fix (the parent of FUZZ_OPS_SRC), with the flags of the preset
# arm64-android-snapdragon-release and the reproducibility flags of scripts/build-native.sh, for
# the phone check of the fix (landing rule L5). It never writes build/native or the jniLibs of the
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
    echo "fuzz-ops: build the app libraries and llama-bench from $rel into $bdir"
    container_run "$SNAPDRAGON_IMAGE" bash -euo pipefail -c "
repro='-ffile-prefix-map=/workspace=. -fdebug-prefix-map=/workspace=. -Werror=date-time'
flags=\$(python3 -c 'import json, sys; p = [x for x in json.load(open(sys.argv[1]))[\"configurePresets\"] if x[\"name\"] == \"arm64-android-snapdragon\"][0]; print(p[\"cacheVariables\"][\"CMAKE_C_FLAGS\"])' $rel/CMakeUserPresets.json)
export CFLAGS=\"\$repro\" CXXFLAGS=\"\$repro\"
cmake -S $rel --preset arm64-android-snapdragon-release -B $bdir \
    -DLLAMA_BUILD_NUMBER=$LLAMA_BUILD_NUMBER -DLLAMA_BUILD_COMMIT=${LLAMA_COMMIT:0:7} \
    -DCMAKE_C_FLAGS=\"\$flags \$repro\" -DCMAKE_CXX_FLAGS=\"\$flags \$repro\"
cmake --build $bdir -j$BUILD_JOBS --target $LLAMA_LIBS llama-bench
" > "$REPO_ROOT/$bdir.log" 2>&1 || die "the build failed, refer to $REPO_ROOT/$bdir.log"
    rm -rf "$out"
    mkdir -p "$out"
    for lib in $LLAMA_LIBS; do
        cp -f "$REPO_ROOT/$bdir/bin/lib$lib.so" "$out/"
    done
    cp -f "$REPO_ROOT/$bdir/bin/llama-bench" "$out/"
    (cd "$out" && sha256sum ./* > SHA256SUMS)
    cat "$out/SHA256SUMS"
}

# Stage and print the phone check of a fix (landing rule L5): the shipped libraries (before)
# against the libraries of the fix (after, from phone-libs), in one command set.
#   1. The op pack on CPU and HTP0 with the release none replay driver on each library set. The
#      command "ops_oracle same" then shows if the outputs are bit-identical.
#   2. llama-bench, tg on the 4B model and pp on the 2B model, on the CPU, 5 repetitions, in the
#      order before, after, before, after, with the thermal status, the clock caps and the battery
#      between the runs.
phone_check_commands() {
    [[ -n $TAG ]] || die "phone-check-commands needs FUZZ_OPS_TAG (the libraries of phone-libs)"
    local stage="$MISC/phone" d=$PHONE_DIR lib side
    local after="$stage/libs-$TAG"
    [[ -f $after/llama-bench && -f $stage/release-none/ops_replay && -f $stage/cases.pack ]] \
        || die "run phone-build and phone-libs first"
    check_shipped
    for side in before after; do
        rm -rf "$stage/check-$TAG-$side"
        mkdir -p "$stage/check-$TAG-$side"
        cp -f "$stage/release-none/ops_replay" "$stage/check-$TAG-$side/"
        for lib in $LLAMA_LIBS; do
            if [[ $side == before ]]; then
                cp -f "$SHIPPED_LIBS/lib$lib.so" "$stage/check-$TAG-$side/"
            else
                cp -f "$after/lib$lib.so" "$stage/check-$TAG-$side/"
            fi
        done
    done
    cp -f "$after/llama-bench" "$stage/check-$TAG-after/"
    local status="adb -s $PHONE shell 'dumpsys thermalservice | grep \"Thermal Status\"; cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq; dumpsys battery | grep -E \"powered|status|temperature\"'"
    local m4=/data/local/tmp/qwen/models/Qwen3.5-4B-Q8_0.gguf m2=/data/local/tmp/qwen/models/Qwen3.5-2B-Q8_0.gguf
    local bench="$d/check-$TAG-after/llama-bench -dev none -ngl 0 -t 6 -fa 1 -r 5 -o md"
    echo "# The phone check of FUZZ_OPS_TAG=$TAG (landing rule L5). Do not run on the charger."
    echo "adb -s $PHONE shell 'mkdir -p $d/out'"
    echo "adb -s $PHONE push $stage/check-$TAG-before $stage/check-$TAG-after $d/"
    echo "adb -s $PHONE shell 'chmod 755 $d/check-$TAG-*/ops_replay $d/check-$TAG-after/llama-bench; cd $d && sha256sum check-$TAG-*/*.so check-$TAG-after/llama-bench | cut -c1-16'"
    echo "# 1. The op pack with each library set (the first 600 cases, CPU and HTP0)."
    for side in before after; do
        phone_run_cmd "check-$TAG-$side" libcheck CPU,HTP0 - 1 "$ADSP_DIR" 600 1
    done
    echo "# 2. llama-bench on the CPU: tg on the 4B model, then pp on the 2B model."
    local what model args n
    for what in tg pp; do
        if [[ $what == tg ]]; then model=$m4; args="-p 0 -n 32"; else model=$m2; args="-p 128 -n 0"; fi
        n=0
        for side in before after before after; do
            n=$((n + 1))
            echo "$status"
            echo "adb -s $PHONE shell 'cd $d && timeout -s KILL 100 env LD_LIBRARY_PATH=$d/check-$TAG-$side ADSP_LIBRARY_PATH=$ADSP_DIR $bench -m $model $args 2> out/bench-$TAG-$what-$side-$n.log; echo $what $side $n rc=\$?'"
        done
    done
    echo "$status"
    echo "adb -s $PHONE shell 'pgrep -a llama; pgrep -a ops_replay'"
    echo "# 3. Pull, then: build/fuzz/ops/oracle/ops_oracle same PULLED/res-check-$TAG-before-libcheck.bin PULLED/res-check-$TAG-after-libcheck.bin"
    echo "mkdir -p $stage/pulled-check-$TAG"
    echo "adb -s $PHONE pull $d/out $stage/pulled-check-$TAG/"
}

# Build the ASan runtime for arm64 Android from compiler-rt of the LLVM tag of the host tablegen,
# into $ASAN_RUNTIME (task #176). The runtime of NDK r29 signs the return address in its prctl
# interceptor. Bionic calls prctl(PR_PAC_RESET_KEYS, PR_PAC_APIAKEY) at the start of each thread,
# thus the interceptor authenticates its return address with a new key, and FEAT_FPAC (SM8750)
# traps with SIGILL. compiler-rt 22.1.8 builds that interceptor without PAC. The NDK of the host
# builds the runtime; the builtins, libunwind and libc++abi of that NDK are linked into it.
build_asan_runtime() {
    [[ -f $ASAN_RUNTIME ]] && { echo "fuzz-ops: $ASAN_RUNTIME exists"; return 0; }
    build_symbolizer   # the same sparse LLVM clone
    local tools="$MISC/tools" ndk=${NDK_HOST:-} tc
    if [[ -z $ndk ]]; then
        ndk=$(find "$HOME/Android/Sdk/ndk" -mindepth 1 -maxdepth 1 -type d 2> /dev/null | sort -V | tail -n 1)
    fi
    tc="$ndk/toolchains/llvm/prebuilt/linux-x86_64"
    [[ -d $tools/llvm-src/compiler-rt ]] || git -C "$tools/llvm-src" sparse-checkout add compiler-rt
    local unwind
    unwind=$(find "$tc/lib/clang" -path '*/lib/linux/aarch64/libunwind.a' | head -n 1)
    [[ -f $unwind ]] || die "no aarch64 libunwind.a in $tc/lib/clang"
    nice -n 10 cmake -S "$tools/llvm-src/compiler-rt" -B "$tools/asan-rt-build" -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE="$ndk/build/cmake/android.toolchain.cmake" -DANDROID_ABI=arm64-v8a \
        -DANDROID_PLATFORM=android-24 -DANDROID_STL=none -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-mbranch-protection=standard" -DCMAKE_CXX_FLAGS="-mbranch-protection=standard" \
        -DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON -DCOMPILER_RT_BUILD_BUILTINS=OFF -DCOMPILER_RT_BUILD_SANITIZERS=ON \
        -DCOMPILER_RT_SANITIZERS_TO_BUILD=asan -DCOMPILER_RT_BUILD_XRAY=OFF -DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
        -DCOMPILER_RT_BUILD_PROFILE=OFF -DCOMPILER_RT_BUILD_MEMPROF=OFF -DCOMPILER_RT_BUILD_ORC=OFF \
        -DCOMPILER_RT_BUILD_CTX_PROFILE=OFF -DCOMPILER_RT_BUILD_GWP_ASAN=OFF -DCOMPILER_RT_INCLUDE_TESTS=OFF \
        -DLLVM_CMAKE_DIR="$tools/llvm-src/llvm/cmake/modules" -DCOMPILER_RT_USE_BUILTINS_LIBRARY=ON \
        -DSANITIZER_CXX_ABI=libcxxabi -DSANITIZER_USE_STATIC_CXX_ABI=ON -DCOMPILER_RT_UNWINDER_LINK_LIBS="$unwind" \
        > "$tools/asan-rt-configure.log" 2>&1 || die "the configuration of the ASan runtime failed, refer to $tools/asan-rt-configure.log"
    nice -n 10 ninja -C "$tools/asan-rt-build" -j"$BUILD_JOBS" lib/linux/libclang_rt.asan-aarch64-android.so \
        > "$tools/asan-rt-ninja.log" 2>&1 || die "the build of the ASan runtime failed, refer to $tools/asan-rt-ninja.log"
    cp -f "$tools/asan-rt-build/lib/linux/libclang_rt.asan-aarch64-android.so" "$ASAN_RUNTIME"
    echo "fuzz-ops: $ASAN_RUNTIME ($(sha256sum "$ASAN_RUNTIME" | cut -c1-16))"
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
        phone-libs)     phone_libs ;;
        phone-check-commands) phone_check_commands ;;
        bounds)         build_oracle; "$B_ORACLE/ops_oracle" bounds ;;
        help | -h | --help) usage ;;
        *)              usage; die "unknown mode: $mode" ;;
    esac
}

main "$@"
