#!/usr/bin/env bash
# The fuzz campaign of the host part of the Hexagon backend (hexhost).
# Run it with no argument to see the usage text.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
AREA=hexhost
ALL_TARGETS="kparams graph repack envparse dirty"
X86_CONFIGS="none asan ubsan tsan msan"
PHONE_CONFIGS="none asan hwasan ubsan"
ALL_PROFILES="debug release"

BUDGET=${BUDGET:-600}
JOBS=${JOBS:-4}
BUILD_JOBS=${BUILD_JOBS:-8}
KNOWN=${FUZZ_KNOWN:-1}
PROFILES=$ALL_PROFILES
LLAMA_DIR=${HEXHOST_LLAMA_DIR:-$REPO/third_party/llama.cpp}
UBSAN_SUPP="$REPO/tests/sanitizers/ubsan.supp"
PHONE=${PHONE:-192.168.14.130:5555}
PHONE_DIR=${PHONE_DIR:-/data/local/tmp/qwen/fuzz/hexhost}
PHONE_SECONDS=${PHONE_SECONDS:-80}
PHONE_RANDOM=${PHONE_RANDOM:-60}
DSP_LIB=${DSP_LIB:-$REPO/build/native/llama/ggml/src/ggml-hexagon/libggml-htp-v79.so}
SHIPPED_LIBS="$REPO/android/snapdragon/jniLibs/arm64-v8a"
SHIPPED_HASHES="$REPO/build/hashes-native.txt"
# The Android ASan runtime of compiler-rt 22 (tests/sanitizers/build-asan-android-runtime.sh, task
# #176). The runtime of the NDK stops each new thread with SIGILL on the SM8750.
ASAN_RT_DIR=${ASAN_RT_DIR:-$REPO/build/fuzz/asan-android-runtime}
ASAN_RT=libclang_rt.asan-aarch64-android.so

# The ids of the known findings (fake_dsp.cpp: violation). In the fuzz mode the harness keeps these
# conditions off (HEXHOST_IGNORE), thus the fuzzers look for new defects. The test mode does not
# set them, thus each regression input of an open finding fails.
KNOWN_IDS="env-profile-empty,env-exception,env-devices-range,repack-nonfinite-scale,env-int-overflow"

# The host switches of the phone runs: a name and the GGML_HEXAGON_* variables of each run
PHONE_RUNS=(
    "default:"
    "nofusion:GGML_HEXAGON_OPFUSION=0"
    "batch4:GGML_HEXAGON_OPBATCH=4"
    "batch1q1:GGML_HEXAGON_OPBATCH=1 GGML_HEXAGON_OPQUEUE=1"
    "nocache:GGML_HEXAGON_GRAPHCACHE=0 GGML_HEXAGON_BATCHCACHE=0"
    "verify:GGML_HEXAGON_BATCHCACHE=2"
)

usage() {
    cat << EOF
Usage: tests/fuzz/hexhost/run.sh <test|fuzz> <config> [--profile P] [--budget-seconds N] [--jobs N] [TARGET...]
       tests/fuzz/hexhost/run.sh phone-build <config> [--profile P]
       tests/fuzz/hexhost/run.sh phone-commands <config> [--profile P]

The fuzz targets of the host part of the Hexagon backend (ggml-hexagon.cpp) on x86 with a fake
DSP, and a phone driver that runs the same graphs on HTP0 and on the CPU backend of the phone.
Each pair of a profile and a config has its own build directory:
build/fuzz/hexhost-<profile>-<config> (x86) and build/fuzz/hexhost-android-<profile>-<config>.

Modes:
  test <config>         Build, then run each target one time on its seeds (corpus/<target>) and
                        on its regression inputs (regress/<target>), with no mutation. A finding
                        gives a nonzero exit code.
  fuzz <config>         Build, then fuzz each target for the budget. A crash does not stop the
                        target: it starts again until the budget ends.
  phone-build <config>  Build the phone driver (phone/driver.cpp) and the ggml libraries for arm64
                        Android in the Snapdragon container, and stage the phone files.
  phone-commands <config>
                        Print the adb commands of the phone runs. This script never runs adb.
  cpu-asan, cpu-tsan    The old names of "fuzz asan" and "fuzz tsan".

The x86 configs: $X86_CONFIGS (one sanitizer each).
The phone configs: $PHONE_CONFIGS (the NDK has no TSan and no MSan).
The targets: $ALL_TARGETS (all when no target is given).

Options:
  --profile P         debug or release (default: the two, one after the other)
  --budget-seconds N  The fuzz time of each target (default $BUDGET s = 10 minutes)
  --jobs N            The targets that run at the same time (default $JOBS)

Environment:
  FUZZ_KNOWN=0        Also fuzz the conditions of the known findings (default 1: keep them off)
  BUILD_JOBS          The parallel build jobs (default $BUILD_JOBS)
  HEXHOST_LLAMA_DIR   The llama.cpp tree (default: the submodule third_party/llama.cpp with the
                      patch series applied)
  FUZZ_MSAN_PREFIX    The MSan libc++ (default build/fuzz/msan-libcxx/install)
  PHONE, PHONE_DIR    The phone serial ($PHONE) and the work directory on the phone ($PHONE_DIR)
  PHONE_SECONDS       The driver time of each phone run (default $PHONE_SECONDS s, under the 100 s kill)
  PHONE_RANDOM        The random inputs of each phone run after the corpus (default $PHONE_RANDOM)
  DSP_LIB             The DSP library for the phone (default build/native/.../libggml-htp-v79.so)

Outputs: build/fuzz/hexhost-<profile>-<config>/{results.jsonl,runs/<target>/}. Each run adds
one JSON line for each target to results.jsonl.
EOF
}

die() {
    echo "run.sh: $*" >&2
    exit 1
}

# The build directory of the profile $1 and the x86 config $2
x86_dir() {
    echo "$REPO/build/fuzz/$AREA-$1-$2"
}

# Configure and build the x86 targets of one profile and one config.
build_x86() {
    local prof=$1 cfg=$2 dir
    dir=$(x86_dir "$prof" "$cfg")
    if [[ $cfg == ubsan && ! -f $UBSAN_SUPP ]]; then
        die "the shared file $UBSAN_SUPP does not exist. The ubsan runs stop at the known report of ggml.c:7447 (task #125) without it."
    fi
    mkdir -p "$dir"
    CC=clang CXX=clang++ cmake -S "$HERE" -B "$dir" -G Ninja -DCMAKE_BUILD_TYPE=None \
        -DFUZZ_PROFILE="$prof" -DFUZZ_SANITIZER="$cfg" -DHEXHOST_LLAMA_DIR="$LLAMA_DIR" \
        ${FUZZ_MSAN_PREFIX:+-DFUZZ_MSAN_PREFIX="$FUZZ_MSAN_PREFIX"} > "$dir/configure.log" 2>&1 \
        || die "the configure of $dir failed. Read $dir/configure.log."
    nice -n 10 cmake --build "$dir" -j"$BUILD_JOBS" > "$dir/build.log" 2>&1 \
        || die "the build of $dir failed. Read $dir/build.log."
}

# Print the environment of a run: the shared sanitizer options (tests/sanitizers/env.sh) and, for
# the fuzz mode ($2 = 1), the switches of the known findings.
run_env() {
    local cfg=$1 fuzz=$2 v
    (
        # shellcheck source=../../sanitizers/env.sh
        source "$REPO/tests/sanitizers/env.sh"
        sanitizer_env "$cfg"
        for v in ASAN_OPTIONS LSAN_OPTIONS UBSAN_OPTIONS TSAN_OPTIONS MSAN_OPTIONS; do
            [[ -n ${!v:-} ]] && echo "$v=${!v}"
        done
        true
    )
    echo "GGML_NO_BACKTRACE=1"
    if [[ $fuzz == 1 && $KNOWN == 1 ]]; then
        echo "HEXHOST_IGNORE=$KNOWN_IDS"
    fi
}

# Write one JSON line for a target to results.jsonl. The crash files are the remaining arguments.
result_line() {
    local dir=$1 target=$2 prof=$3 cfg=$4 mode=$5 seconds=$6 execs=$7 findings=$8
    shift 8
    local files
    files=$(printf '%s\n' "$@" | jq -R . | jq -sc 'map(select(length > 0))')
    jq -nc --arg area "$AREA" --arg target "$target" --arg profile "$prof" --arg sanitizer "$cfg" --arg mode "$mode" \
        --argjson seconds "$seconds" --argjson executions "$execs" --argjson findings "$findings" \
        --argjson crash_files "$files" \
        '{area: $area, target: $target, profile: $profile, sanitizer: $sanitizer, mode: $mode, seconds: $seconds,
          executions: $executions, findings: $findings, crash_files: $crash_files}' >> "$dir/results.jsonl"
}

# Fuzz one target for the budget. After a crash the target starts again with its corpus.
fuzz_one() {
    local prof=$1 cfg=$2 t=$3 dir
    dir=$(x86_dir "$prof" "$cfg")
    local out="$dir/runs/$t"
    mkdir -p "$out/corpus" "$out/artifacts" "$HERE/corpus/$t"
    local log="$out/log.txt"
    : > "$log"
    local -a envs
    mapfile -t envs < <(run_env "$cfg" 1)
    envs+=("FUZZ_ARTIFACT_DIR=$out/artifacts")
    local start=$SECONDS left starts=0 rc
    while :; do
        left=$(( BUDGET - (SECONDS - start) ))
        (( left > 5 && starts < 200 )) || break
        starts=$(( starts + 1 ))
        rc=0
        # The outer kill stops a hang that libFuzzer cannot stop (a TSan report can deadlock in the
        # death callback of libFuzzer). A kill is a finding: a hang record goes to the artifacts.
        timeout -s KILL $(( left + 180 )) env "${envs[@]}" nice -n 10 "$dir/fuzz_$t" "$out/corpus" "$HERE/corpus/$t" \
            -max_total_time="$left" -rss_limit_mb=4096 -malloc_limit_mb=4096 -timeout=60 -max_len=4096 \
            -artifact_prefix="$out/artifacts/" -print_final_stats=1 >> "$log" 2>&1 || rc=$?
        echo "run.sh: start $starts ended with code $rc after $(( SECONDS - start )) s" >> "$log"
        if [[ $rc == 137 ]]; then
            tail -n 40 "$log" > "$out/artifacts/hang-start-$starts.txt"
        fi
        [[ $rc == 0 ]] && break
    done
    # the executions: the sum of the last progress count of each start
    local execs=0 prev=0 n
    while read -r n; do
        (( n < prev )) && execs=$(( execs + prev ))
        prev=$n
    done < <(rg -o --no-line-number '^#[0-9]+' "$log" | tr -d '#')
    execs=$(( execs + prev ))
    local -a arts=()
    mapfile -t arts < <(fd -t f . "$out/artifacts" | sort)
    result_line "$dir" "$t" "$prof" "$cfg" fuzz "$(( SECONDS - start ))" "$execs" "${#arts[@]}" "${arts[@]}"
    echo "$AREA-$prof-$cfg $t: $(( SECONDS - start )) s, $starts starts, $execs executions, corpus $(fd -t f . "$out/corpus" | wc -l), crash files ${#arts[@]}"
}

# Run one target one time on each seed and each regression input, with no known finding kept off.
test_one() {
    local prof=$1 cfg=$2 t=$3 dir
    dir=$(x86_dir "$prof" "$cfg")
    local out="$dir/runs/$t"
    mkdir -p "$out"
    local log="$out/test-log.txt"
    : > "$log"
    local -a envs files=() failed=()
    mapfile -t envs < <(run_env "$cfg" 0)
    envs+=("FUZZ_ARTIFACT_DIR=$out")
    [[ -d "$HERE/corpus/$t" ]] && mapfile -t -O 0 files < <(fd -t f . "$HERE/corpus/$t" | sort)
    [[ -d "$HERE/regress/$t" ]] && mapfile -t -O "${#files[@]}" files < <(fd -t f . "$HERE/regress/$t" | sort)
    local start=$SECONDS f rc
    for f in "${files[@]}"; do
        rc=0
        # the outer kill: a hang is a finding (code 137)
        timeout -s KILL 300 env "${envs[@]}" nice -n 10 "$dir/fuzz_$t" -rss_limit_mb=4096 -malloc_limit_mb=4096 \
            -timeout=60 -artifact_prefix="$out/" "$f" >> "$log" 2>&1 || rc=$?
        echo "run.sh: $f gives the code $rc" >> "$log"
        [[ $rc != 0 ]] && failed+=("$f")
    done
    result_line "$dir" "$t" "$prof" "$cfg" test "$(( SECONDS - start ))" "${#files[@]}" "${#failed[@]}" "${failed[@]}"
    local names=""
    for f in "${failed[@]}"; do
        names+=" ${f#"$HERE"/}"
    done
    echo "$AREA-$prof-$cfg $t: ${#files[@]} inputs, ${#failed[@]} findings${names:+:$names}"
}

# Run the mode $1 (test or fuzz) with the config $2 on the targets that follow, JOBS at a time,
# for each profile. Gives the code 1 in the test mode when a target has a finding.
run_mode() {
    local mode=$1 cfg=$2
    shift 2
    [[ " $X86_CONFIGS " == *" $cfg "* ]] || die "the x86 config must be one of: $X86_CONFIGS"
    local targets="${*:-$ALL_TARGETS}" t prof dir summary bad=0
    for t in $targets; do
        [[ " $ALL_TARGETS " == *" $t "* ]] || die "no target $t. The targets: $ALL_TARGETS"
    done
    for prof in $PROFILES; do
        build_x86 "$prof" "$cfg"
        dir=$(x86_dir "$prof" "$cfg")
        summary="$dir/$mode-summary.txt"
        : > "$summary"
        for t in $targets; do
            while (( $(jobs -rp | wc -l) >= JOBS )); do
                sleep 5
            done
            if [[ $mode == fuzz ]]; then
                fuzz_one "$prof" "$cfg" "$t" >> "$summary" &
            else
                test_one "$prof" "$cfg" "$t" >> "$summary" &
            fi
        done
        wait
        cat "$summary"
        echo "run.sh: the JSON lines are in $dir/results.jsonl"
        if rg -q ', [1-9][0-9]* findings' "$summary"; then
            bad=1
        fi
    done
    [[ $mode == test && $bad == 1 ]] && return 1
    return 0
}

# ---- The phone

# The build directory of the phone build of the profile $1 and the config $2
phone_dir() {
    echo "$REPO/build/fuzz/$AREA-android-$1-$2"
}

# Build the phone driver and the ggml libraries of one profile and one config in the container, and
# stage the files: bin/, lib/ (the ggml libraries and the sanitizer runtime), dsp/, in/, ubsan.supp.
phone_build_one() {
    local prof=$1 cfg=$2 dir rel stage
    dir=$(phone_dir "$prof" "$cfg")
    rel=${dir#"$REPO"/}
    stage="$dir/stage"
    [[ -f $DSP_LIB ]] || die "the DSP library $DSP_LIB does not exist. Run scripts/build-native.sh first, or set DSP_LIB."
    if [[ $cfg == asan ]]; then
        (cd "$ASAN_RT_DIR" 2> /dev/null && sha256sum -c --quiet "$ASAN_RT.sha256") \
            || die "the ASan runtime $ASAN_RT_DIR/$ASAN_RT is missing or does not match its sha256. Run tests/sanitizers/build-asan-android-runtime.sh first."
    fi
    [[ $LLAMA_DIR == "$REPO"/* ]] || die "the llama.cpp tree $LLAMA_DIR is not inside the repository, thus the container cannot see it"
    mkdir -p "$dir/include/fuzzer"
    # FuzzedDataProvider.h is one header; the copy of the host clang is the same file as the NDK copy
    cp -f "$(clang -print-resource-dir)/include/fuzzer/FuzzedDataProvider.h" "$dir/include/fuzzer/"
    echo "run.sh: build the phone driver ($prof, $cfg) in $rel"
    container_run "$SNAPDRAGON_IMAGE" bash -euo pipefail -c "
cmake -S tests/fuzz/hexhost/phone -B $rel -G Ninja -DCMAKE_BUILD_TYPE=None \
    -DCMAKE_TOOLCHAIN_FILE=\$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 \
    -DFUZZ_PROFILE=$prof -DFUZZ_SANITIZER=$cfg \
    -DHEXHOST_LLAMA_DIR=/workspace/${LLAMA_DIR#"$REPO"/} -DHEXHOST_FUZZER_INCLUDE=/workspace/$rel/include \
    -DHEXAGON_SDK_ROOT=\$HEXAGON_SDK_ROOT -DHEXAGON_TOOLS_ROOT=\$HEXAGON_TOOLS_ROOT -DPREBUILT_LIB_DIR=android_aarch64
cmake --build $rel -j$BUILD_JOBS --target hexhost_phone
rm -rf $rel/runtime && mkdir -p $rel/runtime
case $cfg in
    hwasan) rt=libclang_rt.hwasan-aarch64-android.so ;;
    ubsan) rt=libclang_rt.ubsan_standalone-aarch64-android.so ;;
    *) rt= ;;
esac
if [[ -n \$rt ]]; then
    cp -f \$(find \$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt -name \$rt | head -n 1) $rel/runtime/
fi
" > "$dir.build.log" 2>&1 || die "the phone build failed. Read $dir.build.log."
    # The ASan runtime comes from the shared build (task #176), not from the NDK
    [[ $cfg == asan ]] && cp -f "$ASAN_RT_DIR/$ASAN_RT" "$dir/runtime/"

    rm -rf "$stage"
    mkdir -p "$stage/bin" "$stage/lib" "$stage/dsp" "$stage/in"
    cp -f "$dir/hexhost_phone" "$stage/bin/"
    if [[ $prof == release && $cfg == none ]]; then
        # The release "none" run loads the shipped libraries, checked against build/hashes-native.txt
        local lib want have
        # The shipped libggml.so also needs libggml-opencl.so (the app builds with GGML_OPENCL=ON)
        for lib in libggml.so libggml-base.so libggml-cpu.so libggml-hexagon.so libggml-opencl.so; do
            want=$(rg -F "  $lib" "$SHIPPED_HASHES" | cut -d' ' -f1)
            have=$(sha256sum "$SHIPPED_LIBS/$lib" | cut -d' ' -f1)
            [[ -n $want && $want == "$have" ]] || die "$SHIPPED_LIBS/$lib does not match $SHIPPED_HASHES"
            cp -f "$SHIPPED_LIBS/$lib" "$stage/lib/"
        done
    else
        fd -t f -e so . "$dir/ggml" -x cp -f {} "$stage/lib/"
    fi
    fd -t f -e so . "$dir/runtime" -x cp -f {} "$stage/lib/" 2> /dev/null || true
    cp -f "$DSP_LIB" "$stage/dsp/libggml-htp-v79.so"
    [[ -f $UBSAN_SUPP ]] && cp -f "$UBSAN_SUPP" "$stage/ubsan.supp"
    # The inputs: the seeds and the regression inputs of fuzz_graph, then the corpus of the x86 runs
    fd -t f . "$HERE/corpus/graph" "$HERE/regress/graph" -x cp -f {} "$stage/in/" 2> /dev/null || true
    local c
    for c in "$REPO"/build/fuzz/$AREA-*-asan/runs/graph/corpus; do
        [[ -d $c ]] && fd -t f -S -2k . "$c" | sort | head -n 150 | xargs -r cp -f -t "$stage/in/"
    done
    (cd "$stage" && fd -t f . | sort | xargs sha256sum > SHA256SUMS)
    echo "run.sh: staged $(fd -t f . "$stage/in" | wc -l) inputs and $(fd -t f . "$stage/lib" | wc -l) libraries in ${stage#"$REPO"/}"
}

phone_build() {
    local cfg=$1 prof
    [[ " $PHONE_CONFIGS " == *" $cfg "* ]] || die "the phone config must be one of: $PHONE_CONFIGS"
    # container_run and SNAPDRAGON_IMAGE. The file sets readonly variables, thus one source only.
    # shellcheck source=../../../scripts/lib.sh
    [[ -n ${SNAPDRAGON_IMAGE:-} ]] || source "$REPO/scripts/lib.sh"
    for prof in $PROFILES; do
        phone_build_one "$prof" "$cfg"
    done
}

# The sanitizer options of a phone run
phone_san_env() {
    local cfg=$1 d=$2
    case $cfg in
        asan) echo "ASAN_OPTIONS=halt_on_error=1:detect_leaks=0:abort_on_error=1" ;;
        hwasan) echo "HWASAN_OPTIONS=halt_on_error=1:abort_on_error=1" ;;
        ubsan) echo "UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1:report_error_type=1:suppressions=$d/ubsan.supp" ;;
        *) echo "" ;;
    esac
}

phone_commands() {
    local cfg=$1 prof stage d run name vars san tag
    [[ " $PHONE_CONFIGS " == *" $cfg "* ]] || die "the phone config must be one of: $PHONE_CONFIGS"
    for prof in $PROFILES; do
        stage="$(phone_dir "$prof" "$cfg")/stage"
        [[ -f "$stage/bin/hexhost_phone" ]] || die "run tests/fuzz/hexhost/run.sh phone-build $cfg --profile $prof first"
        d="$PHONE_DIR/$prof-$cfg"
        san=$(phone_san_env "$cfg" "$d")
        echo "# ==== $prof $cfg: push the files ($(fd -t f . "$stage" | wc -l) files)"
        echo "adb -s $PHONE shell 'rm -rf $d && mkdir -p $d/out'"
        echo "adb -s $PHONE push ${stage#"$REPO"/}/. $d/"
        echo "adb -s $PHONE shell 'chmod 755 $d/bin/hexhost_phone'"
        if [[ $cfg == asan ]]; then
            # Task #176: a failure of the thread start is a failure of the environment, not a finding
            echo "# If this line does not give \"thread self-test ok\", stop: record an environment failure."
            echo "adb -s $PHONE shell 'cd $d && timeout -s KILL 30 env LD_LIBRARY_PATH=$d/lib $san ./bin/hexhost_phone --selftest-threads'"
        fi
        for run in "${PHONE_RUNS[@]}"; do
            name=${run%%:*}
            vars=${run#*:}
            tag="$prof-$cfg-$name"
            echo "# ---- $tag"
            echo "adb -s $PHONE shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
            echo "adb -s $PHONE shell 'cd $d && mkdir -p out/save-$name && timeout -s KILL 100 env LD_LIBRARY_PATH=$d/lib ADSP_LIBRARY_PATH=$d/dsp $san $vars ./bin/hexhost_phone --seconds $PHONE_SECONDS --random $PHONE_RANDOM --save out/save-$name in > out/$name.txt 2> out/$name.err; echo exit=\$? >> out/$name.txt'"
            echo "adb -s $PHONE shell 'dumpsys thermalservice | grep \"Thermal Status\"'"
            echo "adb -s $PHONE shell 'pgrep -a hexhost_phone; tail -n 2 $d/out/$name.txt'"
        done
        echo "# ==== $prof $cfg: pull the results"
        echo "adb -s $PHONE pull $d/out ${stage%/stage}/phone-out"
    done
}

main() {
    [[ $# -ge 1 ]] || { usage; exit 2; }
    local mode=$1
    shift
    case $mode in
        cpu-asan) mode=fuzz; set -- asan "$@" ;;
        cpu-tsan) mode=fuzz; set -- tsan "$@" ;;
        -h|--help|help) usage; exit 0 ;;
        *) ;;
    esac
    local cfg=${1:-}
    [[ -n $cfg ]] || { usage; exit 2; }
    shift
    local -a targets=()
    while [[ $# -gt 0 ]]; do
        case $1 in
            --budget-seconds) BUDGET=$2; shift 2 ;;
            --jobs) JOBS=$2; shift 2 ;;
            --profile) PROFILES=$2; shift 2 ;;
            -*) die "unknown option $1" ;;
            *) targets+=("$1"); shift ;;
        esac
    done
    [[ $BUDGET =~ ^[0-9]+$ && $JOBS =~ ^[1-9][0-9]*$ ]] || die "--budget-seconds and --jobs take positive numbers"
    [[ " $ALL_PROFILES " == *" $PROFILES "* || $PROFILES == "$ALL_PROFILES" ]] || die "--profile takes debug or release"
    case $mode in
        test|fuzz) run_mode "$mode" "$cfg" "${targets[@]}" ;;
        phone-build) phone_build "$cfg" ;;
        phone-commands) phone_commands "$cfg" ;;
        *) usage; exit 2 ;;
    esac
}

main "$@"
