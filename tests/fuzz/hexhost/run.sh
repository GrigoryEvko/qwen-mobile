#!/usr/bin/env bash
# The fuzz targets of the host part of the Hexagon backend (hexhost).
# Run it with no argument to see the usage text.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=../../sanitizers/fuzz-lib.sh
source "$HERE/../../sanitizers/fuzz-lib.sh"

REPO=$FUZZ_REPO
AREA=hexhost
ALL_TARGETS="kparams graph repack envparse dirty mmap"
X86_CONFIGS=$FUZZ_HOST_CONFIGS
PHONE_CONFIGS=$FUZZ_PHONE_CONFIGS
ALL_PROFILES=$FUZZ_PROFILES

BUDGET=$FUZZ_BUDGET
JOBS=$FUZZ_JOBS
BUILD_JOBS=$FUZZ_BUILD_JOBS
KNOWN=${FUZZ_KNOWN:-1}
PROFILES=$ALL_PROFILES
# The llama.cpp tree: HEXHOST_LLAMA_DIR, or a private copy of the patched tree of HEAD that each
# build refreshes first (refresh_llama)
LLAMA_DIR=${HEXHOST_LLAMA_DIR:-$REPO/build/fuzz/$AREA/llama-src}
LLAMA_COPY=1
[[ -n ${HEXHOST_LLAMA_DIR:-} ]] && LLAMA_COPY=0
LLAMA_FRESH=0
# The suffix of the x86 and graphs build directories of a private tree (HEXHOST_BUILD_TAG)
BUILD_TAG=${HEXHOST_BUILD_TAG:-}
# The test mode and the graphs mode give AEE_EINTERRUPTED to each Nth dspqueue_read and each Nth
# dspqueue_write (HEXHOST_EINTR of common/fake_dsp.h), as a signal to the thread of the host does on
# the phone. The host must do such a call again. 0 gives no such code.
TEST_EINTR=${HEXHOST_TEST_EINTR:-3}
UBSAN_SUPP="$FUZZ_SHARED_DIR/ubsan.supp"
PHONE=$FUZZ_PHONE_SERIAL
PHONE_DIR=${PHONE_DIR:-/data/local/tmp/qwen/fuzz/hexhost}
PHONE_SECONDS=${PHONE_SECONDS:-80}
PHONE_RANDOM=${PHONE_RANDOM:-60}
DSP_LIB=${DSP_LIB:-}
SHIPPED_LIBS="$REPO/android/snapdragon/jniLibs/arm64-v8a"
SHIPPED_HASHES="$REPO/build/hashes-native.txt"

# The names of the checks (fake_dsp.cpp: violation) that detect a defect of the backend with no
# fix in the patch series. In the fuzz mode the harness keeps these conditions off (HEXHOST_IGNORE),
# thus the fuzzers look for other defects. The test mode does not set them, thus each regression
# input of such a defect fails. The list is comma-separated, as HEXHOST_IGNORE reads it. The list is
# empty: the patch series has a fix for each defect that a check of the harness found.
KNOWN_IDS=""

# The known properties of the NPU in the phone runs. The phone driver compares HTP0 with the CPU
# backend of the phone, thus each property shows as a difference:
#
#   stale L2 line   A DMA transfer of the DSP bypasses the L2 cache. The dirty range tracker
#                   (htp/htp-tensor.c) writes back and invalidates lines with a dccleaninva loop,
#                   which does not reach the lines of L2 on the v79. Thus inside one op batch, an
#                   op can read an old line of L2 after a DMA write, or the write-back of an old
#                   line can go over the bytes of a DMA write. The op then gets the value of an
#                   earlier op, thus the difference is large. A batch of one op cannot have this
#                   property, because each batch starts and ends with a full write-back and
#                   invalidate of the data cache. Thus each run replays the random inputs that it
#                   saved with GGML_HEXAGON_OPBATCH=1 (the file NAME-batch1.txt), and the run
#                   batch1q1 runs the corpus in the same mode. An input that differs in a run and
#                   passes with one op for each batch has this property. regress/graph holds five
#                   such inputs (l2-stale-line-*.bin). On the model graphs of the app, the logits
#                   with coherent DMA equal the logits of the shipped library. A DMA transfer
#                   through L2 costs prefill time, thus the property stays.
#   flush to zero   The HVX and HMX kernels give 0 for a result below the smallest normal FP16
#                   value, where the CPU keeps a very small float. The driver accepts a
#                   difference below that value (k_abs_tol of phone/driver.cpp).

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
       tests/fuzz/hexhost/run.sh graphs
       tests/fuzz/hexhost/run.sh phone-build <config> [--profile P]
       tests/fuzz/hexhost/run.sh phone-commands <config> [--profile P]

The fuzz targets of the host part of the Hexagon backend (ggml-hexagon.cpp) on x86 with a fake
DSP, and a phone driver that runs the same graphs on HTP0 and on the CPU backend of the phone.
Each pair of a profile and a config has its own build directory:
build/fuzz/hexhost-<profile>-<config> (x86) and build/fuzz/hexhost-android-<profile>-<config>.

Modes:
  test <config>         Build, then run each target one time on its seeds (corpus/<target>) and
                        on its regression inputs (regress/<target>), with no mutation. Each third
                        dspqueue_read and dspqueue_write gives AEE_EINTERRUPTED (HEXHOST_TEST_EINTR).
                        Then the queue-cancel check: fuzz_graph with HEXHOST_EINTR=cancel:2 on the
                        graph inputs, where the host must abort after a permanent failure of its
                        queue and not do the call again and again. Then run the graphs mode. A
                        finding gives a nonzero exit code.
  graphs                Build hexhost_graphs (graphs/graphs.cpp, no sanitizer) and run the paths
                        of the app on the 2B and the 4B Q8_0 of weights/gguf: decode, decode with
                        4 recurrent state snapshots, prefill 512, the MTP draft step and the image
                        turn at 576 and 768 image tokens. A run fails when a llama_decode fails, a
                        node of a later split reads the state tail of a fused GDN state chain, a
                        decode path does not fuse the GDN conv step, or an image turn does not fuse
                        the attention input of each layer or a LayerNorm of the vision encoder.
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
  FUZZ_KNOWN=0        Also fuzz the conditions of the checks of KNOWN_IDS (default 1: keep them off)
  BUILD_JOBS          The parallel build jobs (default $BUILD_JOBS)
  HEXHOST_LLAMA_DIR   The llama.cpp tree (default: build/fuzz/hexhost/llama-src, a copy of the
                      patched tree of HEAD that tests/sanitizers/llama-copy.sh makes or
                      refreshes at the start of each build)
  HEXHOST_BUILD_TAG   A suffix of the x86 and graphs build directories, for a private tree:
                      build/fuzz/hexhost-<profile>-<config>-<tag> and build/fuzz/hexhost-graphs-<tag>
  HEXHOST_TEST_EINTR  N: the test and graphs modes give AEE_EINTERRUPTED to each Nth
                      dspqueue_read and each Nth dspqueue_write (default $TEST_EINTR, 0: none)
  FUZZ_MSAN_PREFIX    The MSan libc++ (default build/fuzz/msan-libcxx/install)
  FUZZ_PHONE_SERIAL   The phone serial ($PHONE)
  PHONE_DIR           The work directory on the phone ($PHONE_DIR)
  PHONE_SECONDS       The driver time of each phone run (default $PHONE_SECONDS s, under the 100 s kill)
  PHONE_RANDOM        The random inputs of each phone run after the corpus (default $PHONE_RANDOM)
  DSP_LIB             The DSP library for the phone. By default, the release none build stages the
                      shipped library (checked against build/hashes-native.txt), and the other
                      builds make libggml-htp-v79.so from the llama.cpp tree with the release
                      preset of the app, thus the DSP code is the code of the host libraries.

Outputs: build/fuzz/hexhost-<profile>-<config>/{results.jsonl,runs/<target>/}. Each run adds
one JSON line for each target to results.jsonl.
EOF
}

# The build directory of the profile $1 and the x86 config $2
x86_dir() {
    fuzz_build_dir "$AREA" "$1" "$2" "$BUILD_TAG"
}

# Make or refresh the private copy of llama.cpp, one time for each run of this script, when
# HEXHOST_LLAMA_DIR is not set. The copy comes from the git objects of HEAD, thus a landing during a
# build does not change the tree of the build, and a change of the series goes into the next build.
refresh_llama() {
    [[ $LLAMA_COPY == 1 && $LLAMA_FRESH == 0 ]] || return 0
    fuzz_llama_copy "$LLAMA_DIR"
    LLAMA_FRESH=1
}

# Configure and build the x86 targets of one profile and one config.
build_x86() {
    local prof=$1 cfg=$2 dir
    refresh_llama
    dir=$(x86_dir "$prof" "$cfg")
    if [[ $cfg == ubsan && ! -f $UBSAN_SUPP ]]; then
        fuzz_die "the shared file $UBSAN_SUPP does not exist. The UBSan runtime of the ubsan runs reads it."
    fi
    mkdir -p "$dir"
    CC=clang CXX=clang++ cmake -S "$HERE" -B "$dir" -G Ninja -DCMAKE_BUILD_TYPE=None \
        -DFUZZ_PROFILE="$prof" -DFUZZ_SANITIZER="$cfg" -DHEXHOST_LLAMA_DIR="$LLAMA_DIR" \
        ${FUZZ_MSAN_PREFIX:+-DFUZZ_MSAN_PREFIX="$FUZZ_MSAN_PREFIX"} > "$dir/configure.log" 2>&1 \
        || fuzz_die "the configure of $dir failed. Read $dir/configure.log."
    nice -n 10 cmake --build "$dir" -j"$BUILD_JOBS" > "$dir/build.log" 2>&1 \
        || fuzz_die "the build of $dir failed. Read $dir/build.log."
}

# Print the environment of a run: the shared sanitizer options
# (tests/sanitizers/fuzz-lib.sh) and, for the fuzz mode ($2 = 1), the checks of KNOWN_IDS in
# HEXHOST_IGNORE, or for the test mode ($2 = 0), the interrupts of TEST_EINTR.
run_env() {
    local cfg=$1 fuzz=$2
    fuzz_sanitizer_assignments "$cfg"
    echo "GGML_NO_BACKTRACE=1"
    if [[ $fuzz == 1 && $KNOWN == 1 ]]; then
        echo "HEXHOST_IGNORE=$KNOWN_IDS"
    fi
    if [[ $fuzz == 0 ]]; then
        echo "HEXHOST_EINTR=$TEST_EINTR"
    fi
}

# Fuzz one target for the budget. After a crash the target starts again with its corpus.
fuzz_one() {
    local prof=$1 cfg=$2 t=$3 dir
    dir=$(x86_dir "$prof" "$cfg")
    local out="$dir/runs/$t"
    mkdir -p "$out/corpus" "$out/artifacts"
    local log="$out/log.txt"
    : > "$log"
    # A file of the artifacts directory that is not newer than this mark comes from an earlier
    # run, thus it is not a finding of this run.
    local mark="$out/.fuzz-start"
    touch "$mark"
    local -a envs seeds=() d
    mapfile -t envs < <(run_env "$cfg" 1)
    envs+=("FUZZ_ARTIFACT_DIR=$out/artifacts")
    # The seeds: the seed corpus and the regression inputs of the target, as the test mode reads
    # them. libFuzzer writes new inputs only to the first directory, the corpus of the build.
    for d in "$HERE/corpus/$t" "$HERE/regress/$t"; do
        [[ -d $d ]] && seeds+=("$d")
    done
    # One start of libFuzzer with the seconds of $1 that are left. The outer kill of rule L9 stops a
    # hang that libFuzzer cannot stop (a TSan report can deadlock in the death callback).
    hexhost_fuzz_start() {
        timeout -s KILL $(( $1 + 180 )) env "${envs[@]}" nice -n 10 "$dir/fuzz_$t" "$out/corpus" "${seeds[@]}" \
            -max_total_time="$1" -rss_limit_mb=4096 -malloc_limit_mb=4096 -timeout=60 -max_len=4096 \
            -artifact_prefix="$out/artifacts/" -print_final_stats=1 >> "$log" 2>&1
    }
    local start=$SECONDS execs
    fuzz_rounds "$BUDGET" 200 "$log" "$out/artifacts" hexhost_fuzz_start
    execs=$(fuzz_libfuzzer_executions "$log")
    local -a arts=()
    mapfile -t arts < <(fuzz_new_artifacts "$out/artifacts" "$mark")
    fuzz_result_line "$dir/results.jsonl" "$AREA" "$t" "$prof" "$cfg" fuzz "$(( SECONDS - start ))" \
        "$execs" "${#arts[@]}" "${arts[@]}"
    echo "$AREA-$prof-$cfg $t: $(( SECONDS - start )) s, $FUZZ_STARTS starts, $execs executions, corpus $(fd -t f . "$out/corpus" | wc -l), crash files ${#arts[@]}"
}

# Run one target one time on each seed and each regression input, with no check kept off.
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
    fuzz_result_line "$dir/results.jsonl" "$AREA" "$t" "$prof" "$cfg" test "$(( SECONDS - start ))" \
        "${#files[@]}" "${#failed[@]}" "${failed[@]}"
    local names=""
    for f in "${failed[@]}"; do
        names+=" ${f#"$HERE"/}"
    done
    echo "$AREA-$prof-$cfg $t: ${#files[@]} inputs, ${#failed[@]} findings${names:+:$names}"
}

# The queue-cancel check: fuzz_graph with HEXHOST_EINTR=cancel:2 on the seeds and the regression
# inputs of the graph target, one at a time.
# The second dspqueue call of the process stops its queue, as when the DSP process stops, and each
# later call gives AEE_EINTERRUPTED immediately. The host must abort with the code 0x0000002e of the
# read or the write. A host that does the call again and again gets the libFuzzer timeout or the
# outer kill. An input with less than two dspqueue calls does not stop the queue, thus the check goes
# to the next input. The expected abort writes crash files, and the check removes them.
cancel_one() {
    local prof=$1 cfg=$2 dir out log last f="" rc=0 n=0 verdict=""
    dir=$(x86_dir "$prof" "$cfg")
    out="$dir/runs/queue-cancel"
    log="$out/test-log.txt"
    last="$out/last.txt"
    rm -rf "$out"
    mkdir -p "$out/artifacts"
    : > "$log"
    local -a envs seeds=()
    mapfile -t envs < <(run_env "$cfg" 0)
    envs+=("FUZZ_ARTIFACT_DIR=$out/artifacts" "HEXHOST_EINTR=cancel:2")
    [[ -d "$HERE/corpus/graph" ]] && mapfile -t -O 0 seeds < <(fd -t f . "$HERE/corpus/graph" | sort)
    [[ -d "$HERE/regress/graph" ]] && mapfile -t -O "${#seeds[@]}" seeds < <(fd -t f . "$HERE/regress/graph" | sort)
    local start=$SECONDS
    for f in "${seeds[@]}"; do
        n=$(( n + 1 ))
        rc=0
        timeout -s KILL 60 env "${envs[@]}" nice -n 10 "$dir/fuzz_graph" -rss_limit_mb=4096 -malloc_limit_mb=4096 \
            -timeout=30 -artifact_prefix="$out/artifacts/" "$f" > "$last" 2>&1 || rc=$?
        cat "$last" >> "$log"
        echo "run.sh: $f gives the code $rc" >> "$log"
        if ! rg -q 'HEXHOST_EINTR=cancel: the queue [0-9]+ stops' "$last"; then
            [[ $rc == 0 ]] && continue
            verdict="the code $rc before the stop of the queue"
        elif ! rg -q 'dspqueue_(read|write) failed: 0x0000002e' "$last"; then
            verdict="the code $rc and no abort of the read or the write after the stop of the queue"
        fi
        break
    done
    [[ -z $verdict && ! -f $last ]] && verdict="no graph input"
    [[ -z $verdict && $rc == 0 ]] && verdict="no graph input makes two dspqueue calls"
    rm -rf "$out/artifacts" "$last"
    local -a failed=()
    local note=""
    if [[ -n $verdict ]]; then
        failed=("$f")
        note=": ${f#"$HERE"/}: $verdict"
    fi
    fuzz_result_line "$dir/results.jsonl" "$AREA" queue-cancel "$prof" "$cfg" test "$(( SECONDS - start ))" \
        "$n" "${#failed[@]}" "${failed[@]}"
    echo "$AREA-$prof-$cfg queue-cancel: $n inputs, ${#failed[@]} findings$note"
}

# Run the mode $1 (test or fuzz) with the config $2 on the targets that follow, JOBS at a time,
# for each profile. Gives the code 1 in the test mode when a target has a finding, and in each
# mode when a job stops with an error before its result line.
run_mode() {
    local mode=$1 cfg=$2
    shift 2
    fuzz_check_config "$cfg" host
    local targets="${*:-$ALL_TARGETS}" t prof dir summary bad=0 job_error=0
    for t in $targets; do
        [[ " $ALL_TARGETS " == *" $t "* ]] || fuzz_die "no target $t. The targets: $ALL_TARGETS"
    done
    for prof in $PROFILES; do
        build_x86 "$prof" "$cfg"
        dir=$(x86_dir "$prof" "$cfg")
        summary="$dir/$mode-summary.txt"
        : > "$summary"
        for t in $targets; do
            fuzz_start_job "$JOBS" "$t" "${mode}_one" "$prof" "$cfg" "$t" >> "$summary"
        done
        fuzz_wait_jobs >> "$summary" || job_error=1
        if [[ $mode == test && " $targets " == *" graph "* ]]; then
            cancel_one "$prof" "$cfg" >> "$summary"
        fi
        cat "$summary"
        echo "run.sh: the JSON lines are in $dir/results.jsonl"
        if rg -q ', [1-9][0-9]* findings' "$summary"; then
            bad=1
        fi
    done
    [[ $job_error == 1 ]] && return 1
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
    [[ -z $DSP_LIB || -f $DSP_LIB ]] || fuzz_die "the DSP library $DSP_LIB does not exist"
    refresh_llama
    # The DSP library: the shipped one for the release none run, else a build of the same tree
    local shipped=0 dsp_build=0
    [[ $prof == release && $cfg == none ]] && shipped=1
    [[ -z $DSP_LIB && $shipped == 0 ]] && dsp_build=1
    if [[ $dsp_build == 1 ]]; then
        # The preset of the app, as scripts/build-native.sh uses it
        cp -f "$REPO/android/snapdragon/CMakeUserPresets.json" "$LLAMA_DIR/CMakeUserPresets.json"
    fi
    local asan_rt=""
    [[ $cfg == asan ]] && asan_rt=$(fuzz_asan_runtime_path)
    [[ $LLAMA_DIR == "$REPO"/* ]] || fuzz_die "the llama.cpp tree $LLAMA_DIR is not inside the repository, thus the container cannot see it"
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
if [[ $dsp_build == 1 ]]; then
    cmake -S /workspace/${LLAMA_DIR#"$REPO"/} --preset arm64-android-snapdragon-release -B $rel/dsp > $rel/dsp.configure.log 2>&1
    cmake --build $rel/dsp -j$BUILD_JOBS --target htp-v79 > $rel/dsp.build.log 2>&1
fi
rm -rf $rel/runtime && mkdir -p $rel/runtime
# The ASan runtime comes from the shared build below, not from the NDK. The ubsan build links its
# runtime statically (-static-libsan, refer to the vptr text in phone/CMakeLists.txt).
rt='$(fuzz_phone_runtime "$cfg")'
if [[ -n \$rt && \$rt != '$FUZZ_ASAN_RT_NAME' ]]; then
    cp -f \$(find \$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt -name \$rt | head -n 1) $rel/runtime/
fi
" > "$dir/build.log" 2>&1 || fuzz_die "the phone build failed. Read $dir/build.log."
    [[ -n $asan_rt ]] && cp -f "$asan_rt" "$dir/runtime/"

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
            [[ -n $want && $want == "$have" ]] || fuzz_die "$SHIPPED_LIBS/$lib does not match $SHIPPED_HASHES"
            cp -f "$SHIPPED_LIBS/$lib" "$stage/lib/"
        done
    else
        fd -t f -e so . "$dir/ggml" -x cp -f {} "$stage/lib/"
    fi
    fd -t f -e so . "$dir/runtime" -x cp -f {} "$stage/lib/" 2> /dev/null || true
    local dsp=$DSP_LIB
    if [[ -z $dsp && $shipped == 1 ]]; then
        dsp="$SHIPPED_LIBS/libggml-htp-v79.so"
        want=$(rg -F "  libggml-htp-v79.so" "$SHIPPED_HASHES" | cut -d' ' -f1)
        have=$(sha256sum "$dsp" | cut -d' ' -f1)
        [[ -n $want && $want == "$have" ]] || fuzz_die "$dsp does not match $SHIPPED_HASHES"
    elif [[ -z $dsp ]]; then
        dsp="$dir/dsp/ggml/src/ggml-hexagon/libggml-htp-v79.so"
        [[ -f $dsp ]] || fuzz_die "the DSP build made no $dsp. Read $dir/dsp.build.log."
    fi
    cp -f "$dsp" "$stage/dsp/libggml-htp-v79.so"
    [[ -f $UBSAN_SUPP ]] && cp -f "$UBSAN_SUPP" "$stage/ubsan.supp"
    # The inputs: the seeds and the regression inputs of fuzz_graph, then the corpus of the x86 runs
    fd -t f . "$HERE/corpus/graph" "$HERE/regress/graph" -x cp -f {} "$stage/in/" 2> /dev/null || true
    # The first 150 small inputs of each corpus. A slice of an array, not head: head closes the
    # pipe early, and with pipefail the SIGPIPE of sort stops the script.
    local c
    local -a picked
    for c in "$REPO"/build/fuzz/$AREA-*-asan/runs/graph/corpus; do
        [[ -d $c ]] || continue
        mapfile -t picked < <(fd -t f -S -2k . "$c" | sort)
        (( ${#picked[@]} > 0 )) && cp -f -t "$stage/in/" "${picked[@]:0:150}"
    done
    # The list does not hold SHA256SUMS itself: the shell makes the file before fd runs
    (cd "$stage" && fd -t f -E SHA256SUMS . | sort | xargs sha256sum > SHA256SUMS)
    echo "run.sh: staged $(fd -t f . "$stage/in" | wc -l) inputs and $(fd -t f . "$stage/lib" | wc -l) libraries in ${stage#"$REPO"/}"
}

phone_build() {
    local cfg=$1 prof
    fuzz_check_config "$cfg" phone
    # container_run and SNAPDRAGON_IMAGE. The file sets readonly variables, thus one source only.
    # shellcheck source=../../../scripts/lib.sh
    [[ -n ${SNAPDRAGON_IMAGE:-} ]] || source "$REPO/scripts/lib.sh"
    for prof in $PROFILES; do
        phone_build_one "$prof" "$cfg"
    done
}

phone_commands() {
    local cfg=$1 prof stage d run name vars san supp tag thermal
    fuzz_check_config "$cfg" phone
    thermal=$(fuzz_phone_thermal_cmd)
    for prof in $PROFILES; do
        stage="$(phone_dir "$prof" "$cfg")/stage"
        [[ -f "$stage/bin/hexhost_phone" ]] || fuzz_die "run tests/fuzz/hexhost/run.sh phone-build $cfg --profile $prof first"
        d="$PHONE_DIR/$prof-$cfg"
        # The stage holds the shared UBSan suppression file only. The other configurations get no
        # suppression path: a sanitizer runtime stops when its suppression file does not exist.
        supp=""
        [[ $cfg == ubsan ]] && supp="$d/ubsan.supp"
        san=$(fuzz_phone_options "$cfg" "$supp")
        echo "# ==== $prof $cfg: push the files ($(fd -t f . "$stage" | wc -l) files)"
        echo "adb -s $PHONE shell 'rm -rf $d && mkdir -p $d/out'"
        echo "adb -s $PHONE push ${stage#"$REPO"/}/. $d/"
        echo "adb -s $PHONE shell 'chmod 755 $d/bin/hexhost_phone'"
        if [[ $cfg == asan ]]; then
            # A failure of the thread start is a failure of the environment, not a defect of the backend
            echo "# If this line does not give \"thread self-test ok\", stop: record an environment failure."
            echo "adb -s $PHONE shell 'cd $d && timeout -s KILL 30 env LD_LIBRARY_PATH=$d/lib $san ./bin/hexhost_phone --selftest-threads'"
        fi
        for run in "${PHONE_RUNS[@]}"; do
            name=${run%%:*}
            vars=${run#*:}
            tag="$prof-$cfg-$name"
            echo "# ---- $tag"
            echo "$thermal"
            echo "adb -s $PHONE shell 'cd $d && mkdir -p out/save-$name && timeout -s KILL 100 env LD_LIBRARY_PATH=$d/lib ADSP_LIBRARY_PATH=$d/dsp $san $vars ./bin/hexhost_phone --seconds $PHONE_SECONDS --random $PHONE_RANDOM --save out/save-$name in > out/$name.txt 2> out/$name.err; echo exit=\$? >> out/$name.txt'"
            echo "$thermal"
            echo "adb -s $PHONE shell 'pgrep -a hexhost_phone; tail -n 2 $d/out/$name.txt'"
            # The saved random inputs again with one op for each batch (the stale L2 line property)
            echo "adb -s $PHONE shell 'cd $d && if [ -z \"\$(ls out/save-$name)\" ]; then echo \"no saved input\" > out/$name-batch1.txt; else timeout -s KILL 100 env LD_LIBRARY_PATH=$d/lib ADSP_LIBRARY_PATH=$d/dsp $san $vars GGML_HEXAGON_OPBATCH=1 GGML_HEXAGON_OPQUEUE=1 ./bin/hexhost_phone --seconds $PHONE_SECONDS out/save-$name > out/$name-batch1.txt 2> out/$name-batch1.err; echo exit=\$? >> out/$name-batch1.txt; fi'"
            echo "adb -s $PHONE shell 'pgrep -a hexhost_phone; tail -n 2 $d/out/$name-batch1.txt'"
        done
        echo "# ==== $prof $cfg: pull the results"
        echo "adb -s $PHONE pull $d/out ${stage%/stage}/phone-out"
    done
}

# ---- The model graphs

# Build hexhost_graphs and run it on the paths of the app: decode, decode with the 4 recurrent state
# snapshots of speculative decoding, prefill 512, the MTP draft step and the image turn at 576 and
# 768 image tokens, on the 2B and on the 4B Q8_0. A run fails when a llama_decode fails, when a node
# of a later split reads the state tail of a fused GDN state chain, when a decode path does not
# fuse the GDN conv step (the matcher rejects the layout of the app), or when an image turn does not
# fuse the attention input of each layer (HTP_OP_ROPE_QKV) or leaves a NORM of the vision encoder (the
# text model has no NORM). The program loads full models, thus it has no sanitizer. Gives the code 1
# when a run fails.
graphs_check() {
    local dir="$REPO/build/fuzz/$AREA-graphs${BUILD_TAG:+-$BUILD_TAG}" m model mmproj rc bad=0
    refresh_llama
    mkdir -p "$dir/out"
    # Shared libraries, as the app ships them
    CC=clang CXX=clang++ cmake -S "$HERE/graphs" -B "$dir" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON -DHEXHOST_LLAMA_DIR="$LLAMA_DIR" > "$dir/configure.log" 2>&1 \
        || fuzz_die "the configure of $dir failed. Read $dir/configure.log."
    nice -n 10 cmake --build "$dir" -j"$BUILD_JOBS" --target hexhost_graphs > "$dir/build.log" 2>&1 \
        || fuzz_die "the build of $dir failed. Read $dir/build.log."
    for m in 2B 4B; do
        model="$REPO/weights/gguf/Qwen3.5-$m-Q8_0.gguf"
        mmproj="$REPO/weights/gguf/Qwen3.5-$m-Q8_0.mmproj.gguf"
        if [[ ! -f $model || ! -f $mmproj ]]; then
            echo "$AREA graphs $m: skip, $model or $mmproj does not exist"
            continue
        fi
        local -a labels=("decode" "decode-rs4" "prefill" "mtp" "vision576" "vision768")
        local -a args=("decode" "decode" "prefill 512" "mtp 8" "vision $mmproj 576" "vision $mmproj 768")
        local -a rs=(0 4 0 0 0 0)
        local i name conv
        for i in "${!labels[@]}"; do
            name="$m-${labels[i]}"
            rc=0
            # shellcheck disable=SC2086
            timeout -s KILL 900 env HEXHOST_RS_SEQ="${rs[i]}" HEXHOST_EINTR="$TEST_EINTR" "$dir/hexhost_graphs" "$model" ${args[i]} "$dir/out/$name" \
                > "$dir/out/$name.stdout" 2>&1 || rc=$?
            conv=$(rg -o '^GDN_CONV_STEP [0-9]+' "$dir/out/$name.ops.txt" 2> /dev/null | cut -d' ' -f2 || true)
            echo "$AREA graphs $name: code $rc, ${conv:-0} fused conv steps, $(tail -n 1 "$dir/out/$name.stdout" | rg -o '[0-9]+ fused state chains, [0-9]+ state tail readers in a later split' || echo 'no summary')"
            if [[ ${labels[i]} == decode* && -z $conv ]]; then
                echo "$AREA graphs $name: the host does not fuse the GDN conv step of the app"
                rc=1
            fi
            # The vision encoder: one ROPE_QKV for each layer of each photo, and no NORM (the text model has none)
            if [[ ${labels[i]} == vision* ]]; then
                local vis re='vision: ([0-9]+) images, ([0-9]+) blocks, ([0-9]+) ROPE_QKV, ([0-9]+) NORM_MUL_ADD, ([0-9]+) NORM'
                vis=$(rg -o "$re" "$dir/out/$name.stdout" || true)
                if [[ $vis =~ $re ]] && (( BASH_REMATCH[2] > 0 && BASH_REMATCH[3] == BASH_REMATCH[1] * BASH_REMATCH[2] &&
                                            BASH_REMATCH[5] == 0 )); then
                    echo "$AREA graphs $name: ${vis}"
                else
                    echo "$AREA graphs $name: the host does not fuse the attention input or the LayerNorm of each layer" \
                         "of the vision encoder (${vis:-no vision line})"
                    rc=1
                fi
            fi
            [[ $rc == 0 ]] || bad=1
        done
    done
    return $bad
}

main() {
    [[ $# -ge 1 ]] || { usage; exit 2; }
    local mode=$1
    shift
    case $mode in
        cpu-asan) mode=fuzz; set -- asan "$@" ;;
        cpu-tsan) mode=fuzz; set -- tsan "$@" ;;
        -h|--help|help) usage; exit 0 ;;
        graphs) graphs_check; exit $? ;;
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
            -*) fuzz_die_code 2 "unknown option $1" ;;
            *) targets+=("$1"); shift ;;
        esac
    done
    [[ $BUDGET =~ ^[0-9]+$ && $JOBS =~ ^[1-9][0-9]*$ ]] || fuzz_die "--budget-seconds and --jobs take positive numbers"
    [[ " $ALL_PROFILES " == *" $PROFILES "* || $PROFILES == "$ALL_PROFILES" ]] || fuzz_die "--profile takes debug or release"
    case $mode in
        test)
            # The targets, then the model graphs (the same check for each config)
            local code=0
            run_mode test "$cfg" "${targets[@]}" || code=1
            graphs_check || code=1
            return $code
            ;;
        fuzz) run_mode "$mode" "$cfg" "${targets[@]}" ;;
        phone-build) phone_build "$cfg" ;;
        phone-commands) phone_commands "$cfg" ;;
        *) usage; exit 2 ;;
    esac
}

main "$@"
