#!/system/bin/sh
# Run the cases of the op fuzzer on the phone until a deadline, with one build of the replay
# driver (one profile and one sanitizer). The host pushes this file with the builds, and
# tests/fuzz/ops/run.sh phone-commands prints the commands that run it.
#
#   sh phone_run.sh BUILD TAG BACKENDS SUFFIX FUSION ADSP_DIR SECONDS COUNT UBSAN_HALT
#
#   BUILD       PROFILE-CONFIG, for example release-none or debug-asan: the directory of the build in
#               the work directory. The release-none directory also holds the shipped libraries.
#   TAG         the name of the result, progress and log files (out/res-BUILD-TAG.bin and the others)
#   BACKENDS    the devices, for example CPU,HTP0
#   SUFFIX      a tag suffix of the results, "-" for none. The results always get "-BUILD".
#   FUSION      the value of GGML_HEXAGON_OPFUSION and GGML_HEXAGON_OPFUSION_STATE (0 or 1)
#   ADSP_DIR    the ADSP_LIBRARY_PATH with libggml-htp-v79.so. The script prints the SHA-256 of each
#               DSP library in it before the first case.
#   SECONDS     the run time before the deadline. Keep it 15 s below the outer timeout.
#   COUNT       "all", or the number of cases from the start of the pack (a probe)
#   UBSAN_HALT  1 stops at the first UBSan report without a suppression, 0 prints every report
#               and continues (the probe that collects evidence)
#   TRACE       optional: "trace" runs ops_replay under its ptrace tracer (ops_replay --trace), which
#               prints the pc, the module, the instruction and the backtrace of each fatal signal of
#               each thread, also when the thread blocks the signal
#
# Optional environment of this script:
#   FUZZ_OPS_PACK   the pack in the work directory (cases.pack)
#   FUZZ_OPS_EXTRA  more options of ops_replay, for example "--repeat 20 --only mul_mat_id"
#   FUZZ_OPS_ENV    more environment of ops_replay, for example "GGML_HEXAGON_NHVX=1"
#
# ops_replay continues after a crash: a new process records the case that stopped the last one.
# The loop starts a new process until ops_replay stops with 0 (all runs done) or 3 (the deadline).
# The exit code 4 is an environment failure (no ASan runtime, or the thread self-test failed).

# FUZZ_OPS_PHONE_DIR replaces the work directory for a test of this script on the host.
d=${FUZZ_OPS_PHONE_DIR:-/data/local/tmp/qwen/fuzz/ops}
build=$1
tag=$2
backends=$3
suffix=$4
fusion=$5
adsp=$6
seconds=$7
count=$8
halt=$9
trace=""
[ "${10}" = "trace" ] && trace="--trace"
config=${build#*-}
[ "$suffix" = "-" ] && suffix=""
suffix="$suffix-$build"
[ "$count" = "all" ] && count=""
pack=${FUZZ_OPS_PACK:-cases.pack}
extra=${FUZZ_OPS_EXTRA:-}
xenv=${FUZZ_OPS_ENV:-}

cd "$d" || exit 1
[ -x "$d/$build/ops_replay" ] || { echo "no build $d/$build/ops_replay"; exit 1; }
mkdir -p out

# The DSP libraries of the run: the SHA-256 of each libggml-htp-v*.so in ADSP_DIR, in the output of
# the command (the stage log) and in the log of the run. The results pair with these bytes.
ndsp=0
for lib in "$adsp"/libggml-htp-v*.so; do
    [ -f "$lib" ] || continue
    line="dsp: $(sha256sum "$lib")"
    echo "build=$build tag=$tag $line"
    echo "$line" >> "out/log-$build-$tag.txt"
    ndsp=$((ndsp + 1))
done
[ $ndsp -gt 0 ] || echo "build=$build tag=$tag dsp: no libggml-htp-v*.so in $adsp"

# the options of the one sanitizer of the build. With the symbolizer, the report frames have
# function names, and the function-level entries of ubsan.supp can match them.
sym=""
[ -x "$d/llvm-symbolizer" ] && sym=":external_symbolizer_path=$d/llvm-symbolizer"
san=""
case $config in
    asan)   san="ASAN_OPTIONS=detect_leaks=0:halt_on_error=1:abort_on_error=0$sym" ;;
    hwasan) san="HWASAN_OPTIONS=halt_on_error=1:abort_on_error=0$sym" ;;
    ubsan)
        san="UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=$halt$sym"
        [ -f "$d/ubsan.supp" ] && san="$san:suppressions=$d/ubsan.supp"
        ;;
esac

# The libraries: the build directory, and for asan the runtime of compiler-rt 22.1.8 first (the
# runtime of NDK r29 kills each new thread on the SM8750, refer to
# tests/sanitizers/build-asan-android-runtime.sh).
libpath="$d/$build"
if [ "$config" = asan ]; then
    [ -f "$d/asan-rt/libclang_rt.asan-aarch64-android.so" ] || { echo "no ASan runtime in $d/asan-rt"; exit 4; }
    libpath="$d/asan-rt:$d/$build"
    # The first step of an ASan run: one thread must start. A failure is an environment failure,
    # not a finding, and no case runs.
    env $san LD_LIBRARY_PATH="$libpath" "$d/$build/ops_replay" --selftest-threads >> "out/log-$build-$tag.txt" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "build=$build tag=$tag ENVIRONMENT FAILURE: the thread self-test gave rc=$rc (refer to tests/sanitizers/build-asan-android-runtime.sh), no case ran"
        exit 4
    fi
fi

end=$(( $(date +%s) + seconds ))
i=0
rc=1
while [ $i -lt 50 ]; do
    env $san $xenv LD_LIBRARY_PATH="$libpath" ADSP_LIBRARY_PATH="$adsp" \
        GGML_HEXAGON_OPFUSION="$fusion" GGML_HEXAGON_OPFUSION_STATE="$fusion" \
        "$d/$build/ops_replay" $trace --pack "$pack" --out "out/res-$build-$tag.bin" \
        --progress "out/prog-$build-$tag.txt" --backends "$backends" --tag-suffix "$suffix" \
        --deadline "$end" ${count:+--count "$count"} $extra >> "out/log-$build-$tag.txt" 2>&1
    rc=$?
    [ $rc -eq 0 ] && break
    [ $rc -eq 3 ] && break
    i=$((i + 1))
done
echo "build=$build tag=$tag rc=$rc restarts=$i"
