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
#   ADSP_DIR    the ADSP_LIBRARY_PATH with libggml-htp-v79.so
#   SECONDS     the run time before the deadline. Keep it 15 s below the outer timeout.
#   COUNT       "all", or the number of cases from the start of the pack (a probe)
#   UBSAN_HALT  1 stops at the first UBSan report without a suppression, 0 prints every report
#               and continues (the probe that collects evidence)
#   TRACE       optional: "trace" runs ops_replay under its ptrace tracer (ops_replay --trace), which
#               prints the pc, the module, the instruction and the backtrace of each fatal signal of
#               each thread, also when the thread blocks the signal
#
# ops_replay continues after a crash: a new process records the case that stopped the last one.
# The loop starts a new process until ops_replay stops with 0 (all runs done) or 3 (the deadline).

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

cd "$d" || exit 1
[ -x "$d/$build/ops_replay" ] || { echo "no build $d/$build/ops_replay"; exit 1; }

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

end=$(( $(date +%s) + seconds ))
i=0
rc=1
while [ $i -lt 50 ]; do
    env $san LD_LIBRARY_PATH="$d/$build" ADSP_LIBRARY_PATH="$adsp" \
        GGML_HEXAGON_OPFUSION="$fusion" GGML_HEXAGON_OPFUSION_STATE="$fusion" \
        "$d/$build/ops_replay" $trace --pack cases.pack --out "out/res-$build-$tag.bin" \
        --progress "out/prog-$build-$tag.txt" --backends "$backends" --tag-suffix "$suffix" \
        --deadline "$end" ${count:+--count "$count"} >> "out/log-$build-$tag.txt" 2>&1
    rc=$?
    [ $rc -eq 0 ] && break
    [ $rc -eq 3 ] && break
    i=$((i + 1))
done
echo "build=$build tag=$tag rc=$rc restarts=$i"
