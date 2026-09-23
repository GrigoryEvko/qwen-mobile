#!/usr/bin/env bash
# Print the adb commands that run the ISA silicon probe on the phone. The script runs no adb
# command itself: the phone belongs to the lead engineer, who reads and runs the commands.
#
#   tools/htp-lab/probe/push-and-run.sh [--version vNN] [--tag NAME] [--allow-newer-skel] MODE
#
# MODE is one of:
#   --info                     "isaprobe info": the chip, the library, the setup and the kernels.
#   (no mode option)           "isaprobe census" with the ARM CPU oracle. Options:
#       --sim-tag TAG          The tag of the simulator census run (tools/htp-lab/isa/run_census.sh
#                              TAG). The corpus is tools/htp-lab/out-isa/isa-TAG/corpus_*.bin. Default: c.
#       --corpus DIR           A different directory with the corpus_*.bin files.
#       --only TEXT            Run only the ops whose names contain TEXT, for example "qfpair".
#       --no-oracle            Do not compute the oracle files.
#   --kernel ID|NAME --problem FILE [--iters N]
#                              "isaprobe kernel": one candidate kernel on a problem of mv_case.py,
#                              against the ggml scalar reference on the ARM CPU. Default iters: 10.
#
# Common options:
#   --version vNN        The DSP library to run: v73, v75, v79 or v81. Default: v79.
#   --tag NAME           The name of the output directory. Default: the version, plus "-neg" for
#                        the negative test, plus "-k<kernel>" for the kernel mode.
#   --allow-newer-skel   The negative test: run a library for a newer DSP than the chip (for
#                        example --version v81 on the v79 chip). Expect incorrect values.
#
# The commands put everything below /data/local/tmp/qwen/isaprobe on the phone:
#   bin/isaprobe                   The host program
#   vNN/libisaprobe_skel.so        One directory for each version. ADSP_LIBRARY_PATH selects one.
#   corpus/                        The corpus streams (12 MB)
#   problems/                      The problem files of the kernel mode
#   out/<tag>/                     The outputs, the census record isaprobe.txt, and stdout.txt.
#                                  The full census writes about 150 MB of DSP outputs and 75 MB of
#                                  oracle outputs.
# The pull puts out/<tag> into build/isaprobe/phone/<tag> on this machine.
#
# Each command obeys the phone checklist: the thermal status is 0 before the run, the phone does
# not charge, the run has "timeout -s KILL 100", and no isaprobe process stays after the run.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
readonly REPO_ROOT
readonly REMOTE=/data/local/tmp/qwen/isaprobe

version=v79
sim_tag=c
corpus=""
only=""
tag=""
allow_newer=0
mode=census
oracle=1
kernel=""
problem=""
iters=10
while [[ $# -gt 0 ]]; do
    case $1 in
        --version) version=${2:?--version needs a value}; shift 2 ;;
        --sim-tag) sim_tag=${2:?--sim-tag needs a value}; shift 2 ;;
        --corpus) corpus=${2:?--corpus needs a directory}; shift 2 ;;
        --only) only=${2:?--only needs a text}; shift 2 ;;
        --tag) tag=${2:?--tag needs a name}; shift 2 ;;
        --allow-newer-skel) allow_newer=1; shift ;;
        --info) mode=info; shift ;;
        --no-oracle) oracle=0; shift ;;
        --kernel) mode=kernel; kernel=${2:?--kernel needs an ID or a name}; shift 2 ;;
        --problem) problem=${2:?--problem needs a file}; shift 2 ;;
        --iters) iters=${2:?--iters needs a count}; shift 2 ;;
        -h | --help) sed -n '2,37p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "error: unknown argument $1. Refer to --help." >&2; exit 1 ;;
    esac
done
case $version in
    v73 | v75 | v79 | v81) ;;
    *) echo "error: the version $version is not one of v73, v75, v79, v81" >&2; exit 1 ;;
esac
if [[ -z $tag ]]; then
    tag=$version
    if [[ $allow_newer == 1 ]]; then tag+="-neg"; fi
    if [[ $mode == kernel ]]; then tag+="-k$kernel"; fi
fi
for name in "$tag" "$sim_tag" "${kernel:-x}"; do
    if [[ ! $name =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "error: the name $name must have only letters, digits, '.', '_' and '-'" >&2
        exit 1
    fi
done
if [[ ! $iters =~ ^[0-9]+$ || $iters -lt 1 ]]; then
    echo "error: --iters needs a positive count" >&2
    exit 1
fi

# The files must exist on this machine. A missing file gives a warning, not a stop, thus the
# lead can read the commands before the build or the simulator run.
bin=build/isaprobe/bin/isaprobe
skel=build/isaprobe/$version/libisaprobe_skel.so
for f in "$bin" "$skel"; do
    [[ -f $REPO_ROOT/$f ]] || echo "# WARNING: $f does not exist. Run tools/htp-lab/probe/build.sh first." >&2
done
if [[ $mode == census ]]; then
    corpus=${corpus:-tools/htp-lab/out-isa/isa-$sim_tag}
    if ! compgen -G "$REPO_ROOT/$corpus/corpus_*.bin" > /dev/null && ! compgen -G "$corpus/corpus_*.bin" > /dev/null; then
        echo "# WARNING: $corpus has no corpus_*.bin. Run tools/htp-lab/isa/run_census.sh $sim_tag first." >&2
    fi
fi
if [[ $mode == kernel ]]; then
    [[ -n $problem ]] || { echo "error: --kernel needs --problem FILE (tools/htp-lab/probe/mv_case.py make)" >&2; exit 1; }
    [[ -f $problem ]] || echo "# WARNING: $problem does not exist. Make it with tools/htp-lab/probe/mv_case.py make." >&2
fi

flag=""
if [[ $allow_newer == 1 ]]; then flag+=" --allow-newer-skel"; fi
case $mode in
    info) run="./bin/isaprobe info$flag" ;;
    census)
        if [[ -n $only ]]; then flag+=" --only $only"; fi
        if [[ $oracle == 0 ]]; then flag+=" --no-oracle"; fi
        run="./bin/isaprobe census corpus out/$tag$flag"
        ;;
    kernel) run="./bin/isaprobe kernel $kernel problems/$(basename "${problem:-x}") out/$tag --iters $iters$flag" ;;
esac
env_vars="ADSP_LIBRARY_PATH=$REMOTE/$version LD_LIBRARY_PATH=$REMOTE/$version"
label=chip-$tag

cat << EOF
# The ISA silicon probe: library $version, mode $mode, output tag $tag. Run these lines from the repository root.
cd $REPO_ROOT

# 1. Before the run: "Thermal Status: 0", and "AC powered", "USB powered", "Wireless powered" all false.
adb shell 'dumpsys thermalservice | grep "Thermal Status"; dumpsys battery | grep -E "powered|temperature"'

# 2. Push the program and the library of $version.
adb shell mkdir -p $REMOTE/bin $REMOTE/$version $REMOTE/corpus $REMOTE/problems $REMOTE/out
adb push $bin $REMOTE/bin/isaprobe
adb push $skel $REMOTE/$version/libisaprobe_skel.so
adb shell chmod 755 $REMOTE/bin/isaprobe
EOF
if [[ $mode == census ]]; then
    cat << EOF
adb shell rm -f "$REMOTE/corpus/corpus_*.bin"
adb push $corpus/corpus_*.bin $REMOTE/corpus/
EOF
elif [[ $mode == kernel ]]; then
    cat << EOF
adb push $problem $REMOTE/problems/
EOF
fi
cat << EOF

# 3. Run. The exit code goes to the last line of stdout.txt: 0 success, 2 FastRPC or setup error,
#    3 the version gate stopped the run, 4 an op or the kernel gave a nonzero status.
adb shell 'cd $REMOTE && rm -rf out/$tag && mkdir -p out/$tag && $env_vars timeout -s KILL 100 $run > out/$tag/stdout.txt 2>&1; echo "exit \$?" >> out/$tag/stdout.txt; tail -n 20 out/$tag/stdout.txt'

# 4. After the run: the thermal status, no isaprobe process, and the DSP log lines of the probe.
adb shell 'dumpsys thermalservice | grep "Thermal Status"; pgrep -a isaprobe || echo "no isaprobe process"'
mkdir -p build/isaprobe/phone && rm -rf build/isaprobe/phone/$tag
adb pull $REMOTE/out/$tag build/isaprobe/phone/
adb logcat -d -t 3000 | grep -a -i -E "isaprobe|adsprpc" > build/isaprobe/phone/$tag/logcat.txt || true
EOF
if [[ $mode == census ]]; then
    if [[ $oracle == 1 ]]; then
        cat << EOF

# 5. The error of each op on the chip against the ARM CPU oracle (numpy on this machine).
python3 tools/htp-lab/probe/oracle_compare.py census build/isaprobe/phone/$tag --csv build/isaprobe/phone/$tag/oracle.csv
EOF
    fi
    cat << EOF

# 6. The chip against the simulator. compare.py reads each version from
#    tools/htp-lab/out-isa/isa-<label>-<tag> (isa-<tag> for v79) and needs the corpus in each
#    directory. The link adds the chip as the version "$label". compare.py writes
#    tools/htp-lab/out-isa/census.csv again.
cp $corpus/corpus_*.bin build/isaprobe/phone/$tag/
ln -sfn $REPO_ROOT/build/isaprobe/phone/$tag tools/htp-lab/out-isa/isa-$label-$sim_tag
python3 tools/htp-lab/isa/compare.py --tag $sim_tag --archs $version $label
EOF
elif [[ $mode == kernel ]]; then
    cat << EOF

# 5. The kernel against the ggml scalar reference, with the time (numpy on this machine).
python3 tools/htp-lab/probe/oracle_compare.py matvec build/isaprobe/phone/$tag
EOF
fi
