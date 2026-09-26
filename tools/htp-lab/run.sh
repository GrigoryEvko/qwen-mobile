#!/usr/bin/env bash
# htp-lab: the kernel lab for the Hexagon HTP kernels of llama.cpp.
#
# The lab compiles one DSP kernel of the llama.cpp Hexagon backend with hexagon-clang, runs it in
# the cycle-accurate simulator (hexagon-sim in timing mode), and reports the cycles per call, the
# stalls by type, the per-function profile, and the per-packet stall table. It is the Nsight
# Compute equivalent for the kernel work on the phone.
#
# Usage:
#   tools/htp-lab/run.sh build                    Build all lab programs
#   tools/htp-lab/run.sh run <target> [-- args]   Build, then run and profile one target
#   tools/htp-lab/run.sh all                      Run every target of the registry, exit 1 on a failure
#   tools/htp-lab/run.sh list                     Print the registry
#   tools/htp-lab/run.sh shell                    Open a shell in the container
#
# The registry. Each file lab/target_<name>.c declares the mode and the arguments of its canonical
# run in one line of its header comment:
#
#   // lab-run: mode=functional args=--rows 256 --iters 5
#   // lab-run: skip=<the reason in one phrase>
#
# The mode is "functional" or "timing". "timing" adds the cycle-accurate model and the profile
# files. A target that needs a file that the sweep cannot make declares skip, thus "all" reports it
# and runs it not. A target with no declaration runs in the functional mode with no argument, and
# "all" gives a notice with its name. The declaration is in the source of the target, thus a new
# target needs no edit of this file and two agents never write the same line.
#
# "all" is the regression gate of the lab: it runs every target of the registry and writes the
# result of each to out/<target>-all/. Compare two trees with a diff of the "lab:" lines.
#
# The verdict. Each run ends with a line "lab: verdict <target> PASS" or "... FAIL: <reasons>".
# hexagon-sim gives the exit code 0 for each program, thus the verdict comes from the output: an
# exception, an abort or the cycle limit of the simulator, a missing or non-zero "lab: exit status"
# line, a check line with values outside the tolerance, the word FAIL, a "lab: error" line, or no
# result line. "run" exits with 1 for a failed run. "all" runs every target, then prints one line
# for each failed target and exits with 1. test-verdict.sh does a check of each condition.
#
# A target name with the suffix "_after" runs the same program against the kernel directory with
# the proposal patches (tools/htp-lab/proposals/*.patch) applied to a copy in the output directory.
#
# Environment:
#   LLAMA_DIR  The llama.cpp checkout, read only. Default: the submodule third_party/llama.cpp with the
#              patch series applied, thus the lab measures the kernels that the app ships. A different
#              checkout gives a notice, because its numbers are not those of the app.
#              Each report starts with a "lab: tree" line: the path, the commit, and a hash of the
#              kernel directory. The hash changes with each edit of a kernel file, thus a number
#              always names its source. A target with the suffix "_after" also gets a
#              "lab: proposals" line with the name and the hash of each applied proposal.
#   IMAGE      The container image with the Hexagon SDK. Default: ghcr.io/snapdragon-toolchain/arm64-android:v0.7
#   SIM_ARGS   More options for hexagon-sim, for example "--plimit 60000000"
#   TAG        The name of the output directory suffix of "run" (out/<target>-<tag>). Default: run
#   MODE       "functional" runs "run" without the timing model. Default: timing
#   ARCH       The Hexagon version of the build and of the simulated core: v73, v75, v79 or v81.
#              Default: v79. A version other than v79 builds in out/build-<ARCH> and writes its
#              results to out/<target>-<ARCH>-<tag>, thus the four versions of one target can
#              exist side by side for a comparison.
#   SIM_CORE   The simulated core (the option --m<SIM_CORE> of hexagon-sim). The preset value is the
#              phone core of ARCH: v81na_2 for v81 (12 threads, 8 HVX contexts, 8 MB VTCM, the HMX of
#              v79na_1), and ARCH for the others ("--mv79" is v79na_1, the core of the OnePlus 13).
#              "--mv81" alone selects v81dgb_1: 2 MB VTCM, 4 HVX contexts and an HMX with no f16
#              path. "hexagon-sim --help --mv81" lists the v81 cores. The report line "lab: core"
#              gives the VTCM size and the HVX contexts of the core of each run.
#   NO_BUILD   Set to 1 to skip the build step of "run" (for parallel runs of built programs)
#   LAB_OUT    The output directory, relative to the repository. Default: tools/htp-lab/out.
#              Each concurrent user of the lab needs its own, because the build directory and the
#              patched copy of the kernel directory live there.
#   LAB_TARGETS The targets to build, space separated and without the "target_" prefix. The
#              default is every lab/target_*.c. Name your targets when others are editing the
#              lab at the same time: a build failure of a target that you do not name cannot
#              stop yours. The build also passes -k 0, thus one broken target never stops the
#              others, and "run" reports a missing program rather than a build error.
#   ALL_TARGETS The targets that "all" runs, space separated. The default is every target of the
#              registry. Give LAB_TARGETS the same names, thus the build makes those programs only.
#   PROPOSALS  The proposal patches to apply, as space-separated names of tools/htp-lab/proposals.
#              Default: every patch of that directory. "none" applies no patch, thus the "_after"
#              programs measure the kernels of the checkout.
#   PROFILE    The code generation flags. Empty (the default): the lab flags.
#              "release": the flags of the shipped DSP library (CMakeLists.txt gives the source).
#              "debug": the release flags with live asserts (no -DNDEBUG=1). A profile builds in
#              out/build-<ARCH>-<PROFILE>, thus the lab flags and the two profiles do not mix.
#   EXTRA_CFLAGS More compile flags, after the flags of the profile.
#   ALL_PLIMIT The cycle limit of each run of "all". Default: 100000000000. The timing model does
#              not retire an HMX instruction, and a program with a defect can loop without an end,
#              thus the limit stops such a run and the verdict fails it.
#   LAB_NO_LOCK Set it to 1 when the caller holds build/.container.lock in a form that this script
#              cannot see. This script takes that lock around each container, thus one container of
#              the box runs at a time, and the lock covers the life of the container only. A caller
#              of the form "flock build/.container.lock tools/htp-lab/run.sh ..." needs no variable:
#              the script finds the descriptor of the lock that it inherited and takes no second
#              lock, because a second exclusive lock of one file waits for the first without an end.
#
# Output (tools/htp-lab/out/, not in git):
#   build/                       The CMake build directory (Ninja)
#   htp-proposed/                The copy of the kernel directory with the proposal patches applied
#   <target>/stdout.txt          The program output with the "lab:" result lines
#   <target>/pa.json             The per-packet profile of hexagon-sim (--packet_analyze)
#   <target>/pa.html             The same profile as the HTML page of hexagon-profiler
#   <target>/pmu.txt             The PMU counters of the full run
#   <target>/gmon.t_0            The per-function profile (hexagon-gprof reads it)
#   <target>/report.txt          The summary that lab/report.py prints
#
# Simulator flags that work in this SDK (Hexagon Tools 19.0.07, hexagon-sim 3.38):
#   --mv79 --timing              The v79na_1 core, cycle-accurate mode (8 threads, 6 HVX contexts,
#                                8 MB VTCM at 0xd9000000, 1 MB L2)
#   --profile                    Writes gmon.t_<thread> for hexagon-gprof
#   --packet_analyze pa.json     Writes the per-packet stalls and events (needs --timing)
#   --pmu_statsfile pmu.txt      Writes all PMU counters of the run
#   --uarchtrace file            Writes a per-cycle trace with the stall reason of each thread
#   --plimit N                   Stops the simulation after N cycles
#   The simulator needs libncurses.so.5. The script makes a shim from libncurses.so.6 of the image.
#   The program reads the cycle counter through s31:30 (PCYCLE), which counts without setup in the
#   standalone runtime. The user counter c15:14 counts only after SSR bit 23 and SYSCFG bit 6 are set.
#   The HMX needs SSR bit 26 (XE2) for the thread. Refer to lab/lab.c. Without that bit an HMX
#   instruction raises the exception 0x18.
#
# The limit of this SDK: the simulator runs the HMX instructions in functional mode, but the timing
# model does not retire them. After the store of the accumulator the first read of the result never
# completes: the thread stalls in DUNCACHED_DEMAND_MISS_CYCLES until the cycle limit. The v79na_1
# stats have no HMX counter. Thus the hmx target runs in functional mode for the correctness result,
# and its timing run needs --plimit and gives no cycle number.
#
# The comparison of each kernel against its scalar reference runs inside the program: the
# "lab: check" lines report the number of elements outside the tolerance.
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${LAB_DIR}/../.." && pwd)"
CONTAINER_LOCK="${REPO_DIR}/build/.container.lock"
LLAMA_DIR="${LLAMA_DIR:-${REPO_DIR}/third_party/llama.cpp}"
IMAGE="${IMAGE:-ghcr.io/snapdragon-toolchain/arm64-android:v0.7}"
SIM_ARGS="${SIM_ARGS:-}"
SIM_CORE="${SIM_CORE:-}"
OUT_REL="${LAB_OUT:-tools/htp-lab/out}"
PROPOSALS="${PROPOSALS:-}"
LAB_TARGETS="${LAB_TARGETS:-}"
ARCH="${ARCH:-v79}"
case "${ARCH}" in
    v73|v75|v79|v81) ;;
    *) echo "error: ARCH=${ARCH} is not one of v73, v75, v79, v81" >&2; exit 1 ;;
esac
if [ -z "${SIM_CORE}" ]; then
    case "${ARCH}" in
        v81) SIM_CORE=v81na_2 ;;
        *)   SIM_CORE="${ARCH}" ;;
    esac
fi
case "${SIM_CORE}" in
    "${ARCH}"*) ;;
    *) echo "error: SIM_CORE=${SIM_CORE} is not a core of ARCH=${ARCH}" >&2; exit 1 ;;
esac
PROFILE="${PROFILE:-}"
case "${PROFILE}" in
    ""|release|debug) ;;
    *) echo "error: PROFILE=${PROFILE} is not empty, release or debug" >&2; exit 1 ;;
esac
EXTRA_CFLAGS="${EXTRA_CFLAGS:-}"
SUBMODULE_DIR="${REPO_DIR}/third_party/llama.cpp"
HTP_REL="ggml/src/ggml-hexagon/htp"

# Print the identity of the measured tree: the path, the commit, and the first 12 characters of the
# SHA-256 of all files of the kernel directory. O(bytes of the kernel directory).
tree_id() {
    local commit hash
    commit=$(git -C "${LLAMA_DIR}" rev-parse --short HEAD 2> /dev/null || echo unknown)
    hash=$(cd "${LLAMA_DIR}/${HTP_REL}" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-12)
    echo "${LLAMA_DIR} commit ${commit} htp ${hash}"
}

# Print "<name>:<first 8 characters of the SHA-256>" for each active proposal, or "none".
proposal_ids() {
    local p out="" list
    if [ "${PROPOSALS}" = "none" ]; then
        echo none
        return
    fi
    if [ -n "${PROPOSALS}" ]; then
        list=""
        for p in ${PROPOSALS}; do
            list="$list ${LAB_DIR}/proposals/$p"
        done
    else
        list=$(ls "${LAB_DIR}"/proposals/*.patch 2> /dev/null || true)
    fi
    for p in $list; do
        [ -e "$p" ] || continue
        out+="$(basename "$p"):$(sha256sum "$p" | cut -c1-8) "
    done
    echo "${out:-none}"
}

[ -d "${LLAMA_DIR}/${HTP_REL}" ] || {
    echo "error: ${LLAMA_DIR}/${HTP_REL} does not exist. Run 'git submodule update --init', or set LLAMA_DIR to a llama.cpp checkout." >&2
    exit 1
}
if [ "$(cd "${LLAMA_DIR}" && pwd -P)" != "$(cd "${SUBMODULE_DIR}" 2> /dev/null && pwd -P)" ]; then
    echo "lab: NOTICE: LLAMA_DIR=${LLAMA_DIR} is not the submodule. These numbers are not those of the app." >&2
fi
LAB_TREE="$(tree_id)"
LAB_PROPOSALS="$(proposal_ids)"

# Prints the header comment of this script, from line 2 until the first line that is not a comment.
# The block thus never goes stale when a line joins the documentation.
usage() {
    local line
    local n=0
    while IFS= read -r line; do
        n=$((n + 1))
        [ "$n" -eq 1 ] && continue
        case "$line" in
            '#!'*) continue ;;
            '#') echo "" ;;
            '# '*) echo "${line#\# }" ;;
            '#'*) echo "${line#\#}" ;;
            *) return 0 ;;
        esac
    done < "${BASH_SOURCE[0]}"
}

# ---- the registry ----

# Prints the names of every target of lab/target_*.c, one for each line, in alphabetical order.
lab_target_names() {
    local f base
    for f in "${LAB_DIR}"/lab/target_*.c; do
        [ -e "$f" ] || continue
        base="${f##*/target_}"
        echo "${base%.c}"
    done
}

# Prints the run declaration of one target as one line: the mode, one space, and the rest. The mode
# is "functional", "timing" or "skip". For "skip" the rest is the reason, and for the other two it
# is the arguments of the run, which can be empty. A target with no declaration gives "functional"
# and no argument. Arguments: the name of the target.
lab_target_decl() {
    local name="$1"
    local src="${LAB_DIR}/lab/target_${name}.c"
    local line mode="functional" rest=""
    line=$(grep -m1 '^// lab-run:' "$src" 2> /dev/null || true)
    if [ -n "$line" ]; then
        line="${line#// lab-run:}"
        case "$line" in
            *skip=*)
                mode="skip"
                rest="${line#*skip=}"
                ;;
            *)
                case "$line" in
                    *mode=*)
                        mode="${line#*mode=}"
                        mode="${mode%% *}"
                        ;;
                esac
                # The arguments hold spaces, thus they are the last field of the declaration.
                case "$line" in
                    *args=*) rest="${line#*args=}" ;;
                esac
                ;;
        esac
    fi
    echo "${mode} ${rest}"
}

# Returns 0 when this process holds an open descriptor of the container lock. "flock <file> <cmd>"
# opens the file, locks it, and starts the command with the descriptor open, thus a caller of the
# form "flock build/.container.lock tools/htp-lab/run.sh build" gives the lock to this script. A
# second flock of the file opens a second descriptor, and that one waits for the first without an
# end. Complexity O(open descriptors).
lock_inherited() {
    local fd lock
    lock="$(readlink -f "${CONTAINER_LOCK}" 2> /dev/null || true)"
    [ -n "${lock}" ] || return 1
    for fd in /proc/$$/fd/*; do
        [ "$(readlink -f "${fd}" 2> /dev/null)" = "${lock}" ] && return 0
    done
    return 1
}

# Runs a command under the container lock of the box. One container of this box runs at a time: two
# container builds compete for the cores, thus a measurement beside a build is not valid. The lock
# covers the life of the container and nothing else. When the caller holds the lock already (a
# descriptor of the lock file that this process inherited, or LAB_NO_LOCK=1), the command runs
# under the lock of the caller, thus the script cannot wait for itself.
with_container_lock() {
    if [ -n "${LAB_NO_LOCK:-}" ] || lock_inherited; then
        "$@"
        return
    fi
    mkdir -p "$(dirname "${CONTAINER_LOCK}")"
    flock "${CONTAINER_LOCK}" "$@"
}

in_container() {
    # Runs the given script text inside the container with the repository and llama.cpp mounted
    with_container_lock podman run --rm --userns=keep-id --security-opt label=disable \
        -v "${REPO_DIR}:/repo" -v "${LLAMA_DIR}:/llama" -w /repo \
        -e "SIM_ARGS=${SIM_ARGS}" -e "SIM_CORE=${SIM_CORE}" -e "LAB_TREE=${LAB_TREE}" -e "LAB_PROPOSALS=${LAB_PROPOSALS}" \
        -e "OUT_REL=${OUT_REL}" -e "PROPOSALS=${PROPOSALS}" -e "LAB_TARGETS=${LAB_TARGETS}" -e "ARCH=${ARCH}" \
        -e "PROFILE=${PROFILE}" -e "EXTRA_CFLAGS=${EXTRA_CFLAGS}" \
        "${IMAGE}" bash -c "$1"
}

# The shell prologue of every container command: the tools path and the ncurses shim
read -r -d '' PROLOGUE <<'EOF' || true
set -euo pipefail
export TOOLS=/opt/hexagon/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools
export PATH="$TOOLS/bin:$PATH"
mkdir -p /tmp/shim
ln -sf /usr/lib/x86_64-linux-gnu/libncurses.so.6 /tmp/shim/libncurses.so.5
ln -sf /usr/lib/x86_64-linux-gnu/libtinfo.so.6 /tmp/shim/libtinfo.so.5
export LD_LIBRARY_PATH=/tmp/shim
OUT="/repo/${OUT_REL}"
# v79 keeps the build directory of the lab before the ARCH switch
if [ "${ARCH}" = "v79" ]; then BUILD_DIR="$OUT/build"; else BUILD_DIR="$OUT/build-${ARCH}"; fi
# A profile has its own build directory, thus its flags never mix with the lab flags
if [ -n "${PROFILE:-}" ]; then BUILD_DIR="$OUT/build-${ARCH}-${PROFILE}"; fi
EOF

# Builds all programs. The proposal patches are applied to a copy of the kernel directory.
read -r -d '' BUILD <<'EOF' || true
mkdir -p "$OUT"
rm -rf "$OUT/htp-proposed"
cp -r /llama/ggml/src/ggml-hexagon/htp "$OUT/htp-proposed"
PROPOSED=""
if [ "${PROPOSALS:-}" = "none" ]; then
    list=""
elif [ -n "${PROPOSALS:-}" ]; then
    list=""
    for name in $PROPOSALS; do
        [ -e "/repo/tools/htp-lab/proposals/$name" ] || { echo "no proposal $name"; exit 1; }
        list="$list /repo/tools/htp-lab/proposals/$name"
    done
else
    list=$(ls /repo/tools/htp-lab/proposals/*.patch 2> /dev/null || true)
fi
for p in $list; do
    git -C /repo apply -p5 --directory="${OUT_REL}/htp-proposed" "$p"
    PROPOSED="$OUT/htp-proposed"
done
cmake -G Ninja -S /repo/tools/htp-lab -B "$BUILD_DIR" \
    -DCMAKE_TOOLCHAIN_FILE=/repo/tools/htp-lab/toolchain.cmake -DHEXAGON_ARCH="${ARCH}" \
    -DLLAMA_DIR=/llama -DHTP_PROPOSED_DIR="$PROPOSED" -DLAB_TARGETS="${LAB_TARGETS:-}" \
    -DLAB_PROFILE="${PROFILE:-}" -DLAB_EXTRA_C_FLAGS="${EXTRA_CFLAGS:-}" > "$OUT/cmake-${ARCH}${PROFILE:+-$PROFILE}.log"
# -k 0 keeps going after a failure, thus a target that another agent is editing cannot stop yours.
ninja -k 0 -C "$BUILD_DIR" || echo "lab: NOTE: at least one target did not build. The others did."
EOF

# Runs one program under the simulator, writes the profile files, and gives the verdict of the run.
# run_target <target> <tag> [args...]: the output goes to out/<target>-<tag>.
# MODE=functional runs without the timing model (no profile files), the default is the timing mode.
#
# hexagon-sim gives the exit code 0 for each program, thus the verdict comes from the output. A run
# fails when one of these conditions occurs:
#   - the simulator reports an exception, an abort, or the cycle limit (--plimit)
#   - the program wrote no line "lab: exit status = N" (lab.c prints it at each exit), thus it
#     stopped before its end, or the status is not 0
#   - a line "lab: check <what>: N of M outside tolerance" or "lab: check guard" reports N above 0
#   - a result line holds the word FAIL, or the lab runtime wrote a line "lab: error"
#   - the program wrote no result line
# run_target prints "lab: verdict <target> PASS" or "lab: verdict <target> FAIL: <the reasons>",
# writes the line to verdict.txt, and returns 1 for a failed run.
read -r -d '' RUNFN <<'EOF' || true
lab_verdict() {
    local target="$1" out="$2" sim_rc="$3" why="" n st
    grep -q "I think the exception was" "$out" && why="${why}; the simulator stopped the program at an exception"
    grep -q "^abort -- terminating" "$out" && why="${why}; the program called abort"
    grep -q "^Unexpected Run Result" "$out" && why="${why}; the simulator stopped the program at the cycle limit"
    [ "${sim_rc}" = 0 ] || why="${why}; hexagon-sim gave the exit code ${sim_rc}"
    st=$(grep -m1 -o "^lab: exit status = -\?[0-9]*" "$out" || true)
    st="${st##*= }"
    if [ -z "${st}" ]; then
        why="${why}; the program wrote no exit status line, thus it stopped before its end"
    elif [ "${st}" != 0 ]; then
        why="${why}; the program ended with the status ${st}"
    fi
    n=$(grep -cE "^lab: check .*: [1-9][0-9]* of [0-9]+ outside tolerance|^lab: check guard .*: [1-9][0-9]* guard bytes changed" "$out" || true)
    [ "${n}" = 0 ] || why="${why}; ${n} check lines report values outside the tolerance"
    n=$(grep -cE "^lab: .*\bFAIL\b" "$out" || true)
    [ "${n}" = 0 ] || why="${why}; ${n} result lines hold FAIL"
    n=$(grep -c "^lab: error" "$out" || true)
    [ "${n}" = 0 ] || why="${why}; ${n} lines of the lab runtime report an error"
    n=$(grep "^lab:" "$out" | grep -cvE "^lab: (tree|core|build profile|limits|limit [0-9]+:|exit status|NOTICE|proposals)" || true)
    [ "${n}" != 0 ] || why="${why}; the program wrote no result line"
    if [ -z "${why}" ]; then
        echo "lab: verdict ${target} PASS" | tee verdict.txt
        return 0
    fi
    echo "lab: verdict ${target} FAIL: ${why#; }" | tee verdict.txt
    return 1
}

run_target() {
    local target="$1"; local tag="$2"; shift 2
    local dir="$OUT/$target-$tag"
    [ "${ARCH}" = "v79" ] || dir="$OUT/$target-${ARCH}-$tag"
    local elf="$BUILD_DIR/lab_$target"
    local sim_rc=0
    if [ ! -x "$elf" ]; then
        echo "lab: verdict ${target} FAIL: no program ${elf}, thus the target did not build"
        return 1
    fi
    rm -rf "$dir"; mkdir -p "$dir"; cd "$dir"
    echo "== run $target-$tag: $*"
    # the source of the numbers, as the first lines of the output and thus of the report
    echo "lab: tree ${LAB_TREE} arch ${ARCH} core ${SIM_CORE}" > tree.txt
    case "$target" in *_after) echo "lab: proposals ${LAB_PROPOSALS}" >> tree.txt ;; esac
    if [ "${MODE:-timing}" = "functional" ]; then
        { cat tree.txt; hexagon-sim --m${SIM_CORE} $SIM_ARGS "$elf" -- "$@"; } 2>&1 | tee stdout.txt || sim_rc=$?
        grep "^lab:" stdout.txt > report.txt || true
    else
        { cat tree.txt; hexagon-sim --m${SIM_CORE} --timing --profile --packet_analyze pa.json --pmu_statsfile pmu.txt $SIM_ARGS \
            "$elf" -- "$@"; } 2>&1 | tee stdout.txt || sim_rc=$?
        hexagon-profiler --packet_analyze --json=pa.json --elf="$elf" -o pa.html > /dev/null 2>&1 || true
        hexagon-nm -S -n "$elf" > symbols.txt
        hexagon-llvm-objdump -d --no-show-raw-insn "$elf" > disasm.txt
        python3 /repo/tools/htp-lab/lab/report.py --pa pa.json --symbols symbols.txt --disasm disasm.txt \
            --stdout stdout.txt --target "$target-$tag" | tee report.txt || true
    fi
    local rc=0
    lab_verdict "$target" stdout.txt "$sim_rc" || rc=1
    cat verdict.txt >> report.txt
    return "$rc"
}
EOF

cmd="${1:-help}"
shift || true
case "$cmd" in
    build)
        in_container "${PROLOGUE}
${BUILD}"
        ;;
    run)
        target="${1:?target}"; shift || true
        [ "${1:-}" = "--" ] && shift
        build_step="${BUILD}"
        [ -n "${NO_BUILD:-}" ] && build_step=""
        in_container "${PROLOGUE}
${build_step}
${RUNFN}
export MODE=${MODE:-timing}
run_target ${target} ${TAG:-run} $*"
        ;;
    list)
        printf '%-14s %-10s %s\n' TARGET MODE "ARGUMENTS OR THE REASON TO SKIP"
        while IFS= read -r name; do
            read -r mode rest <<< "$(lab_target_decl "$name")"
            printf '%-14s %-10s %s\n' "$name" "$mode" "${rest:-(none)}"
        done < <(lab_target_names)
        ;;
    all)
        # The registry lives in the sources, thus the host reads it and the container gets the
        # commands. A target that declares skip gives a notice and no run.
        ALL_PLIMIT="${ALL_PLIMIT:-100000000000}"
        ALL_CMDS=""
        ALL_COUNT=0
        while IFS= read -r name; do
            if [ -n "${ALL_TARGETS:-}" ] && [[ " ${ALL_TARGETS} " != *" ${name} "* ]]; then
                continue
            fi
            read -r mode rest <<< "$(lab_target_decl "$name")"
            case "${mode}" in
                skip)
                    echo "lab: all skips ${name}: ${rest}" >&2
                    continue
                    ;;
                functional|timing) ;;
                *)
                    echo "lab: all: target ${name} declares the mode '${mode}', which is not functional, timing or skip" >&2
                    exit 1
                    ;;
            esac
            # Each run gets a cycle limit: the timing model does not retire an HMX instruction, and a
            # program with a defect can loop without an end. The limit stops it, and the verdict
            # fails the run. The sweep continues after a failed run, thus the summary names each
            # failed target.
            ALL_CMDS="${ALL_CMDS}SIM_ARGS=\"\$SIM_ARGS --plimit ${ALL_PLIMIT}\" MODE=${mode} run_target ${name} all ${rest} || ALL_FAILED=\"\${ALL_FAILED} ${name}\"
"
            ALL_COUNT=$((ALL_COUNT + 1))
        done < <(lab_target_names)
        # The generators of gen/ must give the bytes of their headers in the kernel tree, else a run
        # of a generator deletes a landed change of its header. The check runs on the host, because
        # the generators need numpy.
        GEN_RC=0
        python3 "${LAB_DIR}/gen/check_tree.py" --htp "${LLAMA_DIR}/${HTP_REL}" || GEN_RC=1
        ALL_RC=0
        in_container "${PROLOGUE}
${BUILD}
${RUNFN}
ALL_FAILED=''
${ALL_CMDS}
echo
if [ -z \"\${ALL_FAILED}\" ]; then
    echo 'lab: all: ${ALL_COUNT} of ${ALL_COUNT} targets pass'
    exit 0
fi
set -- \${ALL_FAILED}
echo \"lab: all: \$# of ${ALL_COUNT} targets fail\"
for t in \${ALL_FAILED}; do
    d=\"\$OUT/\$t-all\"
    [ \"\${ARCH}\" = v79 ] || d=\"\$OUT/\$t-\${ARCH}-all\"
    v=\$(cat \"\$d/verdict.txt\" 2> /dev/null || echo \"lab: verdict \$t FAIL: the target did not build\")
    echo \"lab: all fail \$t: \${v#*FAIL: }\"
done
exit 1" || ALL_RC=1
        if [ "${GEN_RC}" != 0 ]; then
            echo "lab: all fail gen/check_tree.py: a generator does not give the bytes of its header in the tree"
            ALL_RC=1
        fi
        exit "${ALL_RC}"
        ;;
    shell)
        # An interactive shell holds the container lock until it exits.
        with_container_lock podman run --rm -it --userns=keep-id --security-opt label=disable \
            -v "${REPO_DIR}:/repo" -v "${LLAMA_DIR}:/llama" -w /repo "${IMAGE}" bash
        ;;
    *)
        usage
        ;;
esac
