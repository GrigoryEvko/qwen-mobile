# Shared parts of the DSP builds of tools/hmx-bench. Source this file, do not execute it.
#
# Each program is a shared object with a main() for run_main_on_hexagon of the Hexagon SDK.
# Each one links the lab runtime src/dsp_lab.c and no vendor library, thus a clone of this
# repository builds every program.
#
# The image, the container engine and the repository root come from scripts/lib.sh, thus the
# toolchain is the same image that builds the libraries of the app, pinned by its digest.

source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../scripts/lib.sh"

# The Hexagon core of the programs. The phone has v79.
HMX_ARCH=${ARCH:-v79}
readonly HMX_ARCH

# Build one or more DSP programs in the pinned toolchain image, under the container lock.
#
# The first argument is the output directory, relative to the repository. Each further argument
# is "<name>=<source>", where the name is the program without the .so suffix and the source is
# a C file, relative to the repository. The function writes <name>.so for each argument.
#
# The objects go to a directory inside the container, thus the output directory holds programs
# only. The name of that directory is fixed, because the Hexagon linker writes its command line
# into each program: a name from mktemp would put a new string in the program on each run, and two
# runs of one source would give different bytes. Complexity: O(programs), one compiler call each.
build_dsp_programs() {
    local out_rel=$1 spec
    shift
    [[ $# -gt 0 ]] || die "build_dsp_programs needs at least one <name>=<source> argument"
    local -a names=() sources=()
    for spec in "$@"; do
        [[ $spec == *=* ]] || die "the argument $spec is not of the form <name>=<source>"
        names+=("${spec%%=*}")
        sources+=("${spec#*=}")
    done
    mkdir -p "$REPO_ROOT/$out_rel" "$REPO_ROOT/build"
    (
        flock 9
        container_run \
            -e HMX_ARCH="$HMX_ARCH" \
            -e OUT_REL="$out_rel" \
            -e NAMES="${names[*]}" \
            -e SOURCES="${sources[*]}" \
            "$SNAPDRAGON_IMAGE" bash -euo pipefail -c '
SDK=$HEXAGON_SDK_ROOT
TOOLS=$HEXAGON_TOOLS_ROOT
VAR=$DEFAULT_TOOLS_VARIANT
CC=$TOOLS/Tools/bin/hexagon-clang
OUT=/workspace/$OUT_REL
TMP=/tmp/hmx-bench-objects
rm -rf $TMP
mkdir -p $TMP
trap "rm -rf $TMP" EXIT
INC="-I$SDK/rtos/qurt/compute${HMX_ARCH}/include -I$SDK/rtos/qurt/compute${HMX_ARCH}/include/qurt
     -I$SDK/rtos/qurt/compute${HMX_ARCH}/include/posix
     -I$SDK/ipc/fastrpc/rtld/ship/hexagon_${VAR}_${HMX_ARCH}
     -I$SDK/ipc/fastrpc/rpcmem/inc -I$SDK/rtos/qurt -I$SDK/utils/examples
     -isystem $SDK/incs -isystem $SDK/incs/stddef -isystem $SDK/ipc/fastrpc/incs
     -I/workspace/tools/htp-lab/lab"
CFLAGS="-m${HMX_ARCH} -G0 -Wall -Werror -Wno-unused-function -fno-zero-initialized-in-bss
        -fdata-sections -fpic -fPIC -mhvx -mhvx-length=128B -mhmx -O2 -DLAB_DEVICE=1"
LDFLAGS="-m${HMX_ARCH} -G0 -fpic -Wl,-Bsymbolic
         -Wl,-L$TOOLS/Tools/target/hexagon/lib/${HMX_ARCH}/G0/pic
         -Wl,-L$TOOLS/Tools/target/hexagon/lib/ -Wl,--no-threads -Wl,--wrap=malloc
         -Wl,--wrap=calloc -Wl,--wrap=free -Wl,--wrap=realloc -Wl,--wrap=memalign -shared"
# shellcheck disable=SC2086
$CC $INC $CFLAGS -c /workspace/tools/hmx-bench/src/dsp_lab.c -o $TMP/dsp_lab.o
read -r -a names <<< "$NAMES"
read -r -a sources <<< "$SOURCES"
for i in "${!names[@]}"; do
    name=${names[$i]}
    # shellcheck disable=SC2086
    $CC $INC $CFLAGS -c "/workspace/${sources[$i]}" -o "$TMP/$name.o"
    # shellcheck disable=SC2086
    $CC $LDFLAGS -o "$OUT/$name.so" -Wl,-soname,"$name.so" \
        -Wl,--start-group "$TMP/$name.o" $TMP/dsp_lab.o -Wl,--end-group -lc
done
'
    ) 9> "$REPO_ROOT/build/.container.lock"
}

# Copy run_main_on_hexagon of the SDK and the skel library next to the programs of one output
# directory. The argument is that directory, relative to the repository.
#
# The SDK does not supply librun_main_on_hexagon_skel.so, thus the file comes from
# tools/hmx-bench/build, where a person puts it one time. The header of build-i8.sh tells how to
# make it. Complexity: O(1).
copy_dsp_runner() {
    local out_rel=$1
    local skel="$REPO_ROOT/tools/hmx-bench/build/librun_main_on_hexagon_skel.so"
    [[ -f $skel ]] \
        || die "$skel is missing. Make it from the SDK as the header of tools/hmx-bench/build-i8.sh tells."
    (
        flock 9
        container_run -e OUT_REL="$out_rel" "$SNAPDRAGON_IMAGE" bash -euo pipefail -c '
install -m 755 "$HEXAGON_SDK_ROOT/libs/run_main_on_hexagon/ship/android_aarch64/run_main_on_hexagon" \
    "/workspace/$OUT_REL/run_main_on_hexagon"
'
    ) 9> "$REPO_ROOT/build/.container.lock"
    cp -f "$skel" "$REPO_ROOT/$out_rel/"
}
