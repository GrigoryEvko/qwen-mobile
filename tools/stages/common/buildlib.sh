# shellcheck shell=bash
# The parts that each build recipe of a phone stage shares (tools/stages/*/build.sh).
# Source this file, do not execute it. It is valid on the host and in the Snapdragon container,
# because the container has the repository at /workspace and its current directory is /workspace:
#
#   source tools/stages/common/buildlib.sh
#
# On the host, source scripts/lib.sh first. The host functions (snapdragon_run, stage_build) use its
# container_run, its pins and its die. The container functions (preset_flag, ndk_cxx, llama_build,
# memprobe_build) run in the script of snapdragon_run, which sources this file.
#
# A stage library must have the compiler flags of scripts/build-native.sh, or its bytes differ from
# the library of the app for a reason that no measurement shows. Thus a recipe takes the flags from
# the CMake preset of the app and adds only its own flags.
#
# The scripts that source this file read the constants below, thus shellcheck cannot see their
# readers.
# shellcheck disable=SC2034

# The flags that each stage adds to the flags of the preset. The objects name each path relative to
# the repository, and a date macro stops the compile. Thus two machines give the same bytes.
STAGE_REPRO="-ffile-prefix-map=/workspace=. -fdebug-prefix-map=/workspace=. -Werror=date-time"

# The app sources that memprobe compiles: the sources of the app that have no Android dependency.
APP_SRC=android/app/src/main/cpp
APP_FILES="$APP_SRC/state_cache.cpp $APP_SRC/cache_io.cpp $APP_SRC/spec_policy.cpp $APP_SRC/chat_prompt.cpp $APP_SRC/engine_tasks.cpp"

# ---- The host ----

# Run one script in the Snapdragon container, from the host. The script runs with bash -euo pipefail,
# after it sources this file.
#
#   snapdragon_run SCRIPT [ENGINE_OPTION...]
#
# Each ENGINE_OPTION goes to container_run before the image, for example -e NAME=VALUE.
#
# The local names of the host functions start with sr_ and sb_. A recipe has readonly globals (tree,
# bdir, targets), and "local" of a readonly name fails and keeps the global value.
snapdragon_run() {
    local sr_script=$1
    shift
    container_run "$@" "$SNAPDRAGON_IMAGE" bash -euo pipefail -c "
source tools/stages/common/buildlib.sh
$sr_script"
}

# Build one llama.cpp tree of a stage in the Snapdragon container, from the host: llama_build, then
# SCRIPT.
#
#   stage_build [--tests] [--no-remap] TREE BDIR TARGETS [SCRIPT]
#
# BDIR is the build directory, and TARGETS is one text with the names of the CMake targets. SCRIPT is
# a bash text that runs after the build, for example the compile of a tool of the stage. It can use
# the functions of this file, and its environment has TREE, BDIR, TARGETS, JOBS and FLAGS_EXTRA (the
# flags that the stage adds to the flags of the preset).
#
#   --tests     Give -DLLAMA_BUILD_TESTS=ON to the configure step.
#   --no-remap  FLAGS_EXTRA is STAGE_REPRO only. Without this option, FLAGS_EXTRA also records the
#               files of TREE as third_party/llama.cpp and the files of BDIR as build/native/llama,
#               the paths of the app build. Thus a library that the stage does not change can have
#               the bytes of the APK library.
#
# The caller sets JOBS. The container also gets SOURCE_DATE_EPOCH (the date of the last commit) and
# the build number and commit of the pin of scripts/lib.sh.
stage_build() {
    local sb_tests="" sb_remap=1
    while [[ $# -gt 0 && $1 == --* ]]; do
        case $1 in
            --tests) sb_tests=" -DLLAMA_BUILD_TESTS=ON" ;;
            --no-remap) sb_remap=0 ;;
            *) die "stage_build: the option $1 is not known" ;;
        esac
        shift
    done
    [[ $# -ge 3 && $# -le 4 ]] || die "usage: stage_build [--tests] [--no-remap] TREE BDIR TARGETS [SCRIPT]"
    local sb_tree=$1 sb_bdir=$2 sb_targets=$3 sb_script=${4:-} sb_flags=$STAGE_REPRO sb_epoch
    if [[ $sb_remap == 1 ]]; then
        sb_flags+=" -ffile-prefix-map=/workspace/$sb_tree=./third_party/llama.cpp"
        sb_flags+=" -ffile-prefix-map=/workspace/$sb_bdir=./build/native/llama"
    fi
    sb_epoch=$(source_date_epoch)
    snapdragon_run "llama_build \"\$TREE\" \"\$BDIR\" \"\$TARGETS\" \"\$FLAGS_EXTRA\"$sb_tests
$sb_script" \
        -e SOURCE_DATE_EPOCH="$sb_epoch" \
        -e LLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER" \
        -e LLAMA_BUILD_COMMIT_SHORT="${LLAMA_COMMIT:0:7}" \
        -e TARGETS="$sb_targets" \
        -e JOBS="$JOBS" \
        -e FLAGS_EXTRA="$sb_flags" \
        -e TREE="$sb_tree" -e BDIR="$sb_bdir"
}

# ---- The container ----

# Print one cache variable of the Snapdragon preset of a CMakeUserPresets.json.
#
#   preset_flag TREE FLAG
#
# TREE is the source tree that holds CMakeUserPresets.json, and FLAG is the name of the variable,
# for example CMAKE_C_FLAGS.
preset_flag() {
    python3 -c "
import json, sys
presets = json.load(open(sys.argv[1]))[\"configurePresets\"]
preset = [p for p in presets if p[\"name\"] == \"arm64-android-snapdragon\"][0]
print(preset[\"cacheVariables\"][sys.argv[2]])
" "$1/CMakeUserPresets.json" "$2"
}

# Print the path of the NDK clang++ of the container, for the Android API level 34.
ndk_cxx() {
    echo "$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++"
}

# Configure and build one llama.cpp tree with the preset and the flags of the app.
#
#   llama_build TREE BDIR TARGETS FLAGS [CMAKE_OPTION...]
#
# FLAGS is one text with the flags that the stage adds to the C and C++ flags of the preset. Each
# CMAKE_OPTION goes to the configure step, for example -DLLAMA_BUILD_TESTS=ON. The environment gives
# JOBS, LLAMA_BUILD_NUMBER and LLAMA_BUILD_COMMIT_SHORT. The function also exports CFLAGS and
# CXXFLAGS as FLAGS: the DSP library is an external project that the build step configures, thus it
# reads CFLAGS at that time.
llama_build() {
    local tree=$1 bdir=$2 targets=$3 flags=$4 c_flags cxx_flags
    shift 4
    # An assignment, and not an argument of cmake, stops the script when preset_flag fails
    c_flags="$(preset_flag "$tree" CMAKE_C_FLAGS) $flags"
    cxx_flags="$(preset_flag "$tree" CMAKE_CXX_FLAGS) $flags"
    export CFLAGS="$flags" CXXFLAGS="$flags"
    cmake -S "$tree" --preset arm64-android-snapdragon-release -B "$bdir" \
        -DLLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER" -DLLAMA_BUILD_COMMIT="$LLAMA_BUILD_COMMIT_SHORT" "$@" \
        -DCMAKE_C_FLAGS="$c_flags" -DCMAKE_CXX_FLAGS="$cxx_flags"
    # shellcheck disable=SC2086
    cmake --build "$bdir" -j"$JOBS" --target $targets
}

# Compile memprobe (tools/memprobe/memprobe.cpp and the app sources of APP_FILES) with the NDK clang++.
#
#   memprobe_build TREE BDIR OUT FLAGS
#
# TREE gives the headers, BDIR/bin gives the libraries, and OUT is the program. FLAGS is one text with
# the compiler flags after -O2 -std=c++17. On the phone, the program finds its libraries in ../lib.
memprobe_build() {
    local tree=$1 bdir=$2 out=$3 flags=$4
    # shellcheck disable=SC2086
    "$(ndk_cxx)" -O2 -std=c++17 $flags -I"$tree/include" -I"$tree/common" -I"$tree/src" -I"$tree/ggml/include" \
        -I"$tree/vendor" -I"$tree/tools/mtmd" -I"$APP_SRC" tools/memprobe/memprobe.cpp $APP_FILES -o "$out" \
        -L"$bdir/bin" -lmtmd -lllama-common -lllama -lggml -lggml-cpu -lggml-base -Wl,-rpath,"\$ORIGIN/../lib"
}
