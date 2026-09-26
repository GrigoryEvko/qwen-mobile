#!/usr/bin/env bash
# Build the phone stage "ttft": the time to the first token of a chat turn and the speculation step of the 4B Q8_0
# on HTP0, the engine of the app before and after the changes of the chat turn, and the first long message.
#
#   JOBS=24 tools/stages/ttft/build.sh phone   the phone files in build/ttft/phone (the preset)
#   JOBS=24 tools/stages/ttft/build.sh host    a CPU build in build/ttft/host and two host memprobe binaries
#   tools/stages/ttft/build.sh memprobe        only memprobe again, against the libraries of the last phone build
#
# The files of the stage go to build/ttft. The phone files are also the files of the stage imgturn. TTFT_PATCH
# names a llama.cpp patch that the tree gets after the series of HEAD when HEAD does not hold it yet. Without
# it (the preset) the tree is the series of HEAD. A tree that holds the patch already takes it as it is.
#
# phone:
#   1. tests/sanitizers/llama-copy.sh makes build/ttft/src, the llama.cpp tree of HEAD (the pin plus
#      patches/series), and git apply adds TTFT_PATCH when it is set.
#   2. In the Snapdragon container, under the lock build/.container.lock, CMake configures the tree into
#      build/ttft/android with the preset and the compiler flags of scripts/build-native.sh (with -flto), and
#      builds the libraries of the app (LLAMA_LIBS of scripts/lib.sh) and the DSP library v79. The NDK clang++
#      compiles tools/memprobe/memprobe.cpp with the app sources that have no Android dependency.
#   3. The script copies the files into build/ttft/phone and writes SHA256SUMS.
# host: CMake builds llama, llama-common, mtmd and ggml-cpu of build/ttft/src for the CPU of the box, then g++
# builds build/ttft/host/memprobe (-O2) and build/ttft/host/memprobe-asan (AddressSanitizer on the code of
# memprobe and of the app sources). The host run of the 4B Q8_0 on the CPU tests the logic of the modes, not the
# phone times. The host mode uses the tree of an earlier phone build, or makes it.
#
# Time: phone about 15 minutes with JOBS=24 (the LTO links take most of it), memprobe about 1 minute, host about
# 3 minutes.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../scripts/lib.sh"
JOBS=${JOBS:-24}
cd "$REPO_ROOT"

readonly stage=build/ttft
readonly tree=$stage/src
readonly bdir=$stage/android
readonly out=$stage/phone
readonly app_src=android/app/src/main/cpp
readonly app_files="$app_src/state_cache.cpp $app_src/cache_io.cpp $app_src/spec_policy.cpp $app_src/chat_prompt.cpp $app_src/engine_tasks.cpp"
readonly patch=${TTFT_PATCH:-}
readonly repro="-ffile-prefix-map=/workspace=. -fdebug-prefix-map=/workspace=. -Werror=date-time"
readonly mode=${1:-phone}

# git apply in the tree. The tree is in the work tree of this repository, and inside a repository git apply
# takes the paths of the patch from the root of the repository and skips each file of the tree with no error.
# The ceiling stops the search for a repository above the tree, thus the paths are relative to the tree.
tree_apply() {
    (cd "$tree" && GIT_CEILING_DIRECTORIES="$(cd .. && pwd)" git apply "$@")
}

# Make the tree of HEAD, and add TTFT_PATCH when it is set and the series does not hold it. The reverse check
# comes first: git apply searches for the context of a hunk at other offsets too, thus a patch that the tree
# holds can apply a second time at a different place with the same lines. Then make sure that the tree holds
# the patch.
make_tree() {
    tests/sanitizers/llama-copy.sh "$tree"
    if [[ -n $patch ]]; then
        [[ -f $patch ]] || die "the patch $patch does not exist (TTFT_PATCH)"
        if tree_apply --reverse --check "$REPO_ROOT/$patch" 2> /dev/null; then
            echo "ttft: the series of HEAD holds $patch"
        elif tree_apply --check "$REPO_ROOT/$patch" 2> /dev/null; then
            tree_apply "$REPO_ROOT/$patch"
            echo "ttft: $tree has the series of HEAD and $patch"
        else
            die "$patch does not apply to the series of HEAD"
        fi
        tree_apply --reverse --check "$REPO_ROOT/$patch" 2> /dev/null || die "$tree does not hold $patch after the apply"
    else
        echo "ttft: $tree is the series of HEAD"
    fi
    cp -f android/snapdragon/CMakeUserPresets.json "$tree/CMakeUserPresets.json"
}

# Compile memprobe with the NDK clang++ of the Snapdragon container against the libraries in $bdir/bin. The caller
# holds the lock build/.container.lock.
compile_memprobe() {
    container_run -e FLAGS_EXTRA="$repro" -e TREE="$tree" -e BDIR="$bdir" -e APP_SRC="$app_src" -e APP_FILES="$app_files" \
        "$SNAPDRAGON_IMAGE" bash -euo pipefail -c '
cxx=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++
# shellcheck disable=SC2086
$cxx -O2 -std=c++17 $FLAGS_EXTRA -I"$TREE/include" -I"$TREE/common" -I"$TREE/src" -I"$TREE/ggml/include" \
    -I"$TREE/vendor" -I"$TREE/tools/mtmd" -I"$APP_SRC" tools/memprobe/memprobe.cpp $APP_FILES \
    -o "$BDIR/bin/memprobe" \
    -L"$BDIR/bin" -lmtmd -lllama-common -lllama -lggml -lggml-cpu -lggml-base -Wl,-rpath,"\$ORIGIN/../lib"
'
}

# Copy the libraries of the app, the DSP library, memprobe and the gate into $out, and write SHA256SUMS.
copy_phone_files() {
    rm -rf "$out"
    mkdir -p "$out/bin" "$out/lib"
    local lib
    for lib in $LLAMA_LIBS; do
        cp -f "$bdir/bin/lib$lib.so" "$out/lib/"
    done
    cp -f "$bdir/ggml/src/ggml-hexagon/libggml-htp-v79.so" "$out/lib/"
    cp -f "$bdir/bin/memprobe" "$out/bin/"
    cp -f tools/phone/gate.sh "$out/bin/"
    (cd "$out" && sha256sum bin/* lib/* > SHA256SUMS)
    cat "$out/SHA256SUMS"
    echo "ttft: the phone files are in $out"
}

# Compile memprobe again against the libraries of an earlier phone build, which must be of the patches tree of HEAD.
build_memprobe() {
    [[ -f $bdir/bin/libllama.so && -f $tree/.llama-copy-stamp ]] || die "$bdir has no phone build. Run the mode phone first."
    grep -q "patches $(git rev-parse HEAD:patches)" "$tree/.llama-copy-stamp" ||
        die "$tree has a different patches tree than HEAD. Run the mode phone."
    local before=""
    [[ -f $out/SHA256SUMS ]] && before=$(grep ' lib/' "$out/SHA256SUMS")
    (
        flock 9
        compile_memprobe
    ) 9> build/.container.lock > "$stage/build-memprobe.log" 2>&1 || die "the compile failed, refer to $stage/build-memprobe.log"
    copy_phone_files
    [[ -z $before || $before == "$(grep ' lib/' "$out/SHA256SUMS")" ]] || die "the libraries of $out changed"
}

build_phone() {
    make_tree
    mkdir -p "$bdir"
    local epoch
    epoch=$(source_date_epoch)
    (
        flock 9
        container_run \
            -e SOURCE_DATE_EPOCH="$epoch" \
            -e LLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER" \
            -e LLAMA_BUILD_COMMIT_SHORT="${LLAMA_COMMIT:0:7}" \
            -e TARGETS="$LLAMA_LIBS htp-v79" \
            -e JOBS="$JOBS" -e FLAGS_EXTRA="$repro" \
            -e TREE="$tree" -e BDIR="$bdir" \
            "$SNAPDRAGON_IMAGE" bash -euo pipefail -c '
source tools/stages/common/buildlib.sh
c_flags="$(preset_flag "$TREE" CMAKE_C_FLAGS) $FLAGS_EXTRA"
cxx_flags="$(preset_flag "$TREE" CMAKE_CXX_FLAGS) $FLAGS_EXTRA"
export CFLAGS="$FLAGS_EXTRA" CXXFLAGS="$FLAGS_EXTRA"
cmake -S "$TREE" --preset arm64-android-snapdragon-release -B "$BDIR" \
    -DLLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER" -DLLAMA_BUILD_COMMIT="$LLAMA_BUILD_COMMIT_SHORT" \
    -DCMAKE_C_FLAGS="$c_flags" -DCMAKE_CXX_FLAGS="$cxx_flags"
# shellcheck disable=SC2086
cmake --build "$BDIR" -j"$JOBS" --target $TARGETS
'
        compile_memprobe
    ) 9> build/.container.lock > "$stage/build-phone.log" 2>&1 || die "the phone build failed, refer to $stage/build-phone.log"
    copy_phone_files
}

build_host() {
    [[ -f $tree/.llama-copy-stamp ]] || make_tree
    local host=$stage/host
    mkdir -p "$host"
    cmake -S "$tree" -B "$host/llama" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON \
        -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        > "$host/cmake.log" 2>&1 || die "cmake failed, refer to $host/cmake.log"
    cmake --build "$host/llama" -j"$JOBS" --target llama llama-common mtmd ggml-cpu > "$host/build.log" 2>&1 ||
        die "the host build failed, refer to $host/build.log"
    local -a inc=(-I"$tree/include" -I"$tree/common" -I"$tree/src" -I"$tree/ggml/include" -I"$tree/vendor"
                  -I"$tree/tools/mtmd" -I"$app_src")
    local -a libs=(-L"$host/llama/bin" -lmtmd -lllama-common -lllama -lggml -lggml-cpu -lggml-base
                   -Wl,-rpath,"$REPO_ROOT/$host/llama/bin")
    # shellcheck disable=SC2086
    g++ -O2 -std=c++17 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function "${inc[@]}" tools/memprobe/memprobe.cpp \
        $app_files -o "$host/memprobe" "${libs[@]}"
    # shellcheck disable=SC2086
    g++ -O1 -g -fno-omit-frame-pointer -fsanitize=address -std=c++17 "${inc[@]}" tools/memprobe/memprobe.cpp \
        $app_files -o "$host/memprobe-asan" "${libs[@]}"
    echo "ttft: the host binaries are $host/memprobe and $host/memprobe-asan"
}

case $mode in
    phone) build_phone ;;
    memprobe) build_memprobe ;;
    host) build_host ;;
    *) die "usage: tools/stages/ttft/build.sh [phone|memprobe|host]" ;;
esac
