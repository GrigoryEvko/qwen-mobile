#!/usr/bin/env bash
# Build the phone stage "fixed": memprobe with the app modes (--cold, --sweep, --turns) against the
# library set of the stage bench-kv, and a host build of the same tree for the functional tests.
#
#   JOBS=24 tools/stages/fixed/build.sh phone   the phone files in build/fixed/phone (the preset)
#   JOBS=24 tools/stages/fixed/build.sh host    a CPU build in build/fixed/host and two host memprobe binaries
#
# build/fixed/build.sh is a link to this file. The files of the stage go to build/fixed.
#
# phone: the NDK clang++ of the Snapdragon container compiles tools/memprobe/memprobe.cpp with the
# sources of the app that have no Android dependency (state_cache.cpp, cache_io.cpp, spec_policy.cpp,
# chat_prompt.cpp, engine_tasks.cpp), with the recipe of tools/stages/bench-kv/build.sh, and links it against the
# libraries in build/bench-kv/android/bin. The libraries of the stage are the files of
# build/bench-kv/phone/lib: the llama.cpp tree of the patches tree f473cad (the pin plus patches/series)
# with the switch GGML_HEXAGON_FWHT at its preset value, thus the code of the app. The container step
# takes the lock build/.container.lock.
#
# host: CMake builds llama, llama-common, mtmd and ggml-cpu of build/bench-kv/src for the CPU of the box,
# then g++ builds build/fixed/host/memprobe (-O2) and build/fixed/host/memprobe-asan (AddressSanitizer on
# the code of memprobe and of the app sources). The host run of the 4B Q8_0 on the CPU tests the
# logic of the modes, not the phone times.
#
# Time: phone about 1 minute, host about 5 minutes with JOBS=24.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../scripts/lib.sh"
JOBS=${JOBS:-24}
cd "$REPO_ROOT"
source tools/stages/common/buildlib.sh

readonly stage=build/fixed
readonly tree=build/bench-kv/src
readonly bdir=build/bench-kv/android
readonly memprobe_src=tools/memprobe/memprobe.cpp
readonly mode=${1:-phone}

[[ -f $memprobe_src ]] || die "$memprobe_src does not exist"

[[ -f $tree/.llama-copy-stamp ]] || die "$tree has no stamp of tests/sanitizers/llama-copy.sh. Build the stage bench-kv first."
grep -q "patches $(git rev-parse HEAD:patches)" "$tree/.llama-copy-stamp" ||
    die "$tree has a different patches tree than HEAD: $(cat "$tree/.llama-copy-stamp" | tr '\n' ' ')"

build_phone() {
    [[ -f $bdir/bin/libllama.so ]] || die "$bdir/bin has no libllama.so. Build the stage bench-kv first."
    mkdir -p "$stage/android"
    (
        flock 9
        # shellcheck disable=SC2016
        snapdragon_run '
memprobe_build "$TREE" "$BDIR" "$OUT" "-Wall -Wextra -Wno-unused-parameter $FLAGS_EXTRA"' \
            -e FLAGS_EXTRA="$STAGE_REPRO" -e TREE="$tree" -e BDIR="$bdir" -e OUT="$stage/android/memprobe"
    ) 9> build/.container.lock > "$stage/build-phone.log" 2>&1 || die "the phone build failed, refer to $stage/build-phone.log"

    rm -rf "$stage/phone"
    mkdir -p "$stage/phone/bin" "$stage/phone/lib"
    cp -f "$stage/android/memprobe" "$stage/phone/bin/"
    cp -f tools/phone/gate.sh "$stage/phone/bin/"
    cp -f build/bench-kv/phone/lib/*.so "$stage/phone/lib/"
    (cd "$stage/phone" && sha256sum bin/* lib/* > SHA256SUMS)
    cat "$stage/phone/SHA256SUMS"
    # The libraries must be the bytes that the stage bench-kv ran.
    (cd build/bench-kv/phone && grep ' lib/' SHA256SUMS) | diff - <(cd "$stage/phone" && grep ' lib/' SHA256SUMS) ||
        die "the libraries of $stage/phone differ from build/bench-kv/phone"
    echo "fixed: the phone files are in $stage/phone"
}

build_host() {
    local host=$stage/host
    mkdir -p "$host"
    cmake -S "$tree" -B "$host/llama" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON \
        -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
        > "$host/cmake.log" 2>&1 || die "cmake failed, refer to $host/cmake.log"
    cmake --build "$host/llama" -j"$JOBS" --target llama llama-common mtmd ggml-cpu > "$host/build.log" 2>&1 ||
        die "the host build failed, refer to $host/build.log"
    local -a inc=(-I"$tree/include" -I"$tree/common" -I"$tree/src" -I"$tree/ggml/include" -I"$tree/vendor"
                  -I"$tree/tools/mtmd" -I"$APP_SRC")
    local -a libs=(-L"$host/llama/bin" -lmtmd -lllama-common -lllama -lggml -lggml-cpu -lggml-base
                   -Wl,-rpath,"$REPO_ROOT/$host/llama/bin")
    # shellcheck disable=SC2086
    g++ -O2 -std=c++17 -Wall -Wextra -Wno-unused-parameter "${inc[@]}" "$memprobe_src" $APP_FILES \
        -o "$host/memprobe" "${libs[@]}"
    # shellcheck disable=SC2086
    g++ -O1 -g -fno-omit-frame-pointer -fsanitize=address -std=c++17 "${inc[@]}" "$memprobe_src" \
        $APP_FILES -o "$host/memprobe-asan" "${libs[@]}"
    echo "fixed: the host binaries are $host/memprobe and $host/memprobe-asan"
}

case $mode in
    phone) build_phone ;;
    host) build_host ;;
    *) die "usage: tools/stages/fixed/build.sh [phone|host]" ;;
esac
