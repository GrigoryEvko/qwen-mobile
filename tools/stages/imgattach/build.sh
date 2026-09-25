#!/usr/bin/env bash
# Build the phone stage "imgattach": the proof that the stage of an attached image (LlamaNative.stageImage) gives the
# answer of the send alone, and its time to the first token, with the JNI code of the app on the 4B Q8_0 on HTP0.
#
#   JOBS=24 tools/stages/imgattach/build.sh
#
# The program is app_fuzz_driver of tests/fuzz/app (the scenario image-stage, refer to
# tests/fuzz/app/harness/app_harness.cpp): it calls the JNI entry points of android/app/src/main/cpp/llama_jni.cpp of
# the working tree through the fake Java VM of the harness. The NDK compiles it in the Snapdragon container, under the
# lock build/.container.lock, with the release profile and no sanitizer (the flags that ship: -O3 -DNDEBUG -flto),
# without libFuzzer, into build/imgattach/android. It links the llama.cpp libraries of build/ttft/phone/lib (the series
# of HEAD, the libraries of the stage imgturn), and the headers of build/ttft/src. Make them first with
# "tools/stages/ttft/build.sh phone".
#
# The phone files go to build/imgattach/phone: bin/app_fuzz_driver, bin/gate.sh, the libraries, and SHA256SUMS. Time:
# about 2 minutes.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../scripts/lib.sh"
JOBS=${JOBS:-24}
cd "$REPO_ROOT"

readonly stage=build/imgattach
readonly bdir=$stage/android
readonly out=$stage/phone
readonly tree=build/ttft/src
readonly libs=build/ttft/phone/lib

[[ -f $tree/.llama-copy-stamp && -f $libs/libllama.so ]] ||
    die "no $tree or $libs: run tools/stages/ttft/build.sh phone first"
grep -q "patches $(git rev-parse HEAD:patches)" "$tree/.llama-copy-stamp" ||
    die "$tree has a different patches tree than HEAD: run tools/stages/ttft/build.sh phone"
# The libraries must be the ones of the tree: the phone files of the stage ttft hold both.
(cd build/ttft/phone && sha256sum --quiet -c SHA256SUMS) || die "build/ttft/phone does not match its SHA256SUMS"

mkdir -p "$bdir"
(
    flock 9
    container_run -e JOBS="$JOBS" -e TREE="$tree" -e LIBS="$libs" -e BDIR="$bdir" "$SNAPDRAGON_IMAGE" bash -euo pipefail -c '
cmake -S tests/fuzz/app -B "$BDIR" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 \
    -DFUZZ_SANITIZER=none -DFUZZ_PROFILE=release -DFUZZ_LIBFUZZER=OFF \
    -DLLAMA_CPP_DIR="/workspace/$TREE" -DFUZZ_LLAMA_PREBUILT_DIR="/workspace/$LIBS" \
    -DFUZZ_APP_MODEL_DIR=/data/local/tmp/qwen/imgattach/models
cmake --build "$BDIR" -j"$JOBS" --target app_fuzz_driver
'
) 9> build/.container.lock > "$stage/build.log" 2>&1 || die "the build failed, refer to $stage/build.log"

rm -rf "$out"
mkdir -p "$out/bin" "$out/lib"
cp -f "$bdir/app_fuzz_driver" tools/phone/gate.sh "$out/bin/"
cp -f "$libs"/*.so "$out/lib/"
# Each library that the program needs must be in lib: a missing one stops it with "CANNOT LINK EXECUTABLE".
while read -r need; do
    need=${need#[}
    need=${need%]}
    [[ -f $out/lib/$need ]] || die "app_fuzz_driver needs $need, which is not in $out/lib"
done < <(readelf -d "$out/bin/app_fuzz_driver" | grep -oE '\[lib(llama|ggml|mtmd)[^]]*\]')
(cd "$out" && sha256sum bin/* lib/* > SHA256SUMS)
cat "$out/SHA256SUMS"
echo "imgattach: the phone files are in $out"
