#!/usr/bin/env bash
# Build the phone stage of the memory measurements from a private llama.cpp tree (the pinned
# llama.cpp with the series) with the preset and the flags of scripts/build-native.sh, plus the
# tools memprobe and kvkl. Output: build/memory/phone-<name>/ with SHA256SUMS.
#
#   TREE=build/memory/src BDIR=build/memory/android tools/memprobe/build-phone.sh <name>
#
# TREE is the private llama.cpp tree and BDIR its Android build directory. Make the tree of HEAD
# with tests/sanitizers/llama-copy.sh TREE. The NDK clang++ compiles tools/memprobe/memprobe.cpp
# with the three sources of the app that it calls (state_cache.cpp, cache_io.cpp, spec_policy.cpp),
# and tools/memprobe/kvkl.cpp. The stage also holds the gate tools/phone/gate.sh.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../scripts/lib.sh"
name=${1:?usage: build-phone.sh <name>}
JOBS=${JOBS:-32}
cd "$REPO_ROOT"
tree=${TREE:-build/memory/src}
bdir=${BDIR:-build/memory/android}
out=build/memory/phone-$name
app_src=android/app/src/main/cpp
[[ -f $tree/CMakeLists.txt ]] || die "$tree is not a llama.cpp tree. Make it with tests/sanitizers/llama-copy.sh $tree"
cp -f android/snapdragon/CMakeUserPresets.json "$tree/CMakeUserPresets.json"
mkdir -p "$bdir"
container_run "$SNAPDRAGON_IMAGE" bash -euo pipefail -c "
repro='-ffile-prefix-map=/workspace=. -fdebug-prefix-map=/workspace=. -Werror=date-time'
flags=\$(python3 -c 'import json, sys; p = [x for x in json.load(open(sys.argv[1]))[\"configurePresets\"] if x[\"name\"] == \"arm64-android-snapdragon\"][0]; print(p[\"cacheVariables\"][\"CMAKE_C_FLAGS\"])' $tree/CMakeUserPresets.json)
export CFLAGS=\"\$repro\" CXXFLAGS=\"\$repro\"
cmake -S $tree --preset arm64-android-snapdragon-release -B $bdir \
    -DLLAMA_BUILD_NUMBER=$LLAMA_BUILD_NUMBER -DLLAMA_BUILD_COMMIT=${LLAMA_COMMIT:0:7} \
    -DCMAKE_C_FLAGS=\"\$flags \$repro\" -DCMAKE_CXX_FLAGS=\"\$flags \$repro\"
cmake --build $bdir -j$JOBS --target $LLAMA_LIBS llama-bench llama-completion llama-perplexity htp-v79
cxx=\$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++
\$cxx -O2 -std=c++17 -I$tree/include -I$tree/common -I$tree/src -I$tree/ggml/include -I$tree/vendor -I$tree/tools/mtmd \
    -I$app_src tools/memprobe/memprobe.cpp $app_src/state_cache.cpp $app_src/cache_io.cpp $app_src/spec_policy.cpp \
    -o $bdir/bin/memprobe -L$bdir/bin -lmtmd -lllama-common -lllama -lggml -lggml-cpu -lggml-base \
    -Wl,-rpath,'\$ORIGIN/../lib'
\$cxx -O2 -std=c++17 -I$tree/include -I$tree/common -I$tree/ggml/include -I$tree/vendor \
    tools/memprobe/kvkl.cpp -o $bdir/bin/kvkl -L$bdir/bin -lllama-common -lllama -lggml -lggml-base \
    -Wl,-rpath,'\$ORIGIN/../lib'
" > build/memory/phone-build-$name.log 2>&1 || die "the build failed, refer to build/memory/phone-build-$name.log"
rm -rf "$out"
mkdir -p "$out/bin" "$out/lib"
for lib in $LLAMA_LIBS; do
    cp -f "$bdir/bin/lib$lib.so" "$out/lib/"
done
cp -f "$bdir/ggml/src/ggml-hexagon/libggml-htp-v79.so" "$out/lib/"
for tool in llama-bench llama-completion llama-perplexity; do
    cp -f "$bdir/bin/$tool" "$out/bin/"
    impl="$bdir/bin/lib$tool-impl.so"
    [[ -f $impl ]] && cp -f "$impl" "$out/lib/"
done
cp -f "$bdir/bin/memprobe" "$bdir/bin/kvkl" "$out/bin/"
cp -f tools/phone/gate.sh "$out/bin/"
(cd "$out" && sha256sum bin/* lib/* > SHA256SUMS)
cat "$out/SHA256SUMS"
