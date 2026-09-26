#!/usr/bin/env bash
# Build the phone files of the stage "quick": the library set of a private llama.cpp tree, with the preset, the flags
# (with -flto) and the build number of scripts/build-native.sh, as build/bench-kv/build.sh builds them.
#
#   JOBS=16 tools/stages/quick/build.sh base   the tree of HEAD (build/quick/base): lib-base, and the tools of the stage
#   JOBS=16 tools/stages/quick/build.sh new    the tree of HEAD with the patches of build/quick/patches/final
#                                              (build/quick/src): lib-new
#
# build/quick/build.sh is a link to this file. The files of the stage go to build/quick/phone. The two trees must have
# the stamp of tests/sanitizers/llama-copy.sh for the patches tree of HEAD.
#
# The tools (bin/): llama-bench and test-backend-ops of the base build, memprobe (tools/memprobe/memprobe.cpp with
# the five app sources, the recipe of tools/stages/fixed/build.sh) against the base libraries, and gate.sh. The
# container step takes the lock build/.container.lock. Time: about 15 minutes for each tree with JOBS=16.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../scripts/lib.sh"
JOBS=${JOBS:-16}
cd "$REPO_ROOT"

readonly name=${1:?usage: tools/stages/quick/build.sh base|new}
case $name in
    base) tree=build/quick/base ;;
    new)  tree=build/quick/src ;;
    *)    die "the argument is base or new, not $name" ;;
esac
readonly stage=build/quick
readonly bdir=$stage/android-$name
readonly out=$stage/phone
readonly repro="-ffile-prefix-map=/workspace=. -fdebug-prefix-map=/workspace=. -Werror=date-time"
readonly remap="-ffile-prefix-map=/workspace/$tree=./third_party/llama.cpp -ffile-prefix-map=/workspace/$bdir=./build/native/llama"
readonly app_src=android/app/src/main/cpp

[[ -f $tree/.llama-copy-stamp ]] || die "$tree has no stamp of tests/sanitizers/llama-copy.sh"
grep -q "patches $(git rev-parse HEAD:patches)" "$tree/.llama-copy-stamp" ||
    die "$tree has a different patches tree than HEAD: $(tr '\n' ' ' < "$tree/.llama-copy-stamp")"

cp -f android/snapdragon/CMakeUserPresets.json "$tree/CMakeUserPresets.json"
mkdir -p "$bdir"
SOURCE_DATE_EPOCH=$(source_date_epoch)
(
    flock 9
    container_run \
        -e SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
        -e LLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER" \
        -e LLAMA_BUILD_COMMIT_SHORT="${LLAMA_COMMIT:0:7}" \
        -e TARGETS="$LLAMA_LIBS htp-v79 llama-bench test-backend-ops" \
        -e JOBS="$JOBS" \
        -e FLAGS_EXTRA="$repro $remap" \
        -e TREE="$tree" -e BDIR="$bdir" -e NAME="$name" \
        -e APP_SRC="$app_src" -e APP_FILES="$app_src/state_cache.cpp $app_src/cache_io.cpp $app_src/spec_policy.cpp $app_src/chat_prompt.cpp $app_src/engine_tasks.cpp" \
        "$SNAPDRAGON_IMAGE" bash -euo pipefail -c '
source tools/stages/common/buildlib.sh
c_flags="$(preset_flag "$TREE" CMAKE_C_FLAGS) $FLAGS_EXTRA"
cxx_flags="$(preset_flag "$TREE" CMAKE_CXX_FLAGS) $FLAGS_EXTRA"
export CFLAGS="$FLAGS_EXTRA" CXXFLAGS="$FLAGS_EXTRA"
cmake -S "$TREE" --preset arm64-android-snapdragon-release -B "$BDIR" \
    -DLLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER" \
    -DLLAMA_BUILD_COMMIT="$LLAMA_BUILD_COMMIT_SHORT" \
    -DCMAKE_C_FLAGS="$c_flags" \
    -DCMAKE_CXX_FLAGS="$cxx_flags"
# shellcheck disable=SC2086
cmake --build "$BDIR" -j"$JOBS" --target $TARGETS
if [[ $NAME == base ]]; then
    cxx=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++
    # shellcheck disable=SC2086
    $cxx -O2 -std=c++17 -Wall -Wextra -Wno-unused-parameter $FLAGS_EXTRA -I"$TREE/include" -I"$TREE/common" \
        -I"$TREE/src" -I"$TREE/ggml/include" -I"$TREE/vendor" -I"$TREE/tools/mtmd" -I"$APP_SRC" \
        tools/memprobe/memprobe.cpp $APP_FILES -o "$BDIR/bin/memprobe" -L"$BDIR/bin" -lmtmd -lllama-common -lllama \
        -lggml -lggml-cpu -lggml-base -Wl,-rpath,"\$ORIGIN/../lib"
fi
'
) 9> build/.container.lock > "$stage/build-$name.log" 2>&1 || die "the build failed, refer to $stage/build-$name.log"

rm -rf "$out/lib-$name"
mkdir -p "$out/bin" "$out/lib-$name"
for lib in $LLAMA_LIBS llama-bench-impl; do
    cp -f "$bdir/bin/lib$lib.so" "$out/lib-$name/"
done
cp -f "$bdir/ggml/src/ggml-hexagon/libggml-htp-v79.so" "$out/lib-$name/"
if [[ $name == base ]]; then
    cp -f "$bdir/bin/llama-bench" "$bdir/bin/test-backend-ops" "$bdir/bin/memprobe" "$out/bin/"
    cp -f tools/phone/gate.sh "$out/bin/"
else
    # lib-new keeps only the libraries that differ from lib-base: the runs put lib-new before lib-base
    for so in "$out/lib-new"/*.so; do
        cmp -s "$so" "$out/lib-base/$(basename "$so")" && rm -f "$so"
    done
fi
(cd "$out" && find bin lib-* -type f | sort | xargs sha256sum > SHA256SUMS)
cat "$out/SHA256SUMS"
