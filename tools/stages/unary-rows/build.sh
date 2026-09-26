#!/usr/bin/env bash
# Build the phone stage "unary-rows": the HEAD libraries against the libraries with the row change of the pointwise
# unary ops (its switch GGML_HEXAGON_UNARY_FLAT selects the kinds of op), and the check tool unarycheck.
#
#   JOBS=16 tools/stages/unary-rows/build.sh PATCH
#
# PATCH is the llama.cpp patch of the row change. build/unary-rows/build.sh is a link to this file. The files of
# the stage go to build/unary-rows.
#
# The steps:
#   1. tests/sanitizers/llama-copy.sh makes build/unary-rows/base and build/unary-rows/src, the llama.cpp tree of
#      HEAD (the pin plus patches/series). GNU patch puts PATCH on src. The copies are not git repositories, and
#      git apply in a directory below an ignored path of this repository skips files and still exits with 0.
#   2. In the Snapdragon container, under the lock build/.container.lock, CMake configures each tree with the
#      preset, the compiler flags (with -flto) and the build number and commit of scripts/build-native.sh. It
#      builds the libraries of the app (LLAMA_LIBS of scripts/lib.sh) and the DSP library v79 of each tree, and
#      test-backend-ops of base. The NDK clang++ compiles memprobe (tools/memprobe/memprobe.cpp with the five app
#      sources, the recipe of tools/stages/fixed/build.sh) and tools/stages/unary-rows/unarycheck.cpp against the
#      base libraries.
#   3. The script copies the files of the stage into build/unary-rows/phone: bin/, lib-base/ and lib-new/, which
#      keeps only the libraries that differ from lib-base. It writes SHA256SUMS.
#   4. tools/stages/unary-rows/stage.py commands writes build/unary-rows/phone-commands.txt.
#
# Time: about 25 minutes with JOBS=16 (the LTO links take most of it). Disk: about 3 GB.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../scripts/lib.sh"
JOBS=${JOBS:-16}
cd "$REPO_ROOT"

patch_file=${1:?usage: tools/stages/unary-rows/build.sh PATCH}
[[ -f $patch_file ]] || die "no patch file $patch_file"
patch_file=$(readlink -f "$patch_file")

readonly stage=build/unary-rows
readonly out=$stage/phone
readonly repro="-ffile-prefix-map=/workspace=. -fdebug-prefix-map=/workspace=. -Werror=date-time"
readonly app_src=android/app/src/main/cpp
readonly app_files="$app_src/state_cache.cpp $app_src/cache_io.cpp $app_src/spec_policy.cpp $app_src/chat_prompt.cpp $app_src/engine_tasks.cpp"

# 1. The trees
mkdir -p "$stage"
tests/sanitizers/llama-copy.sh "$stage/base"
tests/sanitizers/llama-copy.sh "$stage/src"
patch -p1 -N --dry-run --silent -d "$stage/src" < "$patch_file" > /dev/null || die "$patch_file does not apply to $stage/src"
patch -p1 -N --silent --no-backup-if-mismatch -d "$stage/src" < "$patch_file" || die "patch failed in $stage/src"
grep -q GGML_HEXAGON_UNARY_FLAT "$stage/src/ggml/src/ggml-hexagon/ggml-hexagon.cpp" ||
    die "$stage/src has no switch GGML_HEXAGON_UNARY_FLAT after the patch"
for t in base src; do
    cp -f android/snapdragon/CMakeUserPresets.json "$stage/$t/CMakeUserPresets.json"
done
sha256sum "$patch_file" | python3 -c "import sys; h = sys.stdin.read().split()[0]; print(h + '  ' + sys.argv[1])" \
    "$(basename "$patch_file")" > "$stage/patch.sha256"

# 2. The builds
SOURCE_DATE_EPOCH=$(source_date_epoch)
echo "unary-rows: SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH JOBS=$JOBS"
(
    flock 9
    container_run \
        -e SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
        -e LLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER" \
        -e LLAMA_BUILD_COMMIT_SHORT="${LLAMA_COMMIT:0:7}" \
        -e LIBS="$LLAMA_LIBS htp-v79" \
        -e JOBS="$JOBS" \
        -e REPRO="$repro" \
        -e STAGE="$stage" \
        -e APP_SRC="$app_src" -e APP_FILES="$app_files" \
        "$SNAPDRAGON_IMAGE" bash -euo pipefail -c '
source tools/stages/common/buildlib.sh
for name in base src; do
    tree=$STAGE/$name
    bdir=$STAGE/android-$name
    flags="$REPRO -ffile-prefix-map=/workspace/$tree=./third_party/llama.cpp -ffile-prefix-map=/workspace/$bdir=./build/native/llama"
    targets="$LIBS"
    [[ $name == base ]] && targets="$targets test-backend-ops"
    # The DSP library is an external project that the build step configures, thus it reads CFLAGS at that time
    export CFLAGS="$flags" CXXFLAGS="$flags"
    cmake -S "$tree" --preset arm64-android-snapdragon-release -B "$bdir" \
        -DLLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER" \
        -DLLAMA_BUILD_COMMIT="$LLAMA_BUILD_COMMIT_SHORT" \
        -DCMAKE_C_FLAGS="$(preset_flag "$tree" CMAKE_C_FLAGS) $flags" \
        -DCMAKE_CXX_FLAGS="$(preset_flag "$tree" CMAKE_CXX_FLAGS) $flags"
    # shellcheck disable=SC2086
    cmake --build "$bdir" -j"$JOBS" --target $targets
done
tree=$STAGE/base
bdir=$STAGE/android-base
flags="$REPRO -ffile-prefix-map=/workspace/$tree=./third_party/llama.cpp -ffile-prefix-map=/workspace/$bdir=./build/native/llama"
cxx=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++
# shellcheck disable=SC2086
$cxx -O2 -std=c++17 -Wall -Wextra -Wno-unused-parameter $flags -I"$tree/include" -I"$tree/common" -I"$tree/src" \
    -I"$tree/ggml/include" -I"$tree/vendor" -I"$tree/tools/mtmd" -I"$APP_SRC" tools/memprobe/memprobe.cpp $APP_FILES \
    -o "$bdir/bin/memprobe" -L"$bdir/bin" -lmtmd -lllama-common -lllama -lggml -lggml-cpu -lggml-base \
    -Wl,-rpath,"\$ORIGIN/../lib"
# unarycheck calls only the C API of ggml, thus it links the C++ runtime statically.
$cxx -O2 -std=c++17 -Wall -Wextra $flags -I"$tree/ggml/include" tools/stages/unary-rows/unarycheck.cpp \
    -o "$bdir/bin/unarycheck" -static-libstdc++ -L"$bdir/bin" -lggml -lggml-cpu -lggml-base -Wl,-rpath,"\$ORIGIN/../lib"
'
) 9> build/.container.lock > "$stage/build.log" 2>&1 || die "the build failed, refer to $stage/build.log"

# 3. The stage files
rm -rf "$out"
mkdir -p "$out/bin" "$out/lib-base" "$out/lib-new"
for name in base new; do
    bdir=$stage/android-$([[ $name == base ]] && echo base || echo src)
    for lib in $LLAMA_LIBS; do
        cp -f "$bdir/bin/lib$lib.so" "$out/lib-$name/"
    done
    cp -f "$bdir/ggml/src/ggml-hexagon/libggml-htp-v79.so" "$out/lib-$name/"
done
# lib-new keeps only the libraries that differ from lib-base: the runs put lib-new before lib-base
for so in "$out/lib-new"/*.so; do
    if cmp -s "$so" "$out/lib-base/$(basename "$so")"; then
        rm -f "$so"
    fi
done
cp -f "$stage/android-base/bin/test-backend-ops" "$stage/android-base/bin/memprobe" "$stage/android-base/bin/unarycheck" \
    "$out/bin/"
cp -f tools/phone/gate.sh "$out/bin/"
(cd "$out" && find bin lib-* -type f | sort | xargs sha256sum > SHA256SUMS)
cat "$out/SHA256SUMS"

# 4. The command file
python3 tools/stages/unary-rows/stage.py commands
