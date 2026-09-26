#!/usr/bin/env bash
# Build the phone stage "spf": the time of a short prefill call of the 4B Q8_0 on HTP0 against its token count,
# in the engine of the app, on the libraries of HEAD (the table) and on the libraries with the candidate patches.
#
#   JOBS=64 [STAGE=spf3] [VARIANTS="g:0001-* new:*"] tools/stages/spf/build.sh
#
# The candidate patches are tools/stages/spf/patches/*.patch, in the order of their names. STAGE names the stage of
# tools/stages/spf/stage.py (spf, the preset, spf2 or spf3), and the files of the stage go to build/STAGE.
# build/STAGE/build.sh is a link to this file. The steps below name build/spf for the preset.
# VARIANTS gives the library sets besides base, as NAME:GLOB pairs: the tree build/spf/NAME is base plus the
# candidate patches whose file names match GLOB, and its files go to lib-NAME and bin/test-backend-ops-NAME. The
# preset is "new:*" (each candidate patch).
#
# The steps:
#   1. tests/sanitizers/llama-copy.sh makes build/spf/base and a tree for each variant, the llama.cpp tree of HEAD
#      (the pin plus patches/series). GNU patch puts the candidate patches of the variant on its tree. The copies
#      are not git repositories, and git apply in a directory below an ignored path of this repository skips files
#      and still exits with 0.
#   2. In the Snapdragon container, under the lock build/.container.lock, CMake configures each tree with the
#      preset, the compiler flags (with -flto) and the build number and commit of scripts/build-native.sh. It
#      builds the libraries of the app (LLAMA_LIBS of scripts/lib.sh), the DSP library v79 and test-backend-ops of
#      each tree, and llama-bench and llama-perplexity of base. The NDK clang++ compiles memprobe
#      (tools/memprobe/memprobe.cpp with the five app sources, the recipe of tools/stages/fixed/build.sh) against
#      the base libraries.
#   3. The script copies the files of the stage into build/spf/phone: bin/, lib-base/ and lib-NAME/ for each
#      variant. lib-NAME keeps only the libraries that differ from lib-base, and the runs put lib-NAME before
#      lib-base. The DSP library of each variant differs, thus lib-NAME holds it and ADSP_LIBRARY_PATH of a run of
#      the variant names lib-NAME. The test program of a variant is bin/test-backend-ops-NAME (the candidates
#      change its bound of the chunked gated delta net). Each llama, ggml or mtmd library that a program or a
#      library needs (readelf, NEEDED) must be in lib-base, or for a file of lib-NAME in lib-NAME or lib-base,
#      else the script stops. It writes SHA256SUMS.
#   4. tools/stages/spf/stage.py commands writes build/spf/phone-commands.txt.
#
# Time: about 30 minutes with JOBS=64 (the LTO links take most of it). Disk: about 4 GB.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../scripts/lib.sh"
JOBS=${JOBS:-64}
STAGE=${STAGE:-spf}
VARIANTS=${VARIANTS:-new:*}
cd "$REPO_ROOT"

[[ $STAGE =~ ^spf[0-9]*$ ]] || die "STAGE must be spf or spf<N>, not $STAGE"
readonly stage=build/$STAGE
readonly out=$stage/phone
readonly repro="-ffile-prefix-map=/workspace=. -fdebug-prefix-map=/workspace=. -Werror=date-time"
readonly app_src=android/app/src/main/cpp
readonly app_files="$app_src/state_cache.cpp $app_src/cache_io.cpp $app_src/spec_policy.cpp $app_src/chat_prompt.cpp $app_src/engine_tasks.cpp"

shopt -s nullglob
patches=(tools/stages/spf/patches/*.patch)
shopt -u nullglob
[[ ${#patches[@]} -gt 0 ]] || die "tools/stages/spf/patches holds no patch"

# 1. The trees
# read splits the words without the pathname expansion of an unquoted $VARIANTS
read -r -a variants <<< "$VARIANTS"
names=()
for v in "${variants[@]}"; do
    name=${v%%:*}
    glob=${v#*:}
    [[ $v == *:* && $name =~ ^[a-z][a-z0-9]*$ && $name != base ]] || die "a variant is NAME:GLOB, not $v"
    names+=("$name")
done
mkdir -p "$stage"
tests/sanitizers/llama-copy.sh "$stage/base"
for v in "${variants[@]}"; do
    name=${v%%:*}
    glob=${v#*:}
    tests/sanitizers/llama-copy.sh "$stage/$name"
    n_applied=0
    for p in "${patches[@]}"; do
        # shellcheck disable=SC2053
        [[ $(basename "$p") == $glob ]] || continue
        patch -p1 -N --dry-run --silent -d "$stage/$name" < "$p" > /dev/null || die "$p does not apply to $stage/$name"
        patch -p1 -N --silent --no-backup-if-mismatch -d "$stage/$name" < "$p" || die "patch failed in $stage/$name: $p"
        echo "spf: $name: applied $p"
        n_applied=$((n_applied + 1))
    done
    [[ $n_applied -gt 0 ]] || die "no candidate patch matches $glob"
done
for t in base "${names[@]}"; do
    cp -f android/snapdragon/CMakeUserPresets.json "$stage/$t/CMakeUserPresets.json"
done
(cd tools/stages/spf/patches && sha256sum ./*.patch) > "$stage/patches.sha256"

# 2. The builds
SOURCE_DATE_EPOCH=$(source_date_epoch)
echo "spf: SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH JOBS=$JOBS"
(
    flock 9
    container_run \
        -e SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
        -e LLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER" \
        -e LLAMA_BUILD_COMMIT_SHORT="${LLAMA_COMMIT:0:7}" \
        -e LIBS="$LLAMA_LIBS htp-v79 test-backend-ops" \
        -e JOBS="$JOBS" \
        -e REPRO="$repro" \
        -e STAGE="$stage" \
        -e NAMES="${names[*]}" \
        -e APP_SRC="$app_src" -e APP_FILES="$app_files" \
        "$SNAPDRAGON_IMAGE" bash -euo pipefail -c '
source tools/stages/common/buildlib.sh
for name in base $NAMES; do
    tree=$STAGE/$name
    bdir=$STAGE/android-$name
    flags="$REPRO -ffile-prefix-map=/workspace/$tree=./third_party/llama.cpp -ffile-prefix-map=/workspace/$bdir=./build/native/llama"
    targets="$LIBS"
    [[ $name == base ]] && targets="$targets llama-bench llama-perplexity"
    # The DSP library is an external project that the build step configures, thus it reads CFLAGS at that time
    export CFLAGS="$flags" CXXFLAGS="$flags"
    cmake -S "$tree" --preset arm64-android-snapdragon-release -B "$bdir" \
        -DLLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER" \
        -DLLAMA_BUILD_COMMIT="$LLAMA_BUILD_COMMIT_SHORT" \
        -DLLAMA_BUILD_TESTS=ON \
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
'
) 9> build/.container.lock > "$stage/build.log" 2>&1 || die "the build failed, refer to $stage/build.log"

# 3. The stage files
rm -rf "$out"
mkdir -p "$out/bin" "$out/lib-base"
for lib in $LLAMA_LIBS llama-bench-impl llama-perplexity-impl; do
    cp -f "$stage/android-base/bin/lib$lib.so" "$out/lib-base/"
done
cp -f "$stage/android-base/ggml/src/ggml-hexagon/libggml-htp-v79.so" "$out/lib-base/"
for name in "${names[@]}"; do
    mkdir -p "$out/lib-$name"
    for lib in $LLAMA_LIBS; do
        cp -f "$stage/android-$name/bin/lib$lib.so" "$out/lib-$name/"
    done
    cp -f "$stage/android-$name/ggml/src/ggml-hexagon/libggml-htp-v79.so" "$out/lib-$name/"
    for so in "$out/lib-$name"/*.so; do
        if cmp -s "$so" "$out/lib-base/$(basename "$so")"; then
            rm -f "$so"
        fi
    done
    [[ -f $out/lib-$name/libggml-htp-v79.so ]] ||
        die "the DSP library of $name has the bytes of base: its candidates change no DSP code"
    cp -f "$stage/android-$name/bin/test-backend-ops" "$out/bin/test-backend-ops-$name"
done
cp -f "$stage/android-base/bin/llama-bench" "$stage/android-base/bin/llama-perplexity" \
    "$stage/android-base/bin/test-backend-ops" "$stage/android-base/bin/memprobe" "$out/bin/"
# The phone gives the system libraries. Each llama, ggml or mtmd library that an ELF file needs must be in the
# library directories of its runs, the check of tools/stages/unary-kl/build.sh.
check_needed() {
    local elf=$1 need
    shift
    [[ $(head -c 4 "$elf") == $'\x7fELF' ]] || return 0
    for need in $(readelf -d "$elf" | grep -o 'Shared library: \[[^]]*' | grep -o '\[.*'); do
        need=${need#[}
        case $need in
            libllama* | libggml* | libmtmd*)
                local found=0 dir
                for dir in "$@"; do
                    [[ -f $dir/$need ]] && found=1
                done
                [[ $found == 1 ]] || die "$(basename "$elf") needs $need, which is not in $*"
                ;;
        esac
    done
}
for elf in "$out"/bin/* "$out"/lib-base/*.so; do
    check_needed "$elf" "$out/lib-base"
done
for name in "${names[@]}"; do
    for elf in "$out/lib-$name"/*.so; do
        check_needed "$elf" "$out/lib-$name" "$out/lib-base"
    done
done
cp -f tools/phone/gate.sh "$out/bin/"
(cd "$out" && find bin lib-* -type f | sort | xargs sha256sum > SHA256SUMS)
cat "$out/SHA256SUMS"

# 4. The command file
python3 tools/stages/spf/stage.py --stage "$STAGE" commands
