#!/usr/bin/env bash
# Build the phone stage "fuse-mm": the libraries of the app from a private tree of HEAD plus the matmul
# fusion patches, and the tools of the stage against them.
#
#   JOBS=24 PATCHES="P1 P2" OUT=phone tools/stages/fuse-mm/build.sh
#
#   PATCHES  The patch files to apply after patches/series, in this order. The preset value is each
#            file of build/fuse-mm/patches, sorted by name. A patch that the series holds is not given.
#   OUT      The directory of the stage files in build/fuse-mm, phone as the preset. A run set of stage.py
#            names its directory (the set repack has phone-repack).
#
# The steps:
#   1. tests/sanitizers/llama-copy.sh makes build/fuse-mm/src, the llama.cpp tree of HEAD (the pin
#      plus patches/series). patch -p1 puts each patch of PATCHES on it.
#   2. In the Snapdragon container, under the lock build/.container.lock, CMake configures the tree into
#      build/fuse-mm/android with the preset, the compiler flags (with -flto) and the build number and
#      commit of scripts/build-native.sh. It builds the libraries of the app (LLAMA_LIBS of
#      scripts/lib.sh), the DSP library v79, llama-bench, llama-perplexity and test-backend-ops. The NDK
#      clang++ compiles tools/stages/fuse-mm/ffncheck.cpp.
#   3. The script copies the files of the stage into build/fuse-mm/$OUT and writes SHA256SUMS and
#      patches.sha256 (the patch files of the build) there. It stops if a program or a library of the stage
#      needs a llama or ggml library (readelf, NEEDED) that is not in phone/lib.
#
# The switches of the fusions (GGML_HEXAGON_FUSE_SWIGLU, _SWIGLU_DECODE, _F16_ACT) are environment
# variables, thus one library set gives each variant of the A/B runs.
#
# Time: about 15 minutes with JOBS=24 (the LTO links take most of it). Disk: about 2 GB.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../scripts/lib.sh"
JOBS=${JOBS:-24}
cd "$REPO_ROOT"

readonly stage=build/fuse-mm
readonly tree=$stage/src
readonly bdir=$stage/android
readonly out=$stage/${OUT:-phone}
[[ ${OUT:-phone} =~ ^[A-Za-z0-9._-]+$ ]] || die "OUT must be one directory name, not ${OUT}"
readonly repro="-ffile-prefix-map=/workspace=. -fdebug-prefix-map=/workspace=. -Werror=date-time"
readonly remap="-ffile-prefix-map=/workspace/$tree=./third_party/llama.cpp -ffile-prefix-map=/workspace/$bdir=./build/native/llama"

patches=${PATCHES:-$(ls "$stage"/patches/*.patch 2> /dev/null || true)}

# 1. The tree.
mkdir -p "$stage"
tests/sanitizers/llama-copy.sh "$tree"
# patch and not git apply: git apply in the repository skips the files below an ignored path (build/)
# and still exits 0.
patch_sums=""
for p in $patches; do
    patch -p1 -N --dry-run --silent -d "$tree" < "$p" > /dev/null || die "$p does not apply to $tree"
    patch -p1 -N --silent --no-backup-if-mismatch -d "$tree" < "$p" || die "patch failed in $tree"
    patch_sums+="$(sha256sum < "$p" | cut -d' ' -f1)  $(basename "$p")"$'\n'
done
cp -f android/snapdragon/CMakeUserPresets.json "$tree/CMakeUserPresets.json"

# 2. The build.
mkdir -p "$bdir"
SOURCE_DATE_EPOCH=$(source_date_epoch)
echo "fuse-mm: SOURCE_DATE_EPOCH=$SOURCE_DATE_EPOCH JOBS=$JOBS"
(
    flock 9
    container_run \
        -e SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
        -e LLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER" \
        -e LLAMA_BUILD_COMMIT_SHORT="${LLAMA_COMMIT:0:7}" \
        -e TARGETS="$LLAMA_LIBS htp-v79 llama-bench llama-perplexity test-backend-ops" \
        -e JOBS="$JOBS" \
        -e FLAGS_EXTRA="$repro $remap" \
        -e TREE="$tree" -e BDIR="$bdir" \
        "$SNAPDRAGON_IMAGE" bash -euo pipefail -c '
source tools/stages/common/buildlib.sh
c_flags="$(preset_flag "$TREE" CMAKE_C_FLAGS) $FLAGS_EXTRA"
cxx_flags="$(preset_flag "$TREE" CMAKE_CXX_FLAGS) $FLAGS_EXTRA"
export CFLAGS="$FLAGS_EXTRA" CXXFLAGS="$FLAGS_EXTRA"
cmake -S "$TREE" --preset arm64-android-snapdragon-release -B "$BDIR" \
    -DLLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER" \
    -DLLAMA_BUILD_COMMIT="$LLAMA_BUILD_COMMIT_SHORT" \
    -DLLAMA_BUILD_TESTS=ON \
    -DCMAKE_C_FLAGS="$c_flags" \
    -DCMAKE_CXX_FLAGS="$cxx_flags"
# shellcheck disable=SC2086
cmake --build "$BDIR" -j"$JOBS" --target $TARGETS
cxx=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++
$cxx -O2 -std=c++17 $FLAGS_EXTRA -I"$TREE/ggml/include" tools/stages/fuse-mm/ffncheck.cpp -o "$BDIR/bin/ffncheck" \
    -L"$BDIR/bin" -lggml -lggml-cpu -lggml-base -Wl,-rpath,"\$ORIGIN/../lib"
'
) 9> build/.container.lock > "$stage/build.log" 2>&1 || die "the build failed, refer to $stage/build.log"

# 3. The stage files.
rm -rf "$out"
mkdir -p "$out/bin" "$out/lib"
for lib in $LLAMA_LIBS llama-bench-impl llama-perplexity-impl; do
    cp -f "$bdir/bin/lib$lib.so" "$out/lib/"
done
cp -f "$bdir/ggml/src/ggml-hexagon/libggml-htp-v79.so" "$out/lib/"
cp -f "$bdir/bin/llama-bench" "$bdir/bin/llama-perplexity" "$bdir/bin/test-backend-ops" "$bdir/bin/ffncheck" "$out/bin/"
# The phone gives the system libraries. Each llama, ggml or mtmd library that an ELF file needs must be in lib/.
for elf in "$out"/bin/* "$out"/lib/*.so; do
    [[ $(head -c 4 "$elf") == $'\x7fELF' ]] || continue
    for need in $(readelf -d "$elf" | grep -o 'Shared library: \[[^]]*' | grep -o '\[.*'); do
        need=${need#[}
        case $need in
            libllama* | libggml* | libmtmd*)
                [[ -f $out/lib/$need ]] || die "$(basename "$elf") needs $need, which is not in $out/lib"
                ;;
        esac
    done
done
cp -f tools/phone/gate.sh "$out/bin/"
(cd "$out" && sha256sum bin/* lib/* > SHA256SUMS)
printf '%s' "$patch_sums" > "$out/patches.sha256"
cat "$out/SHA256SUMS"
[[ -s $out/patches.sha256 ]] && cat "$out/patches.sha256" || echo "fuse-mm: no patch after the series"
