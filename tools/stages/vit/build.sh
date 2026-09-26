#!/usr/bin/env bash
# Build the files of the phone stages "vit": the vision encoder of the 4B projector on HTP0, its op profile, its
# speed and its embeddings against the x86 oracle.
#
#   JOBS=24 [PATCHES="P1 P2"] [OUT=phone] tools/stages/vit/build.sh phone
#   tools/stages/vit/build.sh oracle
#
#   PATCHES  The llama.cpp patch files to apply after patches/series, in this order. The preset is no patch, thus
#            the tree is the series of HEAD.
#   OUT      The directory of the phone files in build/vit, phone as the preset (for example phone-head, phone-new).
#
# phone:
#   1. tests/sanitizers/llama-copy.sh makes build/vit/src, the llama.cpp tree of HEAD (the pin plus patches/series).
#      patch -p1 puts each patch of PATCHES on it.
#   2. In the Snapdragon container, under the lock build/.container.lock, CMake configures the tree into
#      build/vit/android with the preset and the compiler flags of scripts/build-native.sh (with -flto). It builds
#      the libraries of the app (LLAMA_LIBS of scripts/lib.sh), the DSP library v79, test-backend-ops and
#      llama-bench. The NDK clang++ compiles tools/vit/vitprobe.cpp.
#   3. The script copies the files into build/vit/$OUT, and writes SHA256SUMS and patches.sha256 there. It stops if a
#      program or a library needs a llama, ggml or mtmd library that is not in $OUT/lib.
# oracle:
#   g++ compiles tools/vit/vitprobe.cpp against the naive x86 oracle build/oracle-x86 (strict IEEE, no SIMD) into
#   build/vit/x86/vitprobe-oracle. The headers come from build/vit/src (make it first with the mode phone, or with
#   tests/sanitizers/llama-copy.sh build/vit/src).
#
# Time: phone about 15 minutes with JOBS=24 (the LTO links take most of it), oracle about 30 s. Disk: about 2 GB.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../scripts/lib.sh"
JOBS=${JOBS:-24}
cd "$REPO_ROOT"

readonly stage=build/vit
readonly tree=$stage/src
readonly bdir=$stage/android
readonly out=$stage/${OUT:-phone}
readonly mode=${1:-phone}
[[ ${OUT:-phone} =~ ^[A-Za-z0-9._-]+$ ]] || die "OUT must be one directory name, not ${OUT}"
readonly repro="-ffile-prefix-map=/workspace=. -fdebug-prefix-map=/workspace=. -Werror=date-time"

# Make the tree of HEAD and put the patches of PATCHES on it. patch and not git apply: git apply in the repository
# skips the files below an ignored path (build/) and still exits 0.
make_tree() {
    mkdir -p "$stage"
    tests/sanitizers/llama-copy.sh "$tree"
    patch_sums=""
    local p
    for p in ${PATCHES:-}; do
        [[ -f $p ]] || die "the patch $p does not exist"
        patch -p1 -N --dry-run --silent -d "$tree" < "$p" > /dev/null || die "$p does not apply to $tree"
        patch -p1 -N --silent --no-backup-if-mismatch -d "$tree" < "$p" || die "patch failed in $tree"
        patch_sums+="$(sha256sum < "$p" | cut -d' ' -f1)  $(basename "$p")"$'\n'
    done
    cp -f android/snapdragon/CMakeUserPresets.json "$tree/CMakeUserPresets.json"
}

build_phone() {
    make_tree
    mkdir -p "$bdir"
    local epoch
    epoch=$(source_date_epoch)
    echo "vit: SOURCE_DATE_EPOCH=$epoch JOBS=$JOBS OUT=$out"
    (
        flock 9
        container_run \
            -e SOURCE_DATE_EPOCH="$epoch" \
            -e LLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER" \
            -e LLAMA_BUILD_COMMIT_SHORT="${LLAMA_COMMIT:0:7}" \
            -e TARGETS="$LLAMA_LIBS htp-v79 test-backend-ops llama-bench" \
            -e JOBS="$JOBS" -e FLAGS_EXTRA="$repro" \
            -e TREE="$tree" -e BDIR="$bdir" \
            "$SNAPDRAGON_IMAGE" bash -euo pipefail -c '
source tools/stages/common/buildlib.sh
c_flags="$(preset_flag "$TREE" CMAKE_C_FLAGS) $FLAGS_EXTRA"
cxx_flags="$(preset_flag "$TREE" CMAKE_CXX_FLAGS) $FLAGS_EXTRA"
export CFLAGS="$FLAGS_EXTRA" CXXFLAGS="$FLAGS_EXTRA"
cmake -S "$TREE" --preset arm64-android-snapdragon-release -B "$BDIR" \
    -DLLAMA_BUILD_NUMBER="$LLAMA_BUILD_NUMBER" -DLLAMA_BUILD_COMMIT="$LLAMA_BUILD_COMMIT_SHORT" \
    -DLLAMA_BUILD_TESTS=ON -DCMAKE_C_FLAGS="$c_flags" -DCMAKE_CXX_FLAGS="$cxx_flags"
# shellcheck disable=SC2086
cmake --build "$BDIR" -j"$JOBS" --target $TARGETS
cxx=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++
# shellcheck disable=SC2086
$cxx -O2 -std=c++17 $FLAGS_EXTRA -I"$TREE/include" -I"$TREE/ggml/include" -I"$TREE/tools/mtmd" \
    tools/vit/vitprobe.cpp -o "$BDIR/bin/vitprobe" \
    -L"$BDIR/bin" -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,"\$ORIGIN/../lib"
'
    ) 9> build/.container.lock > "$stage/build-$(basename "$out").log" 2>&1 ||
        die "the build failed, refer to $stage/build-$(basename "$out").log"

    rm -rf "$out"
    mkdir -p "$out/bin" "$out/lib"
    local lib
    for lib in $LLAMA_LIBS; do
        cp -f "$bdir/bin/lib$lib.so" "$out/lib/"
    done
    cp -f "$bdir/ggml/src/ggml-hexagon/libggml-htp-v79.so" "$bdir/bin/libllama-bench-impl.so" "$out/lib/"
    cp -f "$bdir/bin/vitprobe" "$bdir/bin/test-backend-ops" "$bdir/bin/llama-bench" "$out/bin/"
    # The phone gives the system libraries. Each llama, ggml or mtmd library that an ELF file needs must be in lib/.
    local elf need
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
    [[ -s $out/patches.sha256 ]] && cat "$out/patches.sha256" || echo "vit: no patch after the series"
}

build_oracle() {
    [[ -f $tree/.llama-copy-stamp ]] || die "$tree has no tree. Run the mode phone, or tests/sanitizers/llama-copy.sh $tree"
    local oracle=build/oracle-x86
    [[ -f $oracle/bin/libmtmd.so ]] || die "$oracle has no libmtmd.so"
    mkdir -p "$stage/x86"
    g++ -O2 -std=c++17 -Wall -Wextra -ffp-contract=off -fno-fast-math -I"$tree/include" -I"$tree/ggml/include" \
        -I"$tree/tools/mtmd" tools/vit/vitprobe.cpp -o "$stage/x86/vitprobe-oracle" \
        -L"$oracle/bin" -lmtmd -lllama -lggml -lggml-base -Wl,-rpath,"$REPO_ROOT/$oracle/bin"
    echo "vit: $stage/x86/vitprobe-oracle links $oracle/bin"
}

case $mode in
    phone) build_phone ;;
    oracle) build_oracle ;;
    *) die "usage: tools/stages/vit/build.sh [phone|oracle]" ;;
esac
