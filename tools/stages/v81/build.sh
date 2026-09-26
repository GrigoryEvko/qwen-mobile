#!/usr/bin/env bash
# Build the phone stage "v81" (the v79 phone) and the v81 silicon kit: the libraries of the app from a
# private tree of HEAD plus the v81 patches, with the DSP libraries of v79 and of v81 from the same tree.
#
#   JOBS=24 PATCHES="P1 P2" KIT=1 tools/stages/v81/build.sh
#
#   PATCHES  The patch files to apply after patches/series, in this order. The preset value is each file of
#            build/v81/patches, sorted by name. A patch that the series holds is not given.
#   KIT      1 (the preset value) also builds the ISA probe for v81 (tools/htp-lab/probe/build.sh v81) and
#            writes the kit directory build/v81/kit. 0 skips the probe and the kit.
#
# The steps:
#   1. tests/sanitizers/llama-copy.sh makes build/v81/src (HEAD plus the patches of PATCHES) and
#      build/v81/src-base (HEAD).
#   2. In the Snapdragon container, under the lock build/.container.lock, CMake configures each tree with the
#      preset, the compiler flags (with -flto) and the build number and commit of scripts/build-native.sh.
#      The new tree builds the libraries of the app (LLAMA_LIBS of scripts/lib.sh), the DSP libraries v79 and
#      v81, llama-bench, llama-perplexity and test-backend-ops. The base tree builds the DSP libraries v79 and
#      v81 only. The file prefix maps give the two trees the same paths in the objects, thus a DSP library that
#      the patches do not change has the bytes of the base library. The NDK clang++ compiles
#      tools/stages/v81/canarytime.cpp.
#   3. The script compares the DSP libraries of the two trees and writes build/v81/dsp-compare.txt. A patch that
#      changes only the v81 code must keep the v79 library byte for byte.
#   4. The script copies the files of the stage into build/v81/phone and writes SHA256SUMS and patches.sha256
#      there. It stops if a program or a library of the stage needs a llama, ggml or mtmd library (readelf,
#      NEEDED) that is not in phone/lib. phone/dsp-base holds the DSP libraries of HEAD for the A/B runs.
#   5. With KIT=1: build/v81/kit holds the same programs and libraries, the ISA probe (bin/isaprobe,
#      v81/libisaprobe_skel.so) and the corpus of the simulator census (tools/htp-lab/out-isa/isa-c, from
#      tools/htp-lab/isa/run_census.sh), for the command file of tools/stages/v81/stage.py commands --set kit.
#
# Time: about 20 minutes with JOBS=24 (the LTO links take most of it). Disk: about 3 GB.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../scripts/lib.sh"
JOBS=${JOBS:-24}
KIT=${KIT:-1}
cd "$REPO_ROOT"
source tools/stages/common/buildlib.sh

readonly stage=build/v81
readonly tree=$stage/src
readonly base=$stage/src-base
readonly bdir=$stage/android
readonly bbase=$stage/android-base
readonly out=$stage/phone
readonly kit=$stage/kit

patches=${PATCHES:-$(ls "$stage"/patches/*.patch 2> /dev/null || true)}

# 1. The trees.
mkdir -p "$stage"
tests/sanitizers/llama-copy.sh "$tree"
tests/sanitizers/llama-copy.sh "$base"
# patch and not git apply: git apply in the repository skips the files below an ignored path (build/)
# and still exits 0.
patch_sums=""
for p in $patches; do
    patch -p1 -N --dry-run --silent -d "$tree" < "$p" > /dev/null || die "$p does not apply to $tree"
    patch -p1 -N --silent --no-backup-if-mismatch -d "$tree" < "$p" || die "patch failed in $tree"
    patch_sums+="$(sha256sum < "$p" | cut -d' ' -f1)  $(basename "$p")"$'\n'
done
cp -f android/snapdragon/CMakeUserPresets.json "$tree/CMakeUserPresets.json"
cp -f android/snapdragon/CMakeUserPresets.json "$base/CMakeUserPresets.json"

# 2. The builds.
mkdir -p "$bdir" "$bbase"
echo "v81: SOURCE_DATE_EPOCH=$(source_date_epoch) JOBS=$JOBS"
(
    flock 9
    # shellcheck disable=SC2016
    stage_build --tests "$tree" "$bdir" "$LLAMA_LIBS htp-v79 htp-v81 llama-bench llama-perplexity test-backend-ops" '
"$(ndk_cxx)" -O2 -std=c++17 $FLAGS_EXTRA -I"$TREE/ggml/include" tools/stages/v81/canarytime.cpp -o "$BDIR/bin/canarytime" \
    -L"$BDIR/bin" -lggml -lggml-base -Wl,-rpath,"\$ORIGIN/../lib"' || exit
    stage_build --tests "$base" "$bbase" "htp-v79 htp-v81"
) 9> build/.container.lock > "$stage/build.log" 2>&1 || die "the build failed, refer to $stage/build.log"

# 3. The DSP libraries of the two trees.
{
    for v in v79 v81; do
        new_lib="$bdir/ggml/src/ggml-hexagon/libggml-htp-$v.so"
        base_lib="$bbase/ggml/src/ggml-hexagon/libggml-htp-$v.so"
        if cmp -s "$new_lib" "$base_lib"; then
            echo "$v: the same bytes as HEAD ($(sha256sum < "$new_lib" | cut -c1-16))"
        else
            echo "$v: other bytes than HEAD (new $(sha256sum < "$new_lib" | cut -c1-16), HEAD $(sha256sum < "$base_lib" | cut -c1-16))"
        fi
    done
} | tee "$stage/dsp-compare.txt"

# 4. The stage files.
copy_stage() {
    # copy_stage DIR: the programs and the libraries of the stage into DIR/bin and DIR/lib
    local d=$1 elf need lib
    rm -rf "$d"
    mkdir -p "$d/bin" "$d/lib"
    for lib in $LLAMA_LIBS llama-bench-impl llama-perplexity-impl; do
        cp -f "$bdir/bin/lib$lib.so" "$d/lib/"
    done
    cp -f "$bdir/ggml/src/ggml-hexagon/libggml-htp-v79.so" "$bdir/ggml/src/ggml-hexagon/libggml-htp-v81.so" "$d/lib/"
    cp -f "$bdir/bin/llama-bench" "$bdir/bin/llama-perplexity" "$bdir/bin/test-backend-ops" "$bdir/bin/canarytime" \
        "$d/bin/"
    # The phone gives the system libraries. Each llama, ggml or mtmd library that an ELF file needs must be in lib/.
    for elf in "$d"/bin/* "$d"/lib/*.so; do
        [[ $(head -c 4 "$elf") == $'\x7fELF' ]] || continue
        for need in $(readelf -d "$elf" | grep -o 'Shared library: \[[^]]*' | grep -o '\[.*'); do
            need=${need#[}
            case $need in
                libllama* | libggml* | libmtmd*)
                    [[ -f $d/lib/$need ]] || die "$(basename "$elf") needs $need, which is not in $d/lib"
                    ;;
            esac
        done
    done
    cp -f tools/phone/gate.sh "$d/bin/"
}
copy_stage "$out"
# The DSP libraries of HEAD for the A/B runs of the stage: ADSP_LIBRARY_PATH selects dsp-base or lib.
mkdir -p "$out/dsp-base"
cp -f "$bbase/ggml/src/ggml-hexagon/libggml-htp-v79.so" "$bbase/ggml/src/ggml-hexagon/libggml-htp-v81.so" "$out/dsp-base/"
(cd "$out" && sha256sum bin/* lib/* dsp-base/* > SHA256SUMS)
printf '%s' "$patch_sums" > "$out/patches.sha256"
cat "$out/SHA256SUMS"
[[ -s $out/patches.sha256 ]] && cat "$out/patches.sha256" || echo "v81: no patch after the series"

# 5. The kit.
if [[ $KIT == 1 ]]; then
    [[ -d tools/htp-lab/out-isa/isa-c ]] || die "no simulator census corpus: run ARCHS=\"v79 v81\" tools/htp-lab/isa/run_census.sh"
    tools/htp-lab/probe/build.sh v81 > "$stage/isaprobe-build.log" 2>&1 \
        || die "the ISA probe build failed, refer to $stage/isaprobe-build.log"
    copy_stage "$kit"
    cp -f build/isaprobe/bin/isaprobe "$kit/bin/"
    mkdir -p "$kit/isaprobe/v81" "$kit/corpus"
    cp -f build/isaprobe/v81/libisaprobe_skel.so "$kit/isaprobe/v81/"
    cp -f tools/htp-lab/out-isa/isa-c/corpus_*.bin "$kit/corpus/"
    (cd "$kit" && sha256sum bin/* lib/* isaprobe/v81/* corpus/* > SHA256SUMS)
    printf '%s' "$patch_sums" > "$kit/patches.sha256"
    echo "v81: kit in $kit ($(du -sh "$kit" | cut -f1))"
fi
