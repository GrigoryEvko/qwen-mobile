#!/usr/bin/env bash
# Build the phone stage "unary-tg": the decode speed of the landed row change of the pointwise unary ops
# (patches/hexagon-host/0013), with GGML_HEXAGON_UNARY_FLAT=0 against the preset 1 on one library set.
#
#   JOBS=16 tools/stages/unary-tg/build.sh
#
# build/unary-tg/build.sh is a link to this file. The files of the stage go to build/unary-tg.
#
# The steps:
#   1. tests/sanitizers/llama-copy.sh makes build/unary-tg/src, the llama.cpp tree of HEAD (the pin plus
#      patches/series). The tree must have the switch GGML_HEXAGON_UNARY_FLAT.
#   2. In the Snapdragon container, under the lock build/.container.lock, CMake configures the tree with the preset,
#      the compiler flags (with -flto) and the build number and commit of scripts/build-native.sh. It builds the
#      libraries of the app (LLAMA_LIBS of scripts/lib.sh), the DSP library v79 and llama-bench.
#   3. The script copies the files of the stage into build/unary-tg/phone. It stops if a program or a library of the
#      stage needs a llama, ggml or mtmd library (readelf, NEEDED) that is not in phone/lib, the check of
#      tools/stages/fuse-mm/build.sh. It writes SHA256SUMS.
#   4. tools/stages/unary-tg/stage.py commands writes build/unary-tg/phone-commands.txt.
#
# Time: about 12 minutes with JOBS=16 (the LTO links take most of it). Disk: about 1.5 GB.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../scripts/lib.sh"
JOBS=${JOBS:-16}
cd "$REPO_ROOT"
source tools/stages/common/buildlib.sh

readonly stage=build/unary-tg
readonly tree=$stage/src
readonly bdir=$stage/android
readonly out=$stage/phone

# 1. The tree
mkdir -p "$stage"
tests/sanitizers/llama-copy.sh "$tree"
grep -q GGML_HEXAGON_UNARY_FLAT "$tree/ggml/src/ggml-hexagon/ggml-hexagon.cpp" ||
    die "$tree has no switch GGML_HEXAGON_UNARY_FLAT: patches/series of HEAD does not hold the row change"
cp -f android/snapdragon/CMakeUserPresets.json "$tree/CMakeUserPresets.json"
git log -1 --format='%H %s' > "$stage/head.txt"

# 2. The build
mkdir -p "$bdir"
echo "unary-tg: SOURCE_DATE_EPOCH=$(source_date_epoch) JOBS=$JOBS"
(
    flock 9
    stage_build "$tree" "$bdir" "$LLAMA_LIBS htp-v79 llama-bench"
) 9> build/.container.lock > "$stage/build.log" 2>&1 || die "the build failed, refer to $stage/build.log"

# 3. The stage files
rm -rf "$out"
mkdir -p "$out/bin" "$out/lib"
for lib in $LLAMA_LIBS llama-bench-impl; do
    cp -f "$bdir/bin/lib$lib.so" "$out/lib/"
done
cp -f "$bdir/ggml/src/ggml-hexagon/libggml-htp-v79.so" "$out/lib/"
cp -f "$bdir/bin/llama-bench" "$out/bin/"
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
cat "$out/SHA256SUMS"

# 4. The command file
python3 tools/stages/unary-tg/stage.py commands
