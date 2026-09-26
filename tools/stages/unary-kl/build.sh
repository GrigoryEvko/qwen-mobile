#!/usr/bin/env bash
# Build the phone stage "unary-kl": the HEAD libraries against the libraries with the row change of the pointwise
# unary ops (the switch GGML_HEXAGON_UNARY_FLAT), for the KL check, the logits hashes and the speed of the change.
#
#   JOBS=16 tools/stages/unary-kl/build.sh PATCH
#
# PATCH is the llama.cpp patch of the row change. build/unary-kl/build.sh is a link to this file. The files of the
# stage go to build/unary-kl.
#
# The steps:
#   1. tests/sanitizers/llama-copy.sh makes build/unary-kl/base and build/unary-kl/src, the llama.cpp tree of HEAD
#      (the pin plus patches/series). GNU patch puts PATCH on src. The copies are not git repositories, and git apply
#      in a directory below an ignored path of this repository skips files and still exits with 0.
#   2. In the Snapdragon container, under the lock build/.container.lock, CMake configures each tree with the preset,
#      the compiler flags (with -flto) and the build number and commit of scripts/build-native.sh. It builds the
#      libraries of the app (LLAMA_LIBS of scripts/lib.sh) and the DSP library v79 of each tree, and llama-bench and
#      llama-perplexity of base. The NDK clang++ compiles memprobe (tools/memprobe/memprobe.cpp with the five app
#      sources, the recipe of tools/stages/fixed/build.sh) against the base libraries.
#   3. The script copies the files of the stage into build/unary-kl/phone: bin/, lib-base/ and lib-new/, which keeps
#      only the libraries that differ from lib-base. The runs put lib-new before lib-base. Each llama, ggml or mtmd
#      library that a program or a library needs (readelf, NEEDED) must be in lib-base, or for a file of lib-new in
#      lib-new or lib-base, else the script stops. It writes SHA256SUMS.
#   4. tools/stages/unary-kl/stage.py commands writes build/unary-kl/phone-commands.txt.
#
# Time: about 25 minutes with JOBS=16 (the LTO links take most of it). Disk: about 3 GB.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../scripts/lib.sh"
JOBS=${JOBS:-16}
cd "$REPO_ROOT"
source tools/stages/common/buildlib.sh

patch_file=${1:?usage: tools/stages/unary-kl/build.sh PATCH}
[[ -f $patch_file ]] || die "no patch file $patch_file"
patch_file=$(readlink -f "$patch_file")

readonly stage=build/unary-kl
readonly out=$stage/phone

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
echo "unary-kl: SOURCE_DATE_EPOCH=$(source_date_epoch) JOBS=$JOBS"
(
    flock 9
    # The base build also compiles memprobe against its libraries
    # shellcheck disable=SC2016
    stage_build "$stage/base" "$stage/android-base" "$LLAMA_LIBS htp-v79 llama-bench llama-perplexity" '
memprobe_build "$TREE" "$BDIR" "$BDIR/bin/memprobe" "-Wall -Wextra -Wno-unused-parameter $FLAGS_EXTRA"'
    stage_build "$stage/src" "$stage/android-src" "$LLAMA_LIBS htp-v79"
) 9> build/.container.lock > "$stage/build.log" 2>&1 || die "the build failed, refer to $stage/build.log"

# 3. The stage files
rm -rf "$out"
mkdir -p "$out/bin" "$out/lib-base" "$out/lib-new"
for lib in $LLAMA_LIBS llama-bench-impl llama-perplexity-impl; do
    cp -f "$stage/android-base/bin/lib$lib.so" "$out/lib-base/"
done
cp -f "$stage/android-base/ggml/src/ggml-hexagon/libggml-htp-v79.so" "$out/lib-base/"
for lib in $LLAMA_LIBS; do
    cp -f "$stage/android-src/bin/lib$lib.so" "$out/lib-new/"
done
cp -f "$stage/android-src/ggml/src/ggml-hexagon/libggml-htp-v79.so" "$out/lib-new/"
for so in "$out/lib-new"/*.so; do
    if cmp -s "$so" "$out/lib-base/$(basename "$so")"; then
        rm -f "$so"
    fi
done
cp -f "$stage/android-base/bin/llama-bench" "$stage/android-base/bin/llama-perplexity" \
    "$stage/android-base/bin/memprobe" "$out/bin/"
# The phone gives the system libraries. Each llama, ggml or mtmd library that an ELF file needs must be in the
# library directories of its runs, the same check as tools/stages/gdnk/build.sh.
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
for elf in "$out"/lib-new/*.so; do
    check_needed "$elf" "$out/lib-new" "$out/lib-base"
done
cp -f tools/phone/gate.sh "$out/bin/"
(cd "$out" && find bin lib-* -type f | sort | xargs sha256sum > SHA256SUMS)
cat "$out/SHA256SUMS"

# 4. The command file
python3 tools/stages/unary-kl/stage.py commands
