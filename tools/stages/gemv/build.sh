#!/usr/bin/env bash
# Build the phone stage "gemv": the libraries of the app from a private tree of HEAD plus the patch of
# the packed Q8_0 GEMV tiles, the DSP libraries of HEAD without the patch (the base of the A/B runs),
# and the tools of the stage.
#
#   JOBS=24 PATCH=wip/gemv/0001-....patch tools/stages/gemv/build.sh
#
#   PATCH  The patch file of the packed tiles. It applies after patches/series.
#
# The steps:
#   1. tests/sanitizers/llama-copy.sh makes build/gemv/base, the llama.cpp tree of HEAD (the pin plus
#      patches/series). build/gemv/new is a copy of it with PATCH applied.
#   2. In the Snapdragon container, under the lock build/.container.lock, CMake configures each tree
#      with the preset, the compiler flags (with -flto) and the build number and commit of
#      scripts/build-native.sh. The new tree builds the libraries of the app (LLAMA_LIBS of
#      scripts/lib.sh), the DSP libraries v73, v75, v79 and v81, llama-bench and test-backend-ops.
#      The NDK clang++ compiles tools/gemv/gemvcheck.cpp against the libraries of the new tree, with
#      the C++ library in the program (-static-libstdc++), thus it has no NEEDED entry for libc++_shared.so. The
#      base tree builds the DSP libraries v73, v75 and v79.
#   3. The script copies the files of the stage into build/gemv/phone:
#        bin/        llama-bench, test-backend-ops, gemvcheck, gate.sh
#        lib/        the host libraries of the new tree (no DSP library)
#        dsp/new/    libggml-htp-v73.so, -v75.so and -v79.so of the new tree
#        dsp/base/   the same three libraries of the base tree
#        tests/      the test files of test-backend-ops (stage.py tests)
#      A run selects the DSP library with ADSP_LIBRARY_PATH=dsp/new or dsp/base, thus the two runs of
#      a pair use the same host libraries. The host library differs from the base only in the name of
#      the multi-row kernel in the profile lines (htp-opnode.h).
#      Then SHA256SUMS, patch.sha256, and build/gemv/phone-commands.txt (stage.py commands).
#
# The v81 library is a build check: the phone has a v79 DSP, and a newer library on an older DSP
# computes wrong values without a trap.
#
# Time: about 20 minutes with JOBS=24 (the LTO links take most of it). Disk: about 3 GB.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../scripts/lib.sh"
JOBS=${JOBS:-24}
PATCH=${PATCH:?give the patch file: PATCH=path/to/0001-....patch}
cd "$REPO_ROOT"
source tools/stages/common/buildlib.sh

readonly stage=build/gemv
readonly out=$stage/phone
readonly tool=tools/stages/gemv

[[ -f $PATCH ]] || die "the patch $PATCH does not exist"
avail=$(free -g | { read -r _; read -r _ _ _ _ _ _ a _; echo "$a"; })
[[ $avail -ge 40 ]] || die "free -g shows $avail GB available, wait for 40 GB"

# 1. The trees. patch and not git apply: git apply in the repository skips the files below an ignored
# path (build/) and still exits 0.
tests/sanitizers/llama-copy.sh "$stage/base"
rm -rf "$stage/new"
cp -a "$stage/base" "$stage/new"
patch -p1 -N --dry-run --silent -d "$stage/new" < "$PATCH" > /dev/null || die "$PATCH does not apply to $stage/new"
patch -p1 -N --silent --no-backup-if-mismatch -d "$stage/new" < "$PATCH" || die "patch failed in $stage/new"
for t in base new; do
    cp -f android/snapdragon/CMakeUserPresets.json "$stage/$t/CMakeUserPresets.json"
done

# 2. The builds.
echo "gemv: SOURCE_DATE_EPOCH=$(source_date_epoch) JOBS=$JOBS"
(
    flock 9
    # shellcheck disable=SC2016
    stage_build --tests "$stage/new" "$stage/android-new" \
        "$LLAMA_LIBS htp-v73 htp-v75 htp-v79 htp-v81 llama-bench test-backend-ops" '
"$(ndk_cxx)" -O2 -std=c++17 -Wall -Wextra $FLAGS_EXTRA -static-libstdc++ -I"$TREE/ggml/include" tools/gemv/gemvcheck.cpp \
    -o "$BDIR/bin/gemvcheck" -L"$BDIR/bin" -lggml -lggml-cpu -lggml-base -Wl,-rpath,"\$ORIGIN/../lib"'
    stage_build --tests "$stage/base" "$stage/android-base" "htp-v73 htp-v75 htp-v79"
) 9> build/.container.lock > "$stage/build.log" 2>&1 || die "the build failed, refer to $stage/build.log"

# 3. The stage files.
rm -rf "$out"
mkdir -p "$out/bin" "$out/lib" "$out/dsp/new" "$out/dsp/base" "$out/tests"
for lib in $LLAMA_LIBS llama-bench-impl; do
    cp -f "$stage/android-new/bin/lib$lib.so" "$out/lib/"
done
for t in new base; do
    for v in v73 v75 v79; do
        cp -f "$stage/android-$t/ggml/src/ggml-hexagon/libggml-htp-$v.so" "$out/dsp/$t/"
    done
done
[[ -f $stage/android-new/ggml/src/ggml-hexagon/libggml-htp-v81.so ]] || die "the v81 DSP library did not build"
cp -f "$stage/android-new/bin/llama-bench" "$stage/android-new/bin/test-backend-ops" "$stage/android-new/bin/gemvcheck" "$out/bin/"
cp -f tools/phone/gate.sh "$out/bin/"
python3 "$tool/stage.py" tests --out "$out/tests"
python3 "$tool/stage.py" commands --out "$stage/phone-commands.txt"
ln -sfn "../../$tool/stage.py" "$stage/stage.py"
ln -sfn "../../$tool/build.sh" "$stage/build.sh"
printf '%s  %s\n' "$(sha256sum < "$PATCH" | cut -d' ' -f1)" "$(basename "$PATCH")" > "$stage/patch.sha256"
(cd "$out" && sha256sum bin/* lib/* dsp/*/* tests/* > SHA256SUMS)
cat "$out/SHA256SUMS" "$stage/patch.sha256"
if cmp -s "$out/dsp/new/libggml-htp-v79.so" "$out/dsp/base/libggml-htp-v79.so"; then
    die "the new and the base v79 library have the same bytes"
fi
echo "gemv: the stage files are in $out"
