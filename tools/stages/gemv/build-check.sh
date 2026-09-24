#!/usr/bin/env bash
# Build the phone stage "gemv-check": the bit check runs of the stage gemv (stage.py --check-only), with the
# gemvcheck of the work tree and the libraries of build/gemv/phone.
#
#   tools/stages/gemv/build-check.sh
#
# Run tools/stages/gemv/build.sh first: this script uses its trees and its stage files. Thus the check runs use
# the same library bytes as the runs of the stage gemv.
#
# The steps:
#   1. Each host library of the stage (stage.py libs --check-only) in build/gemv/phone/lib must have the bytes of
#      the library in build/gemv/android-new/bin, the libraries that gemvcheck links against.
#   2. In the Snapdragon container, under the lock build/.container.lock, the NDK clang++ compiles
#      tools/gemv/gemvcheck.cpp against build/gemv/new/ggml/include and build/gemv/android-new/bin. The C++
#      library goes into the program (-static-libstdc++), as in the libraries of the app, thus the program
#      has no NEEDED entry for libc++_shared.so.
#   3. The script copies the files of the stage into build/gemv-check/phone:
#        bin/        gemvcheck, gate.sh
#        lib/        the host libraries of step 1
#        dsp/new/    libggml-htp-v73.so, -v75.so and -v79.so of build/gemv/phone/dsp/new
#        dsp/base/   the same three libraries of build/gemv/phone/dsp/base
#      It reads the NEEDED entries of each program and each library of the stage, and it stops when an entry
#      names a llama, ggml or mtmd library, or libc++_shared.so, that is not in phone/lib.
#      Then SHA256SUMS, patch.sha256 (a copy of build/gemv/patch.sha256), and
#      build/gemv-check/phone-commands.txt (stage.py commands --check-only).
#
# Time: about 1 minute. Disk: about 50 MB.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../scripts/lib.sh"
cd "$REPO_ROOT"

readonly src=build/gemv
readonly stage=build/gemv-check
readonly out=$stage/phone
readonly tool=tools/stages/gemv
readonly repro="-ffile-prefix-map=/workspace=. -fdebug-prefix-map=/workspace=. -Werror=date-time"

# Stop when a program or a library in $1/bin or $1/lib has a NEEDED entry for a llama, ggml or mtmd library, or for
# libc++_shared.so, that is not in $1/lib. The files that are not ELF files (gate.sh) have no NEEDED entry.
# O(files).
check_needed() {
    local dir=$1 f line lib bad=0
    for f in "$dir"/bin/* "$dir"/lib/*; do
        readelf -h "$f" > /dev/null 2>&1 || continue
        while read -r line; do
            [[ $line =~ \(NEEDED\).*\[([^]]+)\] ]] || continue
            lib=${BASH_REMATCH[1]}
            case $lib in
                libllama*|libggml*|libmtmd*|libc++_shared.so)
                    if [[ ! -f $dir/lib/$lib ]]; then
                        echo "gemv-check: ${f#"$dir"/} has a NEEDED entry for $lib, and $dir/lib does not have it" >&2
                        bad=1
                    fi ;;
            esac
        done < <(readelf -W -d "$f")
    done
    [[ $bad == 0 ]] || die "a program or a library of $dir has a NEEDED entry for a library that is not in $dir/lib"
}

# 1. The inputs from build.sh
for f in "$src/phone/SHA256SUMS" "$src/patch.sha256" "$src/new/ggml/include/ggml.h" "$src/android-new/bin/libggml.so"; do
    [[ -f $f ]] || die "$f does not exist: run tools/stages/gemv/build.sh first"
done
mapfile -t libs < <(python3 "$tool/stage.py" libs --check-only)
[[ ${#libs[@]} -gt 0 ]] || die "stage.py libs --check-only gave no library"
for lib in "${libs[@]}"; do
    cmp -s "$src/phone/lib/$lib" "$src/android-new/bin/$lib" \
        || die "$src/phone/lib/$lib does not have the bytes of $src/android-new/bin/$lib"
done
avail=$(free -g | { read -r _; read -r _ _ _ _ _ _ a _; echo "$a"; })
[[ $avail -ge 40 ]] || die "free -g shows $avail GB available, wait for 40 GB"

# 2. gemvcheck
rm -rf "${stage:?}/bin"
mkdir -p "$stage/bin"
(
    flock 9
    container_run -e FLAGS_EXTRA="$repro" "$SNAPDRAGON_IMAGE" bash -euo pipefail -c '
cxx=$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++
$cxx -O2 -std=c++17 -Wall -Wextra -Werror $FLAGS_EXTRA -static-libstdc++ -Ibuild/gemv/new/ggml/include \
    tools/gemv/gemvcheck.cpp -o build/gemv-check/bin/gemvcheck \
    -Lbuild/gemv/android-new/bin -lggml -lggml-cpu -lggml-base -Wl,-rpath,"\$ORIGIN/../lib"
'
) 9> build/.container.lock > "$stage/build.log" 2>&1 || die "the build failed, refer to $stage/build.log"
# -W: without it readelf cuts the long symbol names. No grep -q in a pipe: with pipefail, readelf can get
# SIGPIPE when grep stops early.
syms=$(readelf -W --dyn-syms "$stage/bin/gemvcheck")
[[ $syms == *" ggml_backend_buffer_set_usage"* ]] \
    || die "$stage/bin/gemvcheck does not call ggml_backend_buffer_set_usage: without it the backend does not repack the weights"

# 3. The stage files
rm -rf "${out:?}"
mkdir -p "$out/bin" "$out/lib" "$out/dsp/new" "$out/dsp/base"
cp -f "$stage/bin/gemvcheck" tools/phone/gate.sh "$out/bin/"
for lib in "${libs[@]}"; do
    cp -f "$src/phone/lib/$lib" "$out/lib/"
done
for v in new base; do
    for a in v73 v75 v79; do
        cp -f "$src/phone/dsp/$v/libggml-htp-$a.so" "$out/dsp/$v/"
    done
done
check_needed "$out"
if cmp -s "$out/dsp/new/libggml-htp-v79.so" "$out/dsp/base/libggml-htp-v79.so"; then
    die "the new and the base v79 library have the same bytes"
fi
python3 "$tool/stage.py" commands --check-only --out "$stage/phone-commands.txt"
ln -sfn "../../$tool/stage.py" "$stage/stage.py"
ln -sfn "../../$tool/build-check.sh" "$stage/build-check.sh"
cp -f "$src/patch.sha256" "$stage/patch.sha256"
(cd "$out" && sha256sum bin/* lib/* dsp/*/* > SHA256SUMS)
cat "$out/SHA256SUMS" "$stage/patch.sha256"
echo "gemv-check: the stage files are in $out, the commands in $stage/phone-commands.txt"
