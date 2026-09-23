#!/usr/bin/env bash
# Build llvm-symbolizer of LLVM 22.1.8 for arm64 Android.
#
# The sanitizer runtimes on the phone give function names and lines to a
# report only with a symbolizer on the phone. Without names, no
# function-level suppression entry can match a report frame. The NDK has
# llvm-symbolizer for the host only, thus this script cross-compiles it.
#
# Usage:
#   tests/sanitizers/build-android-symbolizer.sh [--prefix DIR] [--jobs N] [--ndk DIR]
#
#   --prefix DIR   The work directory: DIR/src (the sparse LLVM source),
#                  DIR/build and the output. The preset value is
#                  build/fuzz/android-symbolizer.
#   --jobs N       The number of parallel compile jobs. The preset value is 8.
#   --ndk DIR      The Android NDK. The preset value is the output of
#                  tests/sanitizers/fetch-android-ndk.sh (NDK r30).
#
# Output: DIR/llvm-symbolizer (stripped, arm64 Android, API 34, static libc++)
# and DIR/llvm-symbolizer.sha256.
#
# The host needs llvm-tblgen of the same LLVM version (22.1.8), because a
# cross build of LLVM runs the table generator on the host.
#
# Resources: approximately 400 MB of disk, 3 minutes on 24 cores, 4 GB RAM.
# No container is necessary.
#
# Exit status: 0 if the build and the check pass, not 0 if not.

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly LLVM_TAG=llvmorg-22.1.8
readonly LLVM_COMMIT=ca7933e47d3a3451d81e72ac174dcb5aa28b59d1
PREFIX="$REPO_ROOT/build/fuzz/android-symbolizer"
JOBS=8
NDK=""

# Write a message to stderr with the name of the script.
log() {
    echo "[build-android-symbolizer] $*" >&2
}

# Write an error message to stderr and stop with exit status 1.
die() {
    log "ERROR: $*"
    exit 1
}

# Read the command-line options into the global variables.
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --prefix) PREFIX="$(realpath -m "$2")"; shift 2 ;;
            --jobs) JOBS="$2"; shift 2 ;;
            --ndk) NDK="$(realpath -m "$2")"; shift 2 ;;
            -h|--help) head -28 "${BASH_SOURCE[0]}" | tail -27; exit 0 ;;
            *) die "The option '$1' is not known. Use --help for the usage." ;;
        esac
    done
    if [[ -z "$NDK" ]]; then
        NDK="$("$REPO_ROOT/tests/sanitizers/fetch-android-ndk.sh" | tail -n 1)"
    fi
    [[ -f "$NDK/build/cmake/android.toolchain.cmake" ]] || die "No Android NDK at '$NDK'. Give one with --ndk DIR."
}

# Clone the sparse LLVM source into $PREFIX/src if it is not there, and make
# sure that it is the pinned commit.
fetch_source() {
    local src="$PREFIX/src" have
    if [[ ! -d "$src/.git" ]]; then
        log "Clone llvm-project $LLVM_TAG into $src (sparse and shallow)."
        mkdir -p "$PREFIX"
        git clone --quiet --depth 1 --branch "$LLVM_TAG" --filter=blob:none --no-checkout \
            https://github.com/llvm/llvm-project.git "$src"
        git -C "$src" sparse-checkout set --cone llvm/cmake llvm/include llvm/lib \
            llvm/tools/llvm-symbolizer llvm/tools/llvm-config llvm/utils llvm/bindings llvm/runtimes \
            llvm/projects llvm/resources cmake third-party
        git -C "$src" checkout --quiet "$LLVM_TAG"
    fi
    have="$(git -C "$src" rev-parse HEAD)"
    [[ "$have" == "$LLVM_COMMIT" ]] \
        || die "$src is at $have, not at $LLVM_COMMIT ($LLVM_TAG). Remove $src, then start this script again."
}

main() {
    parse_args "$@"
    command -v cmake > /dev/null || die "cmake is not in PATH."
    command -v ninja > /dev/null || die "ninja is not in PATH."
    local tblgen ver build="$PREFIX/build" out="$PREFIX/llvm-symbolizer" t0=$SECONDS
    tblgen="$(command -v llvm-tblgen)" || die "llvm-tblgen of LLVM 22.1.8 is necessary on the host (Fedora: the package llvm)."
    ver="$("$tblgen" --version | rg -o '[0-9]+\.[0-9]+\.[0-9]+' | head -n 1)"
    [[ "llvmorg-$ver" == "$LLVM_TAG" ]] || die "$tblgen is LLVM $ver, not ${LLVM_TAG#llvmorg-}."
    fetch_source
    rm -rf "$build"
    log "Configure into $build with the NDK $NDK."
    CCACHE_DISABLE=1 cmake -S "$PREFIX/src/llvm" -B "$build" -G Ninja \
        -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" -DANDROID_ABI=arm64-v8a \
        -DANDROID_PLATFORM=android-34 -DANDROID_STL=c++_static -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=OFF -DLLVM_TARGETS_TO_BUILD=AArch64 -DLLVM_HOST_TRIPLE=aarch64-linux-android \
        -DLLVM_TABLEGEN="$tblgen" -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF \
        -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_INCLUDE_DOCS=OFF -DLLVM_INCLUDE_UTILS=OFF -DLLVM_BUILD_UTILS=OFF \
        -DLLVM_ENABLE_ZLIB=OFF -DLLVM_ENABLE_ZSTD=OFF -DLLVM_ENABLE_LIBXML2=OFF -DLLVM_ENABLE_TERMINFO=OFF \
        -DLLVM_ENABLE_LIBEDIT=OFF -DLLVM_ENABLE_LIBPFM=OFF -DLLVM_ENABLE_CURL=OFF -DLLVM_ENABLE_HTTPLIB=OFF \
        -DLLVM_ENABLE_BINDINGS=OFF > "$PREFIX/configure.log" 2>&1 \
        || { tail -30 "$PREFIX/configure.log" >&2; die "The configuration failed. Refer to $PREFIX/configure.log."; }
    log "Build with $JOBS jobs."
    CCACHE_DISABLE=1 nice -n 10 ninja -C "$build" -j "$JOBS" llvm-symbolizer > "$PREFIX/build.log" 2>&1 \
        || { tail -30 "$PREFIX/build.log" >&2; die "The build failed. Refer to $PREFIX/build.log."; }
    "$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" -o "$out" "$build/bin/llvm-symbolizer"
    # The check: an arm64 Android executable that needs only the libraries of
    # the platform (libc++ is static).
    local readelf="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf" needed
    "$readelf" -h "$out" | rg -q 'Machine:\s+AArch64' || die "$out is not an AArch64 executable."
    needed="$("$readelf" -d "$out" | rg -o -r '$1' 'NEEDED.*\[(.+)\]' | sort | tr '\n' ' ')"
    [[ "$needed" != *"c++_shared"* ]] || die "$out needs libc++_shared.so: $needed"
    (cd "$PREFIX" && sha256sum llvm-symbolizer > llvm-symbolizer.sha256)
    log "OK: $out ($(cut -c1-16 "$out.sha256"), needs: $needed) in $((SECONDS - t0)) s."
}

main "$@"
