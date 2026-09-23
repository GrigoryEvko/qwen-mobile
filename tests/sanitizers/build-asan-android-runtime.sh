#!/usr/bin/env bash
# Build the AddressSanitizer runtime for arm64 Android from compiler-rt of
# LLVM 22.1.8.
#
# The rule for each area:
#   - Each Android ASan run uses this runtime, not the runtime of the NDK.
#     Push libclang_rt.asan-aarch64-android.so with the build, and put its
#     directory first in LD_LIBRARY_PATH.
#   - The first step of each phone ASan run starts one thread (a thread
#     self-test, for example "ops_replay --selftest-threads"). If the thread
#     start fails, stop the run and record an environment failure, not a
#     finding.
#
# Why: the ASan runtime of NDK r29 (clang 21, r563880c) signs the return
# address in its prctl interceptor (PACIASP). Bionic calls
# prctl(PR_PAC_RESET_KEYS, PR_PAC_APIAKEY) in __pthread_start of each new
# thread. That call goes through the interceptor, thus the interceptor
# authenticates its return address with the new key. On a core with FEAT_FPAC
# (the SM8750 of the OnePlus 13) the AUTIASP traps with SIGILL, before the
# start routine runs. compiler-rt 22.1.8 builds the aarch64 prctl interceptor
# with target("branch-protection=bti") (sanitizer_common_interceptors.inc),
# thus this runtime has no PAC in that function.
#
# Usage:
#   tests/sanitizers/build-asan-android-runtime.sh [--prefix DIR] [--jobs N]
#                                                   [--ndk DIR] [--verify-only]
#
#   --prefix DIR    The work directory: DIR/src (the source), DIR/build and
#                   the outputs. The preset value is
#                   build/fuzz/asan-android-runtime.
#   --jobs N        The number of parallel compile jobs. The preset value is 8.
#   --ndk DIR       The Android NDK that compiles the runtime. The preset
#                   value is the newest NDK in ~/Android/Sdk/ndk.
#   --verify-only   Do only the checks of an existing runtime.
#
# Outputs, in DIR:
#   libclang_rt.asan-aarch64-android.so         the runtime
#   libclang_rt.asan-aarch64-android.so.sha256  its sha256
#   BUILD-INFO                                  the LLVM tag and commit, and the NDK
#
# Steps:
#   1. Clone a sparse, shallow copy of llvm-project at the pinned tag into
#      DIR/src, and make sure that the commit is the pinned commit. The tag and
#      the commit are the same as the source of build-msan-libcxx.sh.
#   2. Configure compiler-rt (only ASan) with the NDK toolchain file, API 24,
#      -mbranch-protection=standard. The builtins, libunwind and libc++abi of
#      the NDK go into the runtime as static libraries, thus the runtime needs
#      only libc, libdl, libm and liblog, as the runtime of the NDK does.
#   3. Build the shared runtime, and copy it with its sha256.
#   4. Verify: the SONAME, the four dependencies, and the prctl interceptor
#      starts with "bti c" and has no PACIASP.
#
# Resources: approximately 300 MB of disk for the source, 2 minutes on
# 8 cores. No container is necessary.
#
# Exit status: 0 if the build and the checks pass, not 0 if not.

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly LLVM_TAG=llvmorg-22.1.8
readonly LLVM_COMMIT=ca7933e47d3a3451d81e72ac174dcb5aa28b59d1
readonly RUNTIME=libclang_rt.asan-aarch64-android.so
PREFIX="$REPO_ROOT/build/fuzz/asan-android-runtime"
JOBS=8
NDK=""
VERIFY_ONLY=0

# Write a message to stderr with the name of the script.
log() {
    echo "[build-asan-android-runtime] $*" >&2
}

# Write an error message to stderr and stop with exit status 1.
die() {
    log "ERROR: $*"
    exit 1
}

# Write the header comment of this script as the usage text.
print_usage() {
    local line
    while IFS= read -r line; do
        [[ "$line" == "#!"* ]] && continue
        [[ "$line" != "#"* ]] && break
        line="${line#\#}"
        echo "${line# }"
    done < "${BASH_SOURCE[0]}"
}

# Read the command-line options into the global variables.
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --prefix) PREFIX="$(realpath -m "$2")"; shift 2 ;;
            --jobs) JOBS="$2"; shift 2 ;;
            --ndk) NDK="$(realpath -m "$2")"; shift 2 ;;
            --verify-only) VERIFY_ONLY=1; shift ;;
            -h|--help) print_usage; exit 0 ;;
            *) die "The option '$1' is not known. Use --help for the usage." ;;
        esac
    done
    if [[ -z "$NDK" ]]; then
        NDK="$(find "$HOME/Android/Sdk/ndk" -mindepth 1 -maxdepth 1 -type d 2> /dev/null | sort -V | tail -n 1)"
    fi
    [[ -f "$NDK/build/cmake/android.toolchain.cmake" ]] \
        || die "No Android NDK at '$NDK'. Give one with --ndk DIR."
}

# Print the version of the NDK from its source.properties.
ndk_version() {
    local line
    while IFS= read -r line; do
        [[ "$line" == Pkg.Revision* ]] && { echo "${line#*= }"; return 0; }
    done < "$NDK/source.properties"
    echo unknown
}

# Clone the sparse LLVM source into $PREFIX/src if it is not there, and make
# sure that it is the pinned commit.
fetch_source() {
    local src="$PREFIX/src"
    if [[ ! -d "$src/.git" ]]; then
        log "Clone llvm-project $LLVM_TAG into $src (sparse and shallow)."
        mkdir -p "$PREFIX"
        git clone --quiet --depth 1 --branch "$LLVM_TAG" --filter=blob:none --no-checkout \
            https://github.com/llvm/llvm-project.git "$src"
        git -C "$src" sparse-checkout set --cone compiler-rt cmake llvm/cmake llvm/utils/llvm-lit third-party
        git -C "$src" checkout --quiet "$LLVM_TAG"
    fi
    local have
    have="$(git -C "$src" rev-parse HEAD)"
    [[ "$have" == "$LLVM_COMMIT" ]] \
        || die "$src is at $have, not at $LLVM_COMMIT ($LLVM_TAG). Remove $src, then start this script again."
    [[ -f "$src/compiler-rt/lib/sanitizer_common/sanitizer_common_interceptors.inc" ]] \
        || die "$src has no compiler-rt. Remove $src, then start this script again."
}

# Configure and build the shared runtime, then copy it with its sha256.
build_runtime() {
    local src="$PREFIX/src" build="$PREFIX/build" tc unwind
    tc="$NDK/toolchains/llvm/prebuilt/linux-x86_64"
    unwind="$(find "$tc/lib/clang" -path '*/lib/linux/aarch64/libunwind.a' | head -n 1)"
    [[ -f "$unwind" ]] || die "No aarch64 libunwind.a in $tc/lib/clang."
    rm -rf "$build"
    log "Configure compiler-rt (ASan only) into $build with the NDK $(ndk_version)."
    # COMPILER_RT_USE_BUILTINS_LIBRARY links the builtins of the NDK (the
    # outline atomics). SANITIZER_CXX_ABI=libcxxabi with the static option
    # links libc++abi.a (the RTTI of the vptr checks). COMPILER_RT_UNWINDER_LINK_LIBS
    # links libunwind.a (_Unwind_Backtrace). Without them the link has
    # undefined symbols, because the runtime links with -nodefaultlibs.
    CCACHE_DISABLE=1 cmake -G Ninja -S "$src/compiler-rt" -B "$build" \
        -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-24 -DANDROID_STL=none \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-mbranch-protection=standard" -DCMAKE_CXX_FLAGS="-mbranch-protection=standard" \
        -DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON \
        -DCOMPILER_RT_BUILD_BUILTINS=OFF -DCOMPILER_RT_BUILD_SANITIZERS=ON \
        -DCOMPILER_RT_SANITIZERS_TO_BUILD=asan \
        -DCOMPILER_RT_BUILD_XRAY=OFF -DCOMPILER_RT_BUILD_LIBFUZZER=OFF -DCOMPILER_RT_BUILD_PROFILE=OFF \
        -DCOMPILER_RT_BUILD_MEMPROF=OFF -DCOMPILER_RT_BUILD_ORC=OFF -DCOMPILER_RT_BUILD_CTX_PROFILE=OFF \
        -DCOMPILER_RT_BUILD_GWP_ASAN=OFF -DCOMPILER_RT_INCLUDE_TESTS=OFF \
        -DLLVM_CMAKE_DIR="$src/llvm/cmake/modules" \
        -DCOMPILER_RT_USE_BUILTINS_LIBRARY=ON \
        -DSANITIZER_CXX_ABI=libcxxabi -DSANITIZER_USE_STATIC_CXX_ABI=ON \
        -DCOMPILER_RT_UNWINDER_LINK_LIBS="$unwind" \
        > "$PREFIX/configure.log" 2>&1 \
        || die "The configuration failed. Refer to $PREFIX/configure.log."
    log "Build $RUNTIME."
    ninja -C "$build" -j"$JOBS" "lib/linux/$RUNTIME" > "$PREFIX/build.log" 2>&1 \
        || die "The build failed. Refer to $PREFIX/build.log."
    cp -f "$build/lib/linux/$RUNTIME" "$PREFIX/$RUNTIME"
    (cd "$PREFIX" && sha256sum "$RUNTIME" > "$RUNTIME.sha256")
    {
        echo "llvm: $LLVM_TAG $LLVM_COMMIT"
        echo "ndk: $(ndk_version) ($NDK)"
        echo "sha256: $(cut -d' ' -f1 "$PREFIX/$RUNTIME.sha256")"
    } > "$PREFIX/BUILD-INFO"
}

# Check the runtime: the SONAME, the dependencies, and the prctl interceptor.
verify_runtime() {
    local rt="$PREFIX/$RUNTIME" tc objdump readelf nm addr first
    [[ -f "$rt" ]] || die "No runtime at $rt."
    tc="$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin"
    objdump="$tc/llvm-objdump"
    readelf="$tc/llvm-readelf"
    nm="$tc/llvm-nm"
    "$readelf" -d "$rt" | grep -q "SONAME.*\[$RUNTIME\]" || die "The SONAME is not $RUNTIME."
    local needed="" line
    while IFS= read -r line; do
        [[ "$line" =~ NEEDED.*\[(.*)\] ]] && needed+="${BASH_REMATCH[1]} "
    done < <("$readelf" -d "$rt")
    needed="$(tr ' ' '\n' <<< "$needed" | sort | tr '\n' ' ')"
    [[ "$needed" == " libc.so libdl.so liblog.so libm.so " ]] \
        || die "The runtime needs '$needed', not only libc, libdl, liblog and libm."
    addr="$("$nm" -D --defined-only "$rt" | grep ' __interceptor_prctl$' | cut -d' ' -f1)"
    [[ -n "$addr" ]] || die "The runtime has no __interceptor_prctl."
    # The instructions of the interceptor up to its first return.
    local body="" first=""
    while IFS= read -r line; do
        [[ "$line" =~ ^\ +[0-9a-f]+: ]] || continue
        [[ -z "$first" ]] && first="$line"
        body+="$line"$'\n'
        [[ "$line" =~ ret$ ]] && break
    done < <("$objdump" -d --no-show-raw-insn --start-address="0x$addr" \
        --stop-address="$(printf '0x%x' $((0x$addr + 4096)))" "$rt")
    [[ "$first" == *"bti"* ]] || die "__interceptor_prctl starts with '$first', not with bti c."
    if grep -q -E 'paciasp|autiasp' <<< "$body"; then
        die "__interceptor_prctl signs its return address, thus it traps on a core with FEAT_FPAC."
    fi
    log "OK: $rt ($(cut -c1-16 "$rt.sha256")): SONAME, dependencies, and no PAC in __interceptor_prctl."
}

main() {
    parse_args "$@"
    if [[ "$VERIFY_ONLY" -eq 0 ]]; then
        fetch_source
        build_runtime
    fi
    verify_runtime
}

main "$@"
