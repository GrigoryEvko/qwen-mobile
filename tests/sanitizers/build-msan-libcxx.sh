#!/usr/bin/env bash
# Build and install an MSan-instrumented libc++ and libc++abi.
#
# MemorySanitizer (MSan) reports a false "use-of-uninitialized-value" when
# the program calls a C++ standard library that has no MSan instrumentation.
# Thus, each MSan build in this repository links this libc++ and not the
# libstdc++ of the system. The LLVM tag must agree with the installed clang,
# because the headers and the MSan runtime of clang must agree.
#
# Usage:
#   tests/sanitizers/build-msan-libcxx.sh [--prefix DIR] [--jobs N] [--verify-only]
#
#   --prefix DIR    The work directory. The script puts the source in DIR/src,
#                   the build in DIR/build and the install in DIR/install.
#                   The preset value is build/fuzz/msan-libcxx.
#   --jobs N        The number of parallel compile jobs. The preset value is 8.
#   --verify-only   Do only the verification of an installed libc++.
#
# Environment:
#   CC, CXX         The clang compilers. The preset values are clang, clang++.
#   LLVM_TAG        The LLVM git tag. The preset value comes from the version
#                   of $CC, for example llvmorg-22.1.8.
#
# Steps (each step starts from an empty directory when necessary):
#   1. Clone a sparse, shallow copy of llvm-project at LLVM_TAG into DIR/src.
#   2. Configure the "runtimes" project in a clean DIR/build.
#   3. Build and install into DIR/install.
#   4. Verify: a clean C++ program gives no MSan report, and a program that
#      reads uninitialized memory gives one report at the correct line.
#
# Resources: approximately 250 MB of disk for the source, 5 minutes on
# 8 cores, and 2 GB of RAM. No container is necessary.
#
# Exit status: 0 if the install and the verification pass, not 0 if not.

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFIX="$REPO_ROOT/build/fuzz/msan-libcxx"
JOBS=8
VERIFY_ONLY=0

# Write a message to stderr with the name of the script.
log() {
    echo "[build-msan-libcxx] $*" >&2
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
            --verify-only) VERIFY_ONLY=1; shift ;;
            -h|--help) print_usage; exit 0 ;;
            *) die "The option '$1' is not known. Use --help for the usage." ;;
        esac
    done
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

# Find the LLVM tag that agrees with the version of the clang compiler.
# Print the tag, for example llvmorg-22.1.8.
llvm_tag_of_clang() {
    local version
    version="$("$CC" -dumpversion)" || die "The compiler '$CC' does not run."
    [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
        || die "The compiler '$CC' gives the version '$version'. A full x.y.z version is necessary."
    echo "llvmorg-$version"
}

# Clone the sparse LLVM source into $PREFIX/src if it is not there.
# If a source is there, make sure that it has the correct tag.
fetch_source() {
    local src="$PREFIX/src"
    if [[ -d "$src/.git" ]]; then
        local have
        have="$(git -C "$src" describe --tags --exact-match 2>/dev/null || true)"
        [[ "$have" == "$LLVM_TAG" ]] \
            || die "$src has the tag '$have', not '$LLVM_TAG'. Remove $src, then start this script again."
        log "The source at $src has the tag $LLVM_TAG."
    else
        log "Clone llvm-project $LLVM_TAG into $src (sparse and shallow)."
        mkdir -p "$PREFIX"
        git clone --quiet --depth 1 --branch "$LLVM_TAG" --filter=blob:none --no-checkout \
            https://github.com/llvm/llvm-project.git "$src"
        git -C "$src" sparse-checkout set --cone \
            runtimes libcxx libcxxabi libunwind cmake llvm/cmake llvm/utils/llvm-lit third-party libc
        git -C "$src" checkout --quiet "$LLVM_TAG"
    fi
    # libcxx/src/charconv.cpp includes libc/shared/fp_bits.h. Without libc/
    # in the checkout, the build stops at that file.
    [[ -f "$src/libc/shared/fp_bits.h" ]] \
        || die "$src/libc/shared/fp_bits.h is missing. Add libc to the sparse checkout."
}

# Configure, build and install the runtimes into $PREFIX/install.
# The build directory is always new, because a cache from a different
# checkout can keep incorrect paths.
build_and_install() {
    local build="$PREFIX/build" install="$PREFIX/install"
    rm -rf "$build" "$install"
    log "Configure into $build."
    # CCACHE_DISABLE=1 keeps these sanitizer objects out of the ccache of the
    # user. LLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF gives the flat layout
    # (include/c++/v1 and lib/) that FUZZ_MSAN_PREFIX expects.
    #
    # libunwind is not in the runtimes, and libc++abi uses the unwinder of
    # the system (libgcc_s). With an MSan libunwind, the MSan runtime calls
    # _Unwind_Backtrace of that libunwind to print a report. libunwind reads
    # the registers that its assembly saved, MSan reports that read, and the
    # recursion stops with "stack-overflow" and no report. The upstream cache
    # libcxx/cmake/caches/Generic-msan.cmake also sets
    # LIBCXXABI_USE_LLVM_UNWINDER=OFF for MSan.
    CCACHE_DISABLE=1 cmake -G Ninja -S "$PREFIX/src/runtimes" -B "$build" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$CC" -DCMAKE_CXX_COMPILER="$CXX" \
        -DCMAKE_C_COMPILER_LAUNCHER= -DCMAKE_CXX_COMPILER_LAUNCHER= \
        -DCMAKE_INSTALL_PREFIX="$install" \
        -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi" \
        -DLIBCXXABI_USE_LLVM_UNWINDER=OFF \
        -DLLVM_USE_SANITIZER=MemoryWithOrigins \
        -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF \
        -DLLVM_INCLUDE_TESTS=OFF \
        -DLIBCXX_INCLUDE_TESTS=OFF -DLIBCXXABI_INCLUDE_TESTS=OFF -DLIBUNWIND_INCLUDE_TESTS=OFF \
        -DLIBCXX_INCLUDE_BENCHMARKS=OFF \
        > "$PREFIX/cmake.log" 2>&1 \
        || { tail -40 "$PREFIX/cmake.log" >&2; die "The configure step failed. Refer to $PREFIX/cmake.log."; }
    log "Build with $JOBS jobs."
    CCACHE_DISABLE=1 nice -n 10 cmake --build "$build" -j "$JOBS" > "$PREFIX/ninja.log" 2>&1 \
        || { tail -40 "$PREFIX/ninja.log" >&2; die "The build step failed. Refer to $PREFIX/ninja.log."; }
    cmake --install "$build" > "$PREFIX/install.log" 2>&1 \
        || { tail -20 "$PREFIX/install.log" >&2; die "The install step failed. Refer to $PREFIX/install.log."; }
    log "The install is in $install."
}

# Compile one verification program with MSan and this libc++.
# Arguments: the source file, the output binary.
compile_msan_program() {
    local install="$PREFIX/install"
    CCACHE_DISABLE=1 "$CXX" -std=c++17 -g -O1 -fno-omit-frame-pointer \
        -fsanitize=memory -fsanitize-memory-track-origins=2 \
        -stdlib=libc++ -nostdinc++ -isystem "$install/include/c++/v1" \
        "$1" -o "$2" \
        -L"$install/lib" -Wl,-rpath,"$install/lib" -lc++ -lc++abi
}

# Verify the install with two programs. The first program uses
# std::vector<std::string>, std::stringstream, std::map and a C++ exception,
# and must give no MSan report. The second program reads uninitialized
# memory, and MSan must report it at the line that has the marker
# "MSAN-EXPECTED-LINE".
verify() {
    local install="$PREFIX/install" work
    [[ -d "$install/include/c++/v1" && -e "$install/lib/libc++.so" ]] \
        || die "$install does not contain include/c++/v1 and lib/libc++.so. Build it first."
    work="$(mktemp -d)"
    VERIFY_WORK="$work"
    # The EXIT trap removes the temporary directory, also after die().
    trap 'rm -rf "$VERIFY_WORK"' EXIT

    cat > "$work/clean.cpp" <<'EOF'
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
static int thrower(int x) {
    if (x > 0) throw std::runtime_error("value " + std::to_string(x));
    return x;
}
int main(int argc, char **) {
    std::vector<std::string> words = {"alpha", "beta", "gamma", "alpha"};
    std::map<std::string, int> count;
    for (const auto &w : words) count[w]++;
    std::stringstream out;
    for (const auto &kv : count) out << kv.first << '=' << kv.second << ';';
    int n = 0;
    std::stringstream in("17 25");
    int a = 0, b = 0;
    in >> a >> b;
    n = a + b;
    std::string caught;
    try { thrower(argc); } catch (const std::exception &e) { caught = e.what(); }
    return (out.str() == "alpha=2;beta=1;gamma=1;" && n == 42 && caught == "value 1") ? 0 : 3;
}
EOF
    cat > "$work/uninit.cpp" <<'EOF'
#include <cstdio>
#include <vector>
int main(int argc, char **argv) {
    (void)argv;
    int *p = new int[8];
    std::vector<int> v(p, p + 8);
    if (v[argc] > 0)  // MSAN-EXPECTED-LINE
        std::puts("positive");
    delete[] p;
    return 0;
}
EOF
    compile_msan_program "$work/clean.cpp" "$work/clean" || die "The clean program does not compile."
    compile_msan_program "$work/uninit.cpp" "$work/uninit" || die "The uninit program does not compile."

    local log_clean="$work/clean.log" log_uninit="$work/uninit.log" rc=0
    MSAN_OPTIONS=halt_on_error=1 "$work/clean" > "$log_clean" 2>&1 || rc=$?
    if [[ $rc -ne 0 ]] || rg -q 'MemorySanitizer' "$log_clean"; then
        cat "$log_clean" >&2
        die "The clean program gives exit status $rc or an MSan report. The libc++ is not correctly instrumented."
    fi
    log "PASS: the clean program (vector<string>, stringstream, map, exception) gives no MSan report."

    local line
    line="$(rg -n 'MSAN-EXPECTED-LINE' "$work/uninit.cpp" | cut -d: -f1)"
    rc=0
    MSAN_OPTIONS=halt_on_error=1 "$work/uninit" > "$log_uninit" 2>&1 || rc=$?
    if [[ $rc -eq 0 ]] \
        || ! rg -q 'WARNING: MemorySanitizer: use-of-uninitialized-value' "$log_uninit" \
        || ! rg -q "#0 .* in main .*uninit\.cpp:${line}" "$log_uninit"; then
        cat "$log_uninit" >&2
        die "The uninit program does not give an MSan report at uninit.cpp:${line}."
    fi
    # track-origins=2 must show where the memory came from (operator new[]).
    rg -q 'Uninitialized value was created by a heap allocation' "$log_uninit" \
        || { cat "$log_uninit" >&2; die "The MSan report has no origin of the heap allocation."; }
    log "PASS: the uninit program gives an MSan report at uninit.cpp:${line} with the heap origin."
}

main() {
    parse_args "$@"
    CC="${CC:-clang}"
    CXX="${CXX:-clang++}"
    command -v "$CC" > /dev/null || die "The C compiler '$CC' is not in PATH. Install clang."
    command -v "$CXX" > /dev/null || die "The C++ compiler '$CXX' is not in PATH. Install clang."
    LLVM_TAG="${LLVM_TAG:-$(llvm_tag_of_clang)}"
    if [[ $VERIFY_ONLY -eq 0 ]]; then
        command -v cmake > /dev/null || die "cmake is not in PATH."
        command -v ninja > /dev/null || die "ninja is not in PATH."
        command -v git > /dev/null || die "git is not in PATH."
        local t0=$SECONDS
        fetch_source
        build_and_install
        log "The fetch, the build and the install took $((SECONDS - t0)) s."
    fi
    verify
}

main "$@"
