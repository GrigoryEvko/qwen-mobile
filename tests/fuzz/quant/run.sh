#!/usr/bin/env bash
# The fuzz runner of the quant area (quant/ and the GGUF files that it writes).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HERE="$ROOT/tests/fuzz/quant"
SANITIZERS=(none asan ubsan tsan msan)
PROFILES=(debug release)
FUZZ_MSAN_PREFIX="${FUZZ_MSAN_PREFIX:-$ROOT/build/fuzz/msan-libcxx/install}"
# The llama.cpp tree: QFZ_LLAMA_DIR, or a private copy of the patched tree of HEAD that
# tests/sanitizers/llama-copy.sh makes or refreshes one time for each run of this script (refresh_llama).
# The copy comes from the git objects of HEAD, thus an uncommitted edit in the submodule, or a landing
# during a run, does not go into the build, and the rule LLAMA-COPY of check-rules.sh checks its stamp.
LLAMA_DIR="${QFZ_LLAMA_DIR:-$ROOT/build/fuzz/quant/llama-src}"
LLAMA_COPY=1
[[ -n "${QFZ_LLAMA_DIR:-}" ]] && LLAMA_COPY=0
export CUDA_VISIBLE_DEVICES=""
# Two threads for torch and numpy in each target, thus --jobs N does not ask for N x 28 threads.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

usage() {
    cat <<'EOF'
Usage:
  tests/fuzz/quant/run.sh <test|fuzz> <none|asan|ubsan|tsan|msan> [--profile debug|release]
                          [--budget-seconds N] [--jobs N]
  tests/fuzz/quant/run.sh build <none|asan|ubsan|tsan|msan> [--profile debug|release]
  tests/fuzz/quant/run.sh build native
  tests/fuzz/quant/run.sh phone-files
  tests/fuzz/quant/run.sh phone-commands
  tests/fuzz/quant/run.sh cpu [--budget-seconds N] [--jobs N]     (the alias of: fuzz none)
  tests/fuzz/quant/run.sh regress                                 (the alias of: test none)

Modes:
  test    Run each target one time over its seeds and its regression cases:
          the Hypothesis database and the explicit examples, no new generation,
          the atheris targets over their seed files, the native targets over
          the seed files and the phone set. tests/fuzz/quant/seeds holds the
          seed files, tests/fuzz/quant/regress holds each failing input and the
          regression tests (regress/test_qfz_regressions.py). The exit status is
          not zero when a target has a finding. A known defect (a strict xfail
          of the regression tests) is a finding (rule R8).
  fuzz    Generate new inputs for each target for the budget (the preset
          value is 600 s for each target).
  build   Build the native code of one sanitizer and one profile: llama-perplexity
          and the GGUF loader check qfz-gguf-check, in
          build/fuzz/quant-<profile>-<sanitizer>. test and fuzz build it first
          (an incremental build). "build native" builds the native reference of the phone set:
          llama-perplexity with the llama.cpp preset flags (Release, GGML_NATIVE=ON),
          no fast math and no sanitizer, in build/fuzz/quant/native-host.
  phone-files     Write the phone set into build/fuzz/quant/phone: the fuzzed GGUF
                  files, their KL bases of the x86 oracle, and the host results of
                  the native reference and of the two profiles without a sanitizer.
                  The target llama-toy runs the same files in each sanitizer build.
  phone-commands  Print the adb commands that run the phone set on HTP0 and on the CPU.

Profiles (FUZZ_PROFILE, or --profile; the preset value runs the two, debug first):
  release The shipped flags of android/snapdragon/CMakeUserPresets.json: Release,
          -O3 -DNDEBUG -flto -fvectorize -ffp-model=fast -fno-finite-math-only
          -D_GNU_SOURCE, GGML_OPENMP=OFF, GGML_LLAMAFILE=OFF, and -g. The x86
          substitution: GGML_NATIVE=ON in place of -march=armv8.7a+fp16+dotprod+i8mm.
          LTO links with lld, and the archives use llvm-ar.
  debug   -O1 -g -fno-omit-frame-pointer, no NDEBUG, no LTO, the same fp flags,
          GGML_OPENMP=OFF, GGML_LLAMAFILE=OFF, GGML_NATIVE=ON.
  The pure Python targets have no native code, thus they run in the release
  pass only. The regression tests run in the two passes (rule R13).

Sanitizers (one sanitizer for each build and each run, never two):
  none    All the Python fuzzers (Hypothesis and atheris) and the native targets
          without a sanitizer. The Python fuzzers apply only to "none": quant/
          holds no native code, thus a sanitizer has nothing of ours to watch in
          the Python process.
  asan    -fsanitize=address. The native targets only.
  ubsan   -fsanitize=undefined -fno-sanitize-recover=undefined. The native targets
          only. The first report stops the run.
  tsan    -fsanitize=thread. The native targets only.
  msan    -fsanitize=memory with the MSan libc++ at FUZZ_MSAN_PREFIX
          (preset: build/fuzz/msan-libcxx/install). The native targets only.
  Suppressions: only the shared files tests/sanitizers/<sanitizer>.supp, when
  they exist. This area keeps no suppression of its own.

The native targets:
  loader-check  The ggml GGUF loader (qfz-gguf-check) on the seed files and the
                phone set (test), and on each file that the export and reader
                fuzzers write or corrupt (fuzz).
  llama-toy     llama-perplexity on toy Qwen3.5 models that export() writes, against
                the KL base of the x86 oracle (build/oracle-x86, read only).

Options:
  --profile P          debug or release (preset: the two)
  --budget-seconds N   The time of each target in the mode fuzz (preset 600)
  --jobs N             The number of targets that run at the same time (preset 1)
  --targets LIST       A comma list of target names (preset: each target that applies)

Results: one JSON line per target in build/fuzz/quant-<profile>-<sanitizer>/results.jsonl,
and a copy in build/fuzz/quant-<sanitizer>/results.jsonl:
  {area, target, sanitizer, profile, mode, seconds, executions, findings, crash_files}
The llama.cpp tree: build/fuzz/quant/llama-src, a copy of the patched tree of HEAD that
tests/sanitizers/llama-copy.sh makes or refreshes at the start of each run. The native builds
compile it, and the Python targets import its gguf-py (PYTHONPATH, QFZ_LLAMA_DIR).
Environment: QFZ_LLAMA_DIR (a llama.cpp tree of your own in place of the copy), QFZ_KNOWN (fuzz
the inputs of the known defects too), QFZ_FIXED (expect the correct behavior of a known defect),
FUZZ_SANITIZER, FUZZ_PROFILE, FUZZ_MSAN_PREFIX. The
names of the known defects are the keys of KNOWN_DEFECTS in qfz_common.py. The GPU stays
hidden.
EOF
}

die() {
    echo "run.sh: $*" >&2
    exit 2
}

member() {
    local x="$1"
    shift
    local s
    for s in "$@"; do
        [[ "$x" == "$s" ]] && return 0
    done
    return 1
}

# Give the compile flags of one sanitizer for C, C++ and the linker.
sanitizer_flags() {
    case "$1" in
        none)  echo "" ;;
        asan)  echo "-fsanitize=address -fno-omit-frame-pointer" ;;
        ubsan) echo "-fsanitize=undefined -fno-sanitize-recover=undefined" ;;
        tsan)  echo "-fsanitize=thread" ;;
        msan)  echo "-fsanitize=memory -fsanitize-memory-track-origins=2 -fno-omit-frame-pointer" ;;
    esac
}

FP_FLAGS="-fvectorize -ffp-model=fast -fno-finite-math-only -D_GNU_SOURCE"

# Make or refresh the copy of llama.cpp (when QFZ_LLAMA_DIR is not set). Then give the Python targets its
# gguf-py: qfz_common.LLAMA_DIR reads QFZ_LLAMA_DIR, and PYTHONPATH puts the gguf-py of the tree before the
# editable install of the submodule. The lock keeps two runs of this script from writing the copy together.
refresh_llama() {
    if [[ $LLAMA_COPY == 1 ]]; then
        mkdir -p "$(dirname "$LLAMA_DIR")"
        flock "$LLAMA_DIR.lock" "$ROOT/tests/sanitizers/llama-copy.sh" "$LLAMA_DIR" > /dev/null \
            || die "tests/sanitizers/llama-copy.sh could not make the copy $LLAMA_DIR"
    fi
    [[ -f "$LLAMA_DIR/gguf-py/gguf/__init__.py" ]] || die "$LLAMA_DIR is not a llama.cpp tree: it has no gguf-py"
    export QFZ_LLAMA_DIR="$LLAMA_DIR" PYTHONPATH="$LLAMA_DIR/gguf-py${PYTHONPATH:+:$PYTHONPATH}"
}

# Remove the CMake build directory $1 when another llama.cpp tree configured it: CMake refuses a second
# source tree for one build directory.
drop_other_tree() {
    local cache="$1/CMakeCache.txt"
    if [[ -f "$cache" ]] && ! rg -q -x -F "CMAKE_HOME_DIRECTORY:INTERNAL=$LLAMA_DIR" "$cache"; then
        rm -rf "$1"
    fi
}

# Build llama-perplexity and qfz-gguf-check with one sanitizer and one profile. The build is incremental,
# thus a refreshed copy of llama.cpp gives new objects only for its changed files.
build() {
    local san="$1" profile="$2"
    local out="$ROOT/build/fuzz/quant-$profile-$san"
    local flags cxx_extra="" link_extra="" build_type opt_var opt_flags lto="" lto_link="" tools=()
    flags="$(sanitizer_flags "$san")"
    if [[ "$san" == "msan" ]]; then
        [[ -d "$FUZZ_MSAN_PREFIX/lib" ]] || { echo "run.sh: the MSan libc++ is missing at $FUZZ_MSAN_PREFIX" >&2; return 3; }
        cxx_extra="-stdlib=libc++ -nostdinc++ -isystem $FUZZ_MSAN_PREFIX/include/c++/v1"
        link_extra="-stdlib=libc++ -L$FUZZ_MSAN_PREFIX/lib -Wl,-rpath,$FUZZ_MSAN_PREFIX/lib"
    fi
    if [[ "$profile" == "release" ]]; then
        build_type=Release; opt_var=RELEASE; opt_flags="-O3 -DNDEBUG"
        lto="-flto -g"; lto_link="-flto -fuse-ld=lld"
        tools=(-DCMAKE_AR=/usr/bin/llvm-ar -DCMAKE_RANLIB=/usr/bin/llvm-ranlib)
    else
        build_type=Debug; opt_var=DEBUG; opt_flags="-O1 -g -fno-omit-frame-pointer"
    fi
    # The CMake build directory is $out/build, the name that the rule LLAMA-COPY of check-rules.sh reads.
    local cm="$out/build"
    mkdir -p "$out/bin" "$out/logs"
    drop_other_tree "$cm"
    echo "run.sh: build $profile $san from $LLAMA_DIR into $out (logs in $out/logs)"
    cmake -S "$LLAMA_DIR" -B "$cm" -G Ninja \
        -DCMAKE_BUILD_TYPE="$build_type" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ "${tools[@]}" \
        -DCMAKE_C_FLAGS="$FP_FLAGS $lto $flags" -DCMAKE_CXX_FLAGS="$FP_FLAGS $lto $flags $cxx_extra" \
        -DCMAKE_C_FLAGS_"$opt_var"="$opt_flags" -DCMAKE_CXX_FLAGS_"$opt_var"="$opt_flags" \
        -DCMAKE_EXE_LINKER_FLAGS="$lto_link $flags $link_extra" \
        -DGGML_NATIVE=ON -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
        -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF > "$out/logs/cmake.log" 2>&1 \
        || { echo "run.sh: the configure of $cm failed. Read $out/logs/cmake.log." >&2; return 1; }
    nice -n 10 cmake --build "$cm" --target llama-perplexity ggml-base -j "${BUILD_JOBS:-10}" \
        > "$out/logs/build.log" 2>&1 \
        || { echo "run.sh: the build of $cm failed. Read $out/logs/build.log." >&2; return 1; }
    # shellcheck disable=SC2086
    clang -std=c11 $opt_flags $FP_FLAGS $lto $flags -I "$LLAMA_DIR/ggml/include" \
        -c "$HERE/qfz_gguf_check.c" -o "$out/bin/qfz_gguf_check.o" || return 1
    # shellcheck disable=SC2086
    clang++ $lto_link $flags $link_extra "$out/bin/qfz_gguf_check.o" "$cm/ggml/src/libggml-base.a" -lm -lpthread \
        -o "$out/bin/qfz-gguf-check" || return 1
    rm -f "$out/bin/qfz_gguf_check.o"
    echo "run.sh: built $cm/bin/llama-perplexity and $out/bin/qfz-gguf-check"
}

# Build the native reference of the phone set: llama-perplexity with the llama.cpp preset flags
# (Release, GGML_NATIVE=ON), no fast math and no sanitizer, in build/fuzz/quant/native-host.
build_native() {
    local out="$ROOT/build/fuzz/quant/native-host"
    mkdir -p "$out/logs"
    drop_other_tree "$out/llama"
    echo "run.sh: build the native host reference from $LLAMA_DIR into $out (logs in $out/logs)"
    cmake -S "$LLAMA_DIR" -B "$out/llama" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DGGML_NATIVE=ON -DLLAMA_CURL=OFF \
        -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF \
        > "$out/logs/cmake.log" 2>&1 \
        || { echo "run.sh: the configure of $out/llama failed. Read $out/logs/cmake.log." >&2; return 1; }
    nice -n 10 cmake --build "$out/llama" --target llama-perplexity -j "${BUILD_JOBS:-10}" > "$out/logs/build.log" 2>&1 \
        || { echo "run.sh: the build of $out/llama failed. Read $out/logs/build.log." >&2; return 1; }
    echo "run.sh: built $out/llama/bin/llama-perplexity"
}

cd "$ROOT"
[[ $# -ge 1 ]] || { usage; exit 2; }
mode="$1"
shift
san=""
profiles=()
rest=()
case "$mode" in
    cpu) mode="fuzz"; san="none" ;;
    regress) mode="test"; san="none" ;;
esac
if [[ -z "$san" && $# -ge 1 && "$1" != --* ]]; then
    san="$1"
    shift
fi
san="${san:-${FUZZ_SANITIZER:-}}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile) [[ $# -ge 2 ]] || die "--profile needs debug or release"; profiles+=("$2"); shift 2 ;;
        *) rest+=("$1"); shift ;;
    esac
done
if [[ ${#profiles[@]} -eq 0 ]]; then
    if [[ -n "${FUZZ_PROFILE:-}" ]]; then profiles=("$FUZZ_PROFILE"); else profiles=("${PROFILES[@]}"); fi
fi
for p in "${profiles[@]}"; do member "$p" "${PROFILES[@]}" || die "the profile $p is not one of: ${PROFILES[*]}"; done

case "$mode" in
    build|test|fuzz|phone-files|phone-commands) refresh_llama ;;
esac

case "$mode" in
    -h|--help|help)
        usage
        ;;
    build)
        if [[ "$san" == "native" ]]; then
            build_native
            exit 0
        fi
        member "$san" "${SANITIZERS[@]}" || die "build needs one sanitizer of: ${SANITIZERS[*]}, or native"
        for p in "${profiles[@]}"; do build "$san" "$p"; done
        ;;
    test|fuzz)
        member "$san" "${SANITIZERS[@]}" || die "$mode needs one sanitizer of: ${SANITIZERS[*]}"
        status=0
        for p in "${profiles[@]}"; do
            if ! build "$san" "$p"; then
                echo "run.sh: the build of $p $san is not possible, the native targets report it as skipped" >&2
            fi
            code=0
            FUZZ_SANITIZER="$san" FUZZ_PROFILE="$p" uv run python "$HERE/qfz_run.py" "$mode" "$san" --profile "$p" \
                ${rest[@]+"${rest[@]}"} || code=$?
            # A finding (1) wins over a skipped target (3) in the exit status.
            if [[ $code -eq 1 || ( $code -ne 0 && $status -eq 0 ) ]]; then status=$code; fi
        done
        exit "$status"
        ;;
    phone-files)
        for p in "${PROFILES[@]}"; do build none "$p" || die "the build of $p none failed"; done
        build_native || die "the build of the native host reference failed"
        exec uv run python "$HERE/qfz_phone.py" files
        ;;
    phone-commands)
        exec uv run python "$HERE/qfz_phone.py" commands
        ;;
    *)
        usage
        exit 2
        ;;
esac
