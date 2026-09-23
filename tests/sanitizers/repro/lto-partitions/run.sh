#!/usr/bin/env bash
# The reproducer of the rule LTO-PART of tests/sanitizers/check-rules.sh:
# a full LTO link with more than one code generation partition can leave the
# dynamic initializer of a C++17 inline variable out of .init_array
# (common.h gives the program).
#
# Usage:
#   tests/sanitizers/repro/lto-partitions/run.sh [PARTITIONS...]
#
# The script links a.cpp, b.cpp and main.cpp with clang++ -O1 -flto
# -fuse-ld=lld -Wl,--lto-partitions=N for each N (preset: 1 2 4 5 6 7 8 10
# 12 16), runs the program, and writes one line for each N: "correct" or
# "WRONG", the exit status and the size of .init_array. A correct program
# writes "b: called" and "a: positive" and exits with 0.
# With clang and lld 22.1.8 on x86_64, 5, 7, 10 and 16 partitions give
# WRONG (SIGSEGV, exit 139), and 1, 2, 4, 6, 8 and 12 give correct.
#
# Exit status: 1 if the link with one partition (the shipped build) is not
# correct, 2 if a tool is missing, else 0.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for tool in clang++ ld.lld llvm-readelf rg; do
    command -v "$tool" > /dev/null || { echo "run.sh: $tool is necessary." >&2; exit 2; }
done
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
counts=("$@")
[[ ${#counts[@]} -gt 0 ]] || counts=(1 2 4 5 6 7 8 10 12 16)
reference=0
for p in "${counts[@]}"; do
    bin="$work/lto-part-$p"
    CCACHE_DISABLE=1 clang++ -std=c++17 -O1 -flto -fuse-ld=lld "-Wl,--lto-partitions=$p" \
        "$here/a.cpp" "$here/b.cpp" "$here/main.cpp" -o "$bin" || { echo "partitions=$p: the link failed"; continue; }
    out="$("$bin" 2>&1)"
    rc=$?
    size="$(llvm-readelf -S "$bin" | rg -o -r '$1' '\.init_array\s+INIT_ARRAY\s+\S+\s+\S+\s+(\S+)' || true)"
    verdict=WRONG
    [[ $rc -eq 0 && "$out" == *"b: called"* && "$out" == *"a: positive"* ]] && verdict=correct
    echo "partitions=$p $verdict exit=$rc init_array_size=${size:-none}"
    [[ $p -eq 1 && $verdict != correct ]] && reference=1
done
exit $reference
