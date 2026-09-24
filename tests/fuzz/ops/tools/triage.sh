#!/usr/bin/env bash
# Replay the numeric findings of one host build of the op fuzzer on that build, compare them with the
# oracle, and print one RUN line for each finding (verdict, special, err = the largest error,
# ref = the largest finite |ref|). run_classes.py of this directory sorts these lines into classes.
#
#   tests/fuzz/ops/tools/triage.sh PROFILE-CONFIG OUT_DIR
#
# For example: triage.sh release-none build/fuzz/ops/triage/release. The build directory is
# build/fuzz/ops-PROFILE-CONFIG, and the oracle is build/fuzz/ops/oracle/ops_oracle. The script
# writes OUT_DIR/compare.txt and prints its count of RUN lines.
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: tests/fuzz/ops/tools/triage.sh PROFILE-CONFIG OUT_DIR" >&2; exit 2; }
repo=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../.." && pwd)
bdir=$repo/build/fuzz/ops-$1
out=$2
oracle=$repo/build/fuzz/ops/oracle/ops_oracle
mkdir -p "$out"
args=()
for d in "$bdir"/findings/*/*/; do
    compgen -G "$d*.bin" > /dev/null || continue
    args+=(--cases "$(basename "$d"):$d")
done
"$oracle" gen --out "$out/findings.pack" --n 0 "${args[@]}" > "$out/gen.log" 2>&1
"$bdir/ops_replay" --pack "$out/findings.pack" --out "$out/results.bin" --progress "$out/progress.txt" \
    --backends CPU > "$out/replay.log" 2>&1
FUZZ_OPS_COMPARE_LIST=1 "$oracle" compare --pack "$out/findings.pack" --results "$out/results.bin" \
    > "$out/compare.txt" 2>&1
grep -c '^RUN' "$out/compare.txt"
