#!/usr/bin/env bash
# Write the table of the llama.cpp tests: one row for each test, one column
# for each profile and configuration.
#
# Usage:
#   tests/suite/llama-table.sh [--out FILE]
#
# The script reads build/fuzz/matrix-llama-<profile>-<config>/results.jsonl
# (tests/suite/llama-ctest.sh writes them). A cell is:
#   pass     P
#   fail     F<n>, where n is the number of sanitizer reports (F0: the test
#            failed with no report)
#   timeout  T
#   excluded X (the reason is in the records and in tests/suite/llama-exclude.tsv)
#   missing  M (a missing prerequisite)
#   no data  .
# The last rows give the pass count, the fail count and the wall time of the
# ctest run of each column.
#
# Requirements: jq. Exit status: 0.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

OUT=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out) OUT="$2"; shift 2 ;;
        *) suite_die "The option '$1' is not known." ;;
    esac
done

# Write the table to the standard output.
write_table() {
    local cols=() p c f
    for p in $SUITE_PROFILES; do
        for c in $SUITE_CONFIGS; do
            cols+=("$p-$c")
        done
    done
    local files=()
    for f in "${cols[@]}"; do
        files+=("$SUITE_REPO_ROOT/build/fuzz/matrix-llama-$f/results.jsonl")
    done
    # One JSON object: {columns, rows: {test: {column: cell}}, totals}.
    local inputs=() i=0
    for f in "${files[@]}"; do
        if [[ -f "$f" ]]; then
            inputs+=(--slurpfile "c$i" "$f")
        else
            inputs+=(--argjson "c$i" '[]')
        fi
        i=$((i + 1))
    done
    jq -rn "${inputs[@]}" --argjson ncol "${#cols[@]}" --args '
        def cell: if .status == "pass" then "P"
                  elif .status == "fail" then "F\(.findings // 0)"
                  elif .status == "timeout" then "T"
                  elif .status == "excluded" then "X"
                  elif .status == "missing-prerequisite" then "M"
                  else "?" end;
        [range(0; $ncol) | . as $k | $ARGS.named["c\($k)"]] as $data
        | ($data | map(map(.target)) | add // [] | unique) as $tests
        | ($ARGS.positional) as $cols
        | (["test"] + $cols | @tsv),
          ($tests[] as $t
             | [$t] + [$data[] | (map(select(.target == $t)) | if length == 0 then "." else (.[0] | cell) end)]
             | @tsv),
          (["pass"] + [$data[] | map(select(.status == "pass")) | length | tostring] | @tsv),
          (["fail+timeout"] + [$data[] | map(select(.status == "fail" or .status == "timeout")) | length | tostring] | @tsv),
          (["seconds (sum)"] + [$data[] | map(.seconds) | add // 0 | floor | tostring] | @tsv)
    ' "${cols[@]}"
}

if [[ -n "$OUT" ]]; then
    write_table > "$OUT"
    suite_log "Wrote $OUT."
else
    write_table
fi
