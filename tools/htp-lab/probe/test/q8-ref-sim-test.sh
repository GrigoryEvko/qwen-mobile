#!/usr/bin/env bash
# The check of the harness claim "candidate kernel 0 agrees with the CPU oracle bit for bit": the
# ggml scalar reference (q8_0_ref.c) runs on x86-64 and on the Hexagon scalar unit of hexagon-sim
# (v73 and v79), with the flags of the probe (-ffp-contract=off, no fast math), on problems of
# mv_case.py. The results must have the same bytes.
#
#   tools/htp-lab/probe/test/q8-ref-sim-test.sh
#
# The phone runs the same code: the host program on the ARM CPU and kernel 0 in the DSP library.
# This test shows that the two IEEE scalar units give the same bits before the phone run. All
# files stay in /tmp of the container.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../../scripts/lib.sh"

cd "$REPO_ROOT"
container_run "$SNAPDRAGON_IMAGE" bash -euo pipefail -c '
probe=tools/htp-lab/probe
tools=$HEXAGON_TOOLS_ROOT/Tools/bin
work=$(mktemp -d)
mkdir -p "$work/shim"
ln -sf /usr/lib/x86_64-linux-gnu/libncurses.so.6 "$work/shim/libncurses.so.5"
ln -sf /usr/lib/x86_64-linux-gnu/libtinfo.so.6 "$work/shim/libtinfo.so.5"

flags="-O2 -ffp-contract=off -fno-fast-math -I$probe"
# shellcheck disable=SC2086
clang -std=c11 $flags "$probe/test/q8_ref_main.c" "$probe/q8_0_ref.c" -lm -o "$work/ref-x86"
for v in v73 v79; do
    # shellcheck disable=SC2086
    "$tools/hexagon-clang" -m$v -G0 $flags "$probe/test/q8_ref_main.c" "$probe/q8_0_ref.c" -lm -o "$work/ref-$v"
done

fails=0
cases="64:4096:1:0.01 13:96:2:0.01 256:2048:3:0.1 7:9728:4:0.0"
for c in $cases; do
    IFS=: read -r rows cols seed outl <<< "$c"
    p="$work/p-$rows-$cols.bin"
    python3 "$probe/mv_case.py" make "$p" --rows "$rows" --cols "$cols" --seed "$seed" --outliers "$outl" > /dev/null
    "$work/ref-x86" "$p" "$work/y-x86.bin"
    for v in v73 v79; do
        (cd "$work" && LD_LIBRARY_PATH=$work/shim "$tools/hexagon-sim" --m$v --quiet "$work/ref-$v" -- "$p" "$work/y-$v.bin" > /dev/null 2>&1) \
            || { echo "FAIL $rows x $cols: hexagon-sim $v did not run"; fails=$((fails + 1)); continue; }
        if cmp -s "$work/y-x86.bin" "$work/y-$v.bin"; then
            echo "PASS $rows x $cols seed $seed: x86-64 and hexagon $v give the same $rows f32 values"
        else
            echo "FAIL $rows x $cols seed $seed: x86-64 and hexagon $v differ"
            fails=$((fails + 1))
        fi
    done
done
rm -rf "$work"
echo "q8-ref-sim-test: $fails failures"
exit $((fails > 0))
'
