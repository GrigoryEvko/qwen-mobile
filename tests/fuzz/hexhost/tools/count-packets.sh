#!/usr/bin/env bash
# Count the instructions and the packets of some functions of a DSP library, with the Hexagon
# objdump of the Snapdragon container. Each packet of the objdump output starts with "{", and a
# line can hold two instructions of a duplex, separated by ";".
#
#   tests/fuzz/hexhost/tools/count-packets.sh LIB FUNCTION...
#
# LIB is a path relative to the repository root. The output has one line for each function.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
source scripts/lib.sh
lib=$1
shift
container_run -e LIB="$lib" -e FUNCS="$*" "$SNAPDRAGON_IMAGE" bash -euo pipefail -c '
od=/opt/hexagon/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin/hexagon-llvm-objdump
for f in $FUNCS; do
    "$od" -d --no-show-raw-insn --disassemble-symbols="$f" "$LIB" | grep -E "^[[:space:]]+[0-9a-f]+:" > /tmp/dis.txt || true
    n_lines=$(wc -l < /tmp/dis.txt)
    n_duplex=$(grep -o ";" /tmp/dis.txt | wc -l)
    n_pkt=$(grep -c "{" /tmp/dis.txt || true)
    echo "$f instructions $((n_lines + n_duplex)) packets $n_pkt"
done
'
