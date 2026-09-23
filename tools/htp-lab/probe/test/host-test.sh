#!/usr/bin/env bash
# The test of the host program isaprobe on the build machine, without a phone.
#
#   tools/htp-lab/probe/test/host-test.sh
#
# The script compiles host/isaprobe.c for x86-64 with the mock of FastRPC and of the DSP library
# (test/mock_rpc.c), with the address and undefined-behavior sanitizers, in the Snapdragon
# container. census_fixture.py writes a mock op table with ISA_N_OPS ops and a corpus with the
# header and the hash of isa_kernels.h. Then the script runs the program in each case below, and
# compares the exit code and the output files. The mock stops the program with the exit code 99
# when a DSP handle stays open. All files stay in /tmp of the container, and the container is
# removed at the end.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../../scripts/lib.sh"

cd "$REPO_ROOT"
[[ -f tools/htp-lab/isa/isa_kernels.h ]] || die "tools/htp-lab/isa/isa_kernels.h does not exist: the census mode needs it"
container_run "$SNAPDRAGON_IMAGE" bash -euo pipefail -c '
probe=tools/htp-lab/probe
isa=tools/htp-lab/isa
work=$(mktemp -d)
gen=$work/gen
mkdir -p "$gen"
"$HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/bin/qaic" -mdll -o "$gen" -I"$HEXAGON_SDK_ROOT/incs" \
    -I"$HEXAGON_SDK_ROOT/incs/stddef" "$probe/isaprobe.idl"
flags="-std=c11 -D_GNU_SOURCE -O1 -g -Wall -Wextra -Werror -ffp-contract=off -fno-fast-math"
flags="$flags -fsanitize=address,undefined -fno-sanitize-recover=all"
inc="-I$HEXAGON_SDK_ROOT/incs -I$HEXAGON_SDK_ROOT/incs/stddef -I$gen -I$probe -I$probe/host -I$probe/dsp -I$isa"
# shellcheck disable=SC2086
clang $flags $inc -DISAPROBE_HAVE_ISA_KERNELS_H=1 "$probe/host/isaprobe.c" "$probe/host/census_io.c" \
    "$probe/host/oracle.c" "$probe/q8_0_ref.c" "$probe/test/mock_rpc.c" -lm -o "$work/isaprobe"
# The program without the census header has the info and kernel modes only. The mock still reads the header.
# shellcheck disable=SC2086
clang $flags $inc "$probe/host/isaprobe.c" "$probe/q8_0_ref.c" "$probe/test/mock_rpc.c" -lm -o "$work/isaprobe-noisa"

read -r n_ops n_streams < <(python3 -c "
import re, sys
text = open(sys.argv[1]).read()
print(re.search(r\"#define ISA_N_OPS\s+(\d+)\", text).group(1), len(re.findall(r\"\bISA_S_[A-Z0-9_]+ = \", text)))
" "$isa/isa_kernels.h")
fix=$work/fix
python3 "$probe/test/census_fixture.py" make "$fix" --ops "$n_ops" --streams "$n_streams"
export MOCK_TABLE=$fix/table.txt
echo "host-test: $n_ops mock ops, $n_streams stream ids"

fails=0
# expect <name> <exit code> <command...>: run the command and compare its exit code.
expect() {
    local name=$1 want=$2 got=0
    shift 2
    "$@" > "$work/log.txt" 2>&1 || got=$?
    if [[ $got == "$want" ]]; then
        echo "PASS $name (exit $got)"
    else
        echo "FAIL $name: exit $got, expected $want"
        sed "s/^/    /" "$work/log.txt"
        fails=$((fails + 1))
    fi
}
# assert <name> <command...>: the command must succeed.
assert() {
    local name=$1
    shift
    if "$@" > "$work/assert.txt" 2>&1; then
        echo "PASS $name"
    else
        echo "FAIL $name"
        sed "s/^/    /" "$work/assert.txt"
        fails=$((fails + 1))
    fi
}
# fresh <dir>: an empty output directory.
fresh() {
    rm -rf "$1"
    mkdir -p "$1"
}
# check <out_dir> <options>: compare the outputs with the mock rule.
check() {
    python3 "$probe/test/census_fixture.py" check "$fix" "$@"
}

bin=$work/isaprobe
corpus=$fix/corpus
n_unavailable=$(( n_ops / 10 ))

expect "info on a matching chip" 0 "$bin" info
fresh "$work/o1"
expect "census on a matching chip" 0 "$bin" census "$corpus" "$work/o1"
assert "outputs and headers agree with the mock" check "$work/o1" --arch 79
expect "the check sees a wrong header field" 1 check "$work/o1" --arch 80
assert "record has the oracle" grep -q "^oracle 7 cc.Q6_Vsf_vadd_VsfVsf add 0x" "$work/o1/isaprobe.txt"
assert "record counts one oracle" grep -qx "ops_oracle 1" "$work/o1/isaprobe.txt"
fresh "$work/o1n"
expect "census without the oracle" 0 "$bin" census "$corpus" "$work/o1n" --no-oracle
assert "no oracle file" check "$work/o1n" --arch 79 --no-oracle
assert "record says no negative test" grep -qx "negative_test 0" "$work/o1/isaprobe.txt"
assert "record counts the unavailable ops" grep -qx "ops_unavailable $n_unavailable" "$work/o1/isaprobe.txt"

fresh "$work/o2"
expect "only the ops with a name part" 0 "$bin" census "$corpus" "$work/o2" --only op1
assert "the name part selects its ops" check "$work/o2" --arch 79 --only op1

fresh "$work/o3"
expect "v81 library on a v79 chip stops" 3 env MOCK_SKEL_ARCH=81 "$bin" census "$corpus" "$work/o3"
assert "no output after the gate" test ! -e "$work/o3/out_mock.op0.bin" -a ! -e "$work/o3/isaprobe.txt"
expect "negative test runs" 0 env MOCK_SKEL_ARCH=81 "$bin" census "$corpus" "$work/o3" --allow-newer-skel
assert "negative test writes the library version" check "$work/o3" --arch 81
assert "record says negative test" grep -qx "negative_test 1" "$work/o3/isaprobe.txt"

expect "unknown chip stops" 3 env MOCK_CHIP_ARCH=0 "$bin" info
expect "DSP value replaces a failed host query" 0 env MOCK_CHIP_ARCH=0 MOCK_DSP_ARCH=0x79 "$bin" info
expect "older library runs" 0 env MOCK_SKEL_ARCH=75 "$bin" info
expect "no HMX still runs" 0 env MOCK_NO_HMX=1 "$bin" info
expect "fatal setup stops" 2 env MOCK_SETUP=2 "$bin" info
expect "non-fatal setup runs" 0 env MOCK_SETUP=5 "$bin" info

fresh "$work/o4"
expect "a dead DSP process loses one op" 4 env MOCK_CRASH_OP=5 "$bin" census "$corpus" "$work/o4"
assert "the other ops still ran" check "$work/o4" --arch 79 --skip 5
assert "record has the FastRPC error" grep -qx "op 5 mock.op5 rpc_error 0x00000068" "$work/o4/isaprobe.txt"
fresh "$work/o5"
expect "a nonzero op status" 4 env MOCK_STATUS_OP=7 "$bin" census "$corpus" "$work/o5"
assert "the op with the status has no output, and its oracle" check "$work/o5" --arch 79 --skip 7
assert "record has the status" grep -qx "op 7 cc.Q6_Vsf_vadd_VsfVsf status -1" "$work/o5/isaprobe.txt"

fresh "$work/o6"
expect "stub library has no census" 1 env MOCK_N_OPS=0 "$bin" census "$corpus" "$work/o6"
expect "library of a different census" 1 env MOCK_N_OPS=10 "$bin" census "$corpus" "$work/o6"

mkdir -p "$work/bad"
cp "$corpus"/* "$work/bad/"
python3 -c "
import sys
path = sys.argv[1]
data = bytearray(open(path, \"rb\").read())
data[200] ^= 1
open(path, \"wb\").write(data)
" "$work/bad/corpus_s3.bin"
expect "a damaged corpus stream" 1 "$bin" census "$work/bad" "$work/o6"
rm "$work/bad/corpus_s3.bin"
expect "a missing corpus stream" 1 "$bin" census "$work/bad" "$work/o6"
expect "a missing output directory" 1 "$bin" census "$corpus" "$work/no/such/dir"
expect "no arguments" 1 "$bin"
expect "an unknown option" 1 "$bin" info --verbose
expect "--only with no value" 1 "$bin" census "$corpus" "$work/o6" --only
expect "info without isa_kernels.h" 0 "$work/isaprobe-noisa" info
expect "census without isa_kernels.h" 1 "$work/isaprobe-noisa" census "$corpus" "$work/o6"

# The candidate-kernel harness
python3 "$probe/mv_case.py" make "$work/mv.bin" --rows 40 --cols 256 --seed 3 > /dev/null
expect "info lists the kernels" 0 "$bin" info
assert "the list has kernel 0" grep -q "kernel 0 ref.q8_0_generic (op 1)" "$work/log.txt"
fresh "$work/k1"
expect "kernel 0 by ID" 0 "$bin" kernel 0 "$work/mv.bin" "$work/k1" --iters 3
assert "all rows equal to the ggml reference" grep -q "40 of 40 rows equal to the ggml reference, max 0 ulp" "$work/log.txt"
assert "the two result files agree" python3 -c "
import sys
a = open(sys.argv[1], \"rb\").read(); b = open(sys.argv[2], \"rb\").read()
assert len(a) == len(b) == 128 + 40 * 4 and a[128:] == b[128:] and a[:4] == b[:4] == b\"IPMR\"
" "$work/k1/ref.q8_0_generic.bin" "$work/k1/ggml-ref.bin"
fresh "$work/k2"
expect "kernel by name, 3 ulp in row 0" 0 env MOCK_KERNEL_ULP=3 "$bin" kernel ref.q8_0_generic "$work/mv.bin" "$work/k2"
assert "the summary sees 3 ulp" grep -q "39 of 40 rows equal to the ggml reference, max 3 ulp (row 0" "$work/log.txt"
expect "kernel without the census header" 0 "$work/isaprobe-noisa" kernel 0 "$work/mv.bin" "$work/k2"
expect "a kernel failure" 4 env MOCK_KERNEL_STATUS=-2 "$bin" kernel 0 "$work/mv.bin" "$work/k2"
expect "an unknown kernel" 1 "$bin" kernel no.such.kernel "$work/mv.bin" "$work/k2"
head -c 1000 "$work/mv.bin" > "$work/short.bin"
expect "a short problem file" 1 "$bin" kernel 0 "$work/short.bin" "$work/k2"
expect "a missing result directory" 1 "$bin" kernel 0 "$work/mv.bin" "$work/no/such/dir"
expect "a bad iteration count" 1 "$bin" kernel 0 "$work/mv.bin" "$work/k2" --iters 0

rm -rf "$work"
echo "host-test: $fails failures"
exit $((fails > 0))
'
