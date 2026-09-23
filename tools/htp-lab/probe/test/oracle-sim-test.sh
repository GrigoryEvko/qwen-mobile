#!/usr/bin/env bash
# The check of the census oracle (host/oracle.c) against the simulator census, on the build machine.
#
#   tools/htp-lab/probe/test/oracle-sim-test.sh [SIM_TAG]
#
# SIM_TAG is the tag of the simulator run (tools/htp-lab/isa/run_census.sh TAG). Default: c.
#
# The script compiles host/oracle.c with test/oracle_sim.c for x86-64 in the Snapdragon container
# (with -ffp-contract=off, no fast math, and the sanitizers), and computes the oracle of each op on
# the corpus of the simulator run. Then oracle_compare.py compares the output of each simulated
# version with the oracle. The oracle is plain IEEE C, thus x86-64 and arm64 give the same bits
# (exp2f of the C library excepted).
#
# The check: for each op of EXPECT_EXACT below, the v79 simulator result must agree with the
# oracle in each finite lane. A wrong lane layout in the oracle table gives large errors in most
# lanes, thus these ops test the layouts of each kind.
#
# Output (build/isaprobe/oracle-sim/, not in git): compare-<version>.txt and compare-<version>.csv,
# the table of oracle_compare.py for each simulated version. The oracle files (75 MB) go away at
# the end. KEEP=1 keeps them. The comparison needs numpy on the build machine.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../../scripts/lib.sh"

readonly SIM_TAG=${1:-c}
readonly OUT_REL=build/isaprobe/oracle-sim
readonly SIM_REL=tools/htp-lab/out-isa

# The ops that the v79 simulator computes like the oracle (IEEE round to nearest even), one of
# each lane layout. The v79 simulator implements the IEEE-form opcodes by the ISA, thus these are
# a test of the oracle and its layouts, not of the chip.
readonly EXPECT_EXACT=(
    "ieee.Q6_Vsf_vadd_VsfVsf" "ieee.Q6_Vhf_vmpy_VhfVhf" "ieee.Q6_Vsf_vsub_VsfVsf"
    "ieee.Q6_Wsf_vmpy_VhfVhf" "ieee.Q6_Wsf_vadd_VbfVbf" "ieee.Q6_Wsf_vmpyacc_WsfVhfVhf"
    "ieee.Q6_Wsf_vcvt_Vhf" "ieee.Q6_Vhf_vcvt_VsfVsf" "ieee.Q6_Vbf_vcvt_VsfVsf"
    "ieee.Q6_Whf_vcvt_Vb" "ieee.Q6_Whf_vcvt_Vub" "ieee.Q6_Vhf_vcvt_Vh"
    "ieee.Q6_Vsf_vdmpy_VhfVhf" "seq.hvx_vec_f16_to_f32" "seq.hvx_vec_f32_to_f16" "Q6_Vsf_equals_Vw"
)
# Not in the list, because the simulator differs from the naive IEEE oracle by design:
#   ieee.Q6_Vsf_vdmpyacc_VsfVhfVhf rounds a0 b0 + a1 b1 + c one time, the oracle two times, thus a
#     cancellation gives a large difference in 11 % of the lanes.
#   Q6_Q_vcmp_gt_* gives true for x > NaN when the NaN has the sign bit, IEEE gives false.
#   seq.hvx_vec_f32_to_f16 gives +0 for a negative value that rounds to zero (zero+- column).

cd "$REPO_ROOT"
[[ -f $SIM_REL/isa-$SIM_TAG/corpus_sf_a.bin ]] || die "$SIM_REL/isa-$SIM_TAG has no corpus. Run tools/htp-lab/isa/run_census.sh $SIM_TAG first."
rm -rf "$OUT_REL"
mkdir -p "$OUT_REL"

container_run -e SIM_TAG="$SIM_TAG" -e OUT_REL="$OUT_REL" -e SIM_REL="$SIM_REL" "$SNAPDRAGON_IMAGE" bash -euo pipefail -c '
probe=tools/htp-lab/probe
isa=tools/htp-lab/isa
work=$(mktemp -d)
python3 - "$isa/isa_ops.csv" "$isa/gen_census.py" > "$work/table.txt" << "PY"
import csv, re, sys
text = open(sys.argv[2]).read()
streams = re.search(r"STREAMS = \[(.*?)\]", text, re.S).group(1)
streams = [s.strip().strip("\"") for s in streams.replace("\n", " ").split(",") if s.strip()]
etypes = re.search(r"ETYPES = \[(.*?)\]", text, re.S).group(1)
etypes = [s.strip().strip("\"") for s in etypes.split(",")]
for k, s in enumerate(streams):
    print(f"stream {k} {s}")
for r in csv.DictReader(open(sys.argv[1], newline="")):
    out_t = etypes.index("pred") if r["out_ctype"] == "Q" else etypes.index(r["out_type"])
    tys = [etypes.index(t) for t in r["in_types"].split("/")]
    sts = [streams.index(s) for s in r["in_streams"].split("/")]
    bys = [int(b) for b in r["in_bytes"].split("/")]
    print("op", r["id"], r["name"], out_t, r["out_bytes"], r["n_vectors"], *tys, *bys, *sts)
PY
clang -std=c11 -D_GNU_SOURCE -O2 -g -Wall -Wextra -Werror -ffp-contract=off -fno-fast-math \
    -fsanitize=address,undefined -fno-sanitize-recover=all -I"$isa" -I"$probe/host" \
    "$probe/test/oracle_sim.c" "$probe/host/oracle.c" "$probe/host/census_io.c" -lm -o "$work/oracle_sim"
"$work/oracle_sim" "$work/table.txt" "$SIM_REL/isa-$SIM_TAG" "$OUT_REL"
rm -rf "$work"
'

fails=0
for dir in "$SIM_REL"/isa-"$SIM_TAG" "$SIM_REL"/isa-v7[35]-"$SIM_TAG" "$SIM_REL"/isa-v81-"$SIM_TAG"; do
    [[ -d $dir ]] || continue
    case $dir in */isa-"$SIM_TAG") v=v79 ;; *) v=${dir##*/isa-}; v=${v%-"$SIM_TAG"} ;; esac
    python3 tools/htp-lab/probe/oracle_compare.py census "$dir" --oracle-dir "$OUT_REL" \
        --csv "$OUT_REL/compare-$v.csv" > "$OUT_REL/compare-$v.txt"
    echo "oracle-sim-test: $v: $(grep -c . "$OUT_REL/compare-$v.csv") rows in $OUT_REL/compare-$v.csv"
done

# The check of EXPECT_EXACT on v79: exact 1.0, or the only differences in nonfinite lanes.
python3 - "$OUT_REL/compare-v79.csv" "${EXPECT_EXACT[@]}" << 'PY' || fails=1
import csv, sys
rows = {r["name"]: r for r in csv.DictReader(open(sys.argv[1]))}
bad = 0
for name in sys.argv[2:]:
    r = rows.get(name)
    if r is None or r["status"] != "ok":
        print(f"FAIL {name}: {'no result' if r is None else r['status']}")
        bad += 1
    elif float(r["max_err"]) != 0.0:
        print(f"FAIL {name}: exact {r['exact']} max {r['max_err']} {r['unit']} at lane {r['worst_lane']}")
        bad += 1
    else:
        print(f"PASS {name}: exact {r['exact']}, nonfinite {r['nonfinite']}")
sys.exit(1 if bad else 0)
PY
if [[ ${KEEP:-0} != 1 ]]; then
    rm -f "$OUT_REL"/out_*.oracle.bin
fi
echo "oracle-sim-test: $([[ $fails == 0 ]] && echo "all checks passed" || echo "a check failed")"
exit $fails
