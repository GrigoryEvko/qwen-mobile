"""The phone set of the quant fuzz area: fuzzed GGUF files that the packer writes, and the commands to run them.

    uv run python tests/fuzz/quant/qfz_phone.py files      # build/fuzz/quant/phone: files, KL bases, host results
    uv run python tests/fuzz/quant/qfz_phone.py commands   # print the adb commands of the phone runs
    uv run python tests/fuzz/quant/qfz_phone.py seeds      # write the seed files of the atheris reader fuzzer
    uv run python tests/fuzz/quant/qfz_phone.py compare <pulled-dir>   # the table of the pulled phone logs

The set holds a tiny Qwen3.5 model (the geometry PHONE of qfz_toy: 3 GDN
layers and 1 attention layer, the head sizes of the 2B and 4B) with random
weights in four value profiles, exported by quant/export.py to Q8_0 and to
Q4_0, and one tied file with the MTP block and the production plan. The
naive x86 oracle build (build/oracle-x86, read only) writes the KL base of
each file on a text. The phone then runs each file with the HTP0 backend
and with the CPU backend against that base, thus each backend is compared
with the oracle on the same file.

On the host, ``files`` also runs each file against the same base in the
native host build and in the build without a sanitizer of each profile
(build/fuzz/quant-debug-none, build/fuzz/quant-release-none), and in the
ggml loader check. The target llama-toy of run.sh then runs the same files
in each sanitizer build. manifest.json holds the results, the plans and
the SHA-256 of each file.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from qfz_checks import (  # noqa: E402
    NoKLStatistics,
    ggml_loader_check,
    kl_statistics,
    read_tensors,
    run_perplexity,
)
from qfz_common import (  # noqa: E402
    FUZZ_OUT,
    HOST_BIN,
    LLAMA_DIR,
    ORACLE_BIN,
    PROFILES,
    REGRESS_DIR,
    SEED_DIR,
    llama_bin,
    scratch,
)
from qfz_toy import PHONE, SMALL, make_text, output_rot_for, write_source  # noqa: E402
from quant.export import export  # noqa: E402
from quant.plan import Plan  # noqa: E402

LOG = logging.getLogger("qfz.phone")
OUT = FUZZ_OUT / "phone"
SERIAL = "192.168.14.130:5555"
REMOTE = "/data/local/tmp/qwen/fuzz/quant"
REMOTE_BIN = "/data/local/tmp/qwen/kl/bin"
REMOTE_LIB = "/data/local/tmp/qwen/kl/lib"
REMOTE_ADSP = "/data/local/tmp/qwen/hmx/lib"
DEVICES = {"htp0": "-dev HTP0 -ngl 99", "cpu": "-dev none -ngl 0"}
PPL_ARGS = "-f text.txt -c 128 -b 128 --chunks 4"


@dataclasses.dataclass(frozen=True)
class PhoneFile:
    """One file of the phone set: the name, the value profile, the plan, the tie and the MTP block."""

    name: str
    profile: str
    kind: str
    tied: bool = False
    mtp: bool = False
    production: bool = False

    def plan(self, n_layers: int) -> Plan:
        """Give the export plan: one type for every class, or the production plan with Q8_0 for kv and the MTP block."""
        if self.production:
            return Plan(bulk="Q4_0", head="Q8_0" if self.tied else "Q4_0", embedding="Q8_0", kv_proj="Q8_0",
                        gdn_gate="Q4_0", n_layers=n_layers, mtp="Q8_0")
        k = self.kind
        return Plan(bulk=k, head=k, embedding=k, kv_proj=k, gdn_gate=k, n_layers=n_layers, mtp=k)


SET = [PhoneFile(f"{profile}-{kind.lower().replace('_', '')}", profile, kind)
       for profile in ("normal", "tiny", "large", "mixed") for kind in ("Q8_0", "Q4_0")]
SET.append(PhoneFile("tied-mtp-prod", "normal", "Q4_0", tied=True, mtp=True, production=True))


def _sha256(path: Path) -> str:
    """Give the SHA-256 of a file. Complexity is O(file size)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_files(seed: int = 20260923) -> dict:
    """Write the phone set, the text and the KL bases, run the host checks, and write manifest.json.

    Args:
        seed: The seed of the weights and of the text

    Returns:
        The manifest

    Raises:
        FileNotFoundError: If the oracle build is missing
        RuntimeError: If the oracle cannot run a file, or if a host run has no final KL statistics block (the
            manifest records each such run before the error)
    """
    if not (ORACLE_BIN / "llama-perplexity").exists():
        raise FileNotFoundError(f"{ORACLE_BIN}/llama-perplexity is missing: the oracle build is necessary")
    OUT.mkdir(parents=True, exist_ok=True)
    host_logs = OUT / "host-logs"
    host_logs.mkdir(exist_ok=True)
    text = OUT / "text.txt"
    text.write_text(make_text(1400, seed))
    manifest: dict = {"geometry": dataclasses.asdict(PHONE), "text": "text.txt", "files": {}}
    no_block: list[str] = []
    for i, spec in enumerate(SET):
        geo = dataclasses.replace(PHONE, mtp=spec.mtp)
        plan = spec.plan(geo.n_layer)
        model = OUT / f"{spec.name}.gguf"
        with scratch() as tmp:
            src = write_source(tmp / "src.gguf", geo, spec.profile, seed + i, tied=spec.tied)
            rot = None
            if spec.tied:
                rot = tmp / "rot.npy"
                np.save(rot, output_rot_for(geo, seed + i))
            export(src.path, model, tmp / "no-packs", plan, LLAMA_DIR, torch.device("cpu"), tie_head=spec.tied,
                   rot=rot)
        entry = {"profile": spec.profile, "plan": dataclasses.asdict(plan), "tied": spec.tied, "mtp": spec.mtp,
                 "bytes": model.stat().st_size, "sha256": _sha256(model), "host": {}}
        types = read_tensors(model)
        entry["types"] = sorted({kind for kind, _ in types.values()})
        loaded = ggml_loader_check(model)
        entry["host"]["ggml_loader_nonfinite"] = int(sum(int((~np.isfinite(v)).sum()) for v in loaded.values()))
        base = OUT / f"{spec.name}.kld"
        status, log = run_perplexity(ORACLE_BIN, model, text, base, write_base=True, threads=8)
        (host_logs / f"{spec.name}.oracle.log").write_text(log)
        if status != 0:
            raise RuntimeError(f"the oracle failed on {model} with {status}. Log: {host_logs}/{spec.name}.oracle.log")
        # The native host build, then the build without a sanitizer of each profile (rule R11). The target
        # llama-toy of run.sh runs the same files in each sanitizer build, one sanitizer per run (rule R1).
        runs = [("native", HOST_BIN, {})] + [(f"{prof}-none", llama_bin("none", prof), {})
                                             for prof in PROFILES]
        for label, bin_dir, env in runs:
            if not (bin_dir / "llama-perplexity").exists():
                entry["host"][label] = {"status": "missing build"}
                continue
            status, log = run_perplexity(bin_dir, model, text, base, write_base=False, threads=8, extra_env=env)
            (host_logs / f"{spec.name}.{label}.log").write_text(log)
            try:
                kld = kl_statistics(log)
            except NoKLStatistics:
                kld = {"error": "no final KL statistics block"}
                no_block.append(f"{spec.name}.{label}")
            entry["host"][label] = {"status": status, **kld,
                                    "sanitizer_reports": len(re.findall(r"(ERROR: AddressSanitizer|runtime error|"
                                                                        r"WARNING: ThreadSanitizer|"
                                                                        r"WARNING: MemorySanitizer)", log))}
        manifest["files"][f"{spec.name}.gguf"] = entry
        LOG.info("%s: %s", spec.name, json.dumps(entry["host"]))
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if no_block:
        raise RuntimeError(f"{len(no_block)} host runs have no final KL statistics block: {', '.join(no_block)}. "
                           f"Their logs are in {host_logs}. A build without patches/fuzz-quant/0002 can lose "
                           "the block at the exit: rebuild it.")
    return manifest


def phone_commands() -> str:
    """Give the adb commands of the phone runs: push, the runs with the thermal checks, pull.

    Each adb call carries a host timeout, and each run a device timeout, both 100 s or less.

    Returns:
        The commands as a bash text
    """
    names = [f"{spec.name}" for spec in SET]
    adb = f"timeout -s KILL 60 adb -s {SERIAL}"
    env = f"LD_LIBRARY_PATH={REMOTE_LIB} ADSP_LIBRARY_PATH={REMOTE_ADSP}"
    lines = [
        "# The quant fuzz set on the phone. Run from the repository root, on battery (not charging).",
        f"# Files: build/fuzz/quant/phone ({len(names)} models, their .kld bases, text.txt, manifest.json).",
        "",
        "# 1. Push the set.",
        f"{adb} shell 'mkdir -p {REMOTE}/logs'",
        f"timeout -s KILL 100 adb -s {SERIAL} push build/fuzz/quant/phone/text.txt "
        f"build/fuzz/quant/phone/manifest.json {REMOTE}/",
    ]
    for name in names:
        lines.append(f"timeout -s KILL 100 adb -s {SERIAL} push build/fuzz/quant/phone/{name}.gguf "
                     f"build/fuzz/quant/phone/{name}.kld {REMOTE}/")
    lines += [
        "",
        "# 2. The runs: before each run the charger state, the caps and the thermal status; the loop stops at a",
        "#    status other than 0 or on a charger. After each run the status and the llama processes. A run whose",
        "#    log has no final KL statistics block fails: llama-perplexity without patches/fuzz-quant/0002 can lose",
        "#    the block at the exit with the status 0. The run is not done again.",
        "failed=0",
        f"for name in {' '.join(names)}; do",
        "  for dev in htp0 cpu; do",
        f"    {adb} shell 'dumpsys battery | grep -E \"(AC|USB|Wireless) powered|status:|temperature\"' "
        "| tee /tmp/qfz-battery.txt",
        "    if rg -q 'powered: true' /tmp/qfz-battery.txt; then echo 'STOP: the phone charges'; break 2; fi",
        f"    {adb} shell 'cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq "
        "/sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq'",
        f"    status=$({adb} shell 'dumpsys thermalservice | grep \"Thermal Status\"')",
        "    echo \"before $name $dev: $status\"",
        "    case \"$status\" in *'Thermal Status: 0'*) ;; *) echo 'STOP: thermal status is not 0'; break 2;; esac",
        "    if [ $dev = htp0 ]; then flags='-dev HTP0 -ngl 99'; else flags='-dev none -ngl 0'; fi",
        f"    timeout -s KILL 100 adb -s {SERIAL} shell \"cd {REMOTE} && {env} timeout -s KILL 90 "
        f"{REMOTE_BIN}/llama-perplexity -m $name.gguf {PPL_ARGS} --kl-divergence-base $name.kld --kl-divergence "
        "$flags > logs/$name.$dev.log 2>&1; echo exit=\\$? >> logs/$name.$dev.log\"",
        f"    echo \"after $name $dev: $({adb} shell 'dumpsys thermalservice | grep \"Thermal Status\"')\"",
        f"    {adb} shell 'pgrep -a llama'",
        f"    found=$({adb} shell \"grep -c -e 'Mean    KLD' -e 'Same top p' {REMOTE}/logs/$name.$dev.log\")",
        "    if [ \"$found\" != \"2\" ]; then echo \"FAIL: logs/$name.$dev.log has no final KL statistics block\"; "
        "failed=$((failed + 1)); fi",
        "    sleep 15",
        "  done",
        "done",
        "rm -f /tmp/qfz-battery.txt",
        "echo \"runs without the final KL statistics block: $failed\"",
        "",
        "# 3. Pull the logs.",
        "mkdir -p build/fuzz/quant/phone/pulled",
        f"timeout -s KILL 100 adb -s {SERIAL} pull {REMOTE}/logs build/fuzz/quant/phone/pulled/",
        "",
        "# 4. The table of the results against the host and the oracle.",
        "uv run python tests/fuzz/quant/qfz_phone.py compare build/fuzz/quant/phone/pulled/logs",
    ]
    return "\n".join(lines) + "\n"


def compare(pulled: Path) -> tuple[str, int]:
    """Give the table of the phone KL results next to the host results of manifest.json.

    A log with no final KL statistics block is a failed run, as is a
    missing log or an exit status other than 0.

    Args:
        pulled: The directory of the pulled logs <name>.<device>.log

    Returns:
        The table as text, and the number of failed runs
    """
    manifest = json.loads((OUT / "manifest.json").read_text())
    rows = ["| file | unit | exit | mean KL | max KL | same top | host native mean KL |",
            "|---|---|---|---|---|---|---|"]
    failed = 0
    for fname, entry in manifest["files"].items():
        name = fname.removesuffix(".gguf")
        native = entry["host"].get("native", {}).get("mean", float("nan"))
        for dev in DEVICES:
            log = pulled / f"{name}.{dev}.log"
            if not log.exists():
                failed += 1
                rows.append(f"| {name} | {dev} | no log | | | | {native} |")
                continue
            text = log.read_text(errors="replace")
            code = re.search(r"exit=(\d+)", text)
            try:
                kld = kl_statistics(text)
            except NoKLStatistics:
                failed += 1
                rows.append(f"| {name} | {dev} | {code.group(1) if code else '?'} | no final KL statistics block | "
                            f"| | {native} |")
                continue
            failed += not code or code.group(1) != "0"
            rows.append(f"| {name} | {dev} | {code.group(1) if code else '?'} | {kld['mean']} | {kld['max']} | "
                        f"{kld['top1']} | {native} |")
    return "\n".join(rows) + "\n", failed


def write_seeds() -> list[Path]:
    """Write the committed seed files and the failing input files of the fuzzers.

    - seeds/reader: two small toy exports (the gguf-py reader, the atheris
      reader target, the loader check)
    - seeds/pack: raw inputs of the atheris packer target
    - regress/reader: the minimal files of three reader defects: a scalar
      array longer than the file, a tensor offset that wraps, and a block
      tensor with no dimension
    - regress/ggml: the minimal files of the five reports of the ggml loader
      in the metadata mode (the loader check)
    - regress/ggml-full: the minimal file of the abort of the ggml loader in
      the mode that reads the tensor data (the loader check)

    Returns:
        The written files
    """
    import struct

    from qfz_rawgguf import header, tensor_info

    reader, pack = SEED_DIR / "reader", SEED_DIR / "pack"
    bad_reader, ggml, full = REGRESS_DIR / "reader", REGRESS_DIR / "ggml", REGRESS_DIR / "ggml-full"
    for d in (reader, pack, bad_reader, ggml, full):
        d.mkdir(parents=True, exist_ok=True)
    written = []
    geo = dataclasses.replace(SMALL, n_embd=32, n_ff=64, n_layer=2, n_vocab=265)
    with scratch() as tmp:
        # The seed of the Q8_0 toy gives tensor bytes that hold no text which the label rule (NOID) of
        # tests/sanitizers/check-rules.sh reports.
        for kind, tie, seed in (("Q8_0", False, 4), ("Q4_0", True, 1)):
            src = write_source(tmp / "src.gguf", geo, "normal", seed, tied=tie)
            rot = None
            if tie:
                rot = tmp / "rot.npy"
                np.save(rot, output_rot_for(geo, seed))
            plan = Plan(bulk=kind, head=kind, embedding=kind, kv_proj=kind, gdn_gate=kind, n_layers=geo.n_layer)
            path = reader / f"toy-{kind.lower()}{'-tied' if tie else ''}.seed"
            export(src.path, tmp / "out.gguf", tmp / "no-packs", plan, LLAMA_DIR, torch.device("cpu"), tie_head=tie,
                   rot=rot)
            path.write_bytes((tmp / "out.gguf").read_bytes())
            written.append(path)

    def padded(raw: bytes, data: bytes = b"") -> bytes:
        """Pad the header to 32 bytes and add the data section."""
        return raw + b"\0" * ((-len(raw)) % 32) + data

    files = {
        bad_reader / "long-scalar-array.seed": header(0, [("a", struct.pack("<IIQ", 9, 0, 4_000_000))]),
        bad_reader / "offset-wrap.seed": padded(header(1, []) + tensor_info("t", [4], 0, 2**64 - 64),
                                                np.arange(4, dtype=np.float32).tobytes()),
        # A Q4_0 tensor with no dimension: one element, not a full block.
        bad_reader / "zero-dim-block-type.seed": padded(header(1, []) + tensor_info("t", [], 2, 0), bytes(32)),
        # gguf.cpp:576: the KV type 512 goes into enum gguf_type before the range check.
        ggml / "kv-type-enum.seed": header(0, [("a", struct.pack("<I", 512))]),
        # gguf.cpp:585: the array element type 512 goes into enum gguf_type before the range check.
        ggml / "array-type-enum.seed": header(0, [("a", struct.pack("<IIQ", 9, 512, 0))]),
        # gguf.cpp:715: the tensor type 512 goes into enum ggml_type before the range check.
        ggml / "tensor-type-enum.seed": padded(header(1, []) + tensor_info("t", [32], 512, 0), bytes(64)),
        # gguf.cpp:712-715: the file stops in the tensor type, thus the failed read leaves the type
        # uninitialized (MSan, the two profiles) and the range check loads it (UBSan, debug profile).
        ggml / "tensor-type-short-read.seed": header(1, []) + tensor_info("t", [4], 0, 0)[:-12] + b"\1\1",
        # gguf.cpp:694 and ggml.c:1289: ggml_nelements multiplies 2^32 three times in int64 before the guard.
        ggml / "nelements-overflow.seed": padded(header(1, []) + tensor_info("t", [2**32, 2**32, 2**32, 1], 0, 0),
                                                 bytes(64)),
        # gguf.cpp:852 (no_alloc false): the loader asks for 4 TiB of tensor data before it compares the size with
        # the file, and ggml_init aborts the process when the allocation fails.
        full / "alloc-before-size.seed": padded(header(1, []) + tensor_info("t", [2**20, 2**20], 0, 0), bytes(64)),
    }
    for path, raw in files.items():
        path.write_bytes(raw)
        written.append(path)
    gen = np.random.default_rng(5)
    for i, (kind, search, rows, nb) in enumerate(((0, 0, 1, 1), (1, 1, 2, 2), (2, 1, 3, 1), (1, 0, 1, 4))):
        values = (gen.standard_normal(rows * nb * 32) * 10.0 ** gen.uniform(-6, 3)).astype(np.float32)
        path = pack / f"pack-{i}.seed"
        path.write_bytes(bytes([kind, search, rows - 1, nb - 1]) + values.tobytes())
        written.append(path)
    return written


def main() -> int:
    """Run the command line. Give the exit status."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("files", help="write the phone set, the KL bases and the host results")
    sub.add_parser("commands", help="print the adb commands of the phone runs")
    sub.add_parser("seeds", help="write the seed files of the atheris reader fuzzer")
    c = sub.add_parser("compare", help="the table of the pulled phone logs")
    c.add_argument("pulled", type=Path)
    a = p.parse_args()
    if a.cmd == "files":
        manifest = build_files()
        LOG.info("wrote %d files and %s", len(manifest["files"]), OUT / "manifest.json")
    elif a.cmd == "commands":
        sys.stdout.write(phone_commands())
    elif a.cmd == "seeds":
        for path in write_seeds():
            LOG.info("wrote %s (%d bytes)", path, path.stat().st_size)
    else:
        table, failed = compare(a.pulled)
        sys.stdout.write(table)
        if failed:
            LOG.error("%d phone runs failed: no log, an exit status other than 0, or no final KL statistics block",
                      failed)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
