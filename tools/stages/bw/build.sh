#!/usr/bin/env bash
# Assemble the phone files of the stage "bw" (the DDR read ceiling of the NSP and the CPU, and the
# GEMV rates of the backend) in build/bw/phone, and write build/bw/phone-commands.txt.
#
#   tools/stages/bw/build.sh
#
# The steps:
#   1. tools/ddrbw/build.sh builds the probe (bin/ddrbw and v79/libddrbw_skel.so) in the container.
#   2. The script copies the files of the stage bench-kv (the libraries of the app of HEAD e8a3a07,
#      llama-bench, memprobe, gate.sh), and test-backend-ops of build/perf-tbo (the same libraries),
#      under the lock of build/perf-tbo.
#   3. tools/stages/bw/stage.py writes the test files of test-backend-ops and the phone command file.
#   4. The script writes SHA256SUMS, and links build/bw/stage.py and build/bw/build.sh to this directory.
#
# Then copy build/bw/phone-commands.txt to the same path on the laptop. Its first line copies the phone
# files from the box. Time: about 2 minutes (the container build of the probe).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/lib.sh"
cd "$REPO_ROOT"

readonly stage=build/bw
readonly out=$stage/phone
readonly tool=tools/stages/bw

tools/ddrbw/build.sh v79

rm -rf "$out"
mkdir -p "$out/bin" "$out/lib" "$out/tests"
cp -f build/ddrbw/bin/ddrbw "$out/bin/"
cp -f build/ddrbw/v79/libddrbw_skel.so "$out/lib/"
cp -f build/bench-kv/phone/bin/gate.sh build/bench-kv/phone/bin/llama-bench build/bench-kv/phone/bin/memprobe "$out/bin/"
cp -f build/bench-kv/phone/lib/*.so "$out/lib/"
(
    flock 9
    [[ -x build/perf-tbo/phone/bin/test-backend-ops ]] || die "build/perf-tbo/phone/bin/test-backend-ops does not exist"
    cp -f build/perf-tbo/phone/bin/test-backend-ops "$out/bin/"
) 9> build/perf-tbo/.lock

python3 "$tool/stage.py" tests --out "$out/tests"
python3 "$tool/stage.py" commands --out "$stage/phone-commands.txt"
ln -sfn "../../$tool/stage.py" "$stage/stage.py"
ln -sfn "../../$tool/build.sh" "$stage/build.sh"
(cd "$out" && sha256sum bin/* lib/* tests/* > SHA256SUMS)
cat "$out/SHA256SUMS"
