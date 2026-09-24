#!/usr/bin/env bash
# Build the phone files of the stage "sweep" (the HMX peak and the prefill op shapes of the 4B Q8_0) and
# write its command file.
#
#   tools/stages/sweep/build.sh
#
# The steps:
#   1. Under the lock build/perf-tbo/.lock (other stages use the same binary) and the container lock,
#      ninja builds test-backend-ops in build/bench-kv/android, the build directory of the stage bench-kv
#      (tools/stages/bench-kv/build.sh makes it). Thus the binary has the tree and the flags of the
#      libraries of that stage. The script copies it to build/perf-tbo/phone/bin.
#   2. tools/hmx-bench/build-i8.sh and tools/hmx-bench/build-sustain.sh build i8read.so and
#      hmx_sustain.so into build/sweep/phone/hmx.
#   3. tools/stages/sweep/stage.py files writes build/sweep/phone (the binary, the libraries of
#      build/bench-kv/phone/lib, the programs, the test files, SHA256SUMS) and build/sweep/phone-commands.txt.
#
# Then copy build/sweep/phone-commands.txt to the same path on the laptop. Its first line copies the phone
# files from the box. Time: about 5 minutes. Disk: about 130 MB.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/lib.sh"
cd "$REPO_ROOT"

[ -f build/bench-kv/android/build.ninja ] || die "build/bench-kv/android has no build: run tools/stages/bench-kv/build.sh"
mkdir -p build/perf-tbo/phone/bin build/sweep/phone/hmx

# 1. test-backend-ops
(
    flock 8
    (
        flock 9
        container_run "$SNAPDRAGON_IMAGE" bash -c "ninja -C build/bench-kv/android -j${JOBS:-24} test-backend-ops"
    ) 9> build/.container.lock > build/perf-tbo/build.log 2>&1 || die "the build failed, refer to build/perf-tbo/build.log"
    cp -f build/bench-kv/android/bin/test-backend-ops build/perf-tbo/phone/bin/
    (cd build/perf-tbo/phone && sha256sum bin/test-backend-ops > SHA256SUMS)
) 8> build/perf-tbo/.lock

# 2. The HMX programs. build-i8.sh does not take the container lock itself.
(
    flock 9
    tools/hmx-bench/build-i8.sh build/sweep/phone/hmx
) 9> build/.container.lock
tools/hmx-bench/build-sustain.sh build/sweep/phone/hmx

# 3. The stage files and the command file
tools/stages/sweep/stage.py files
