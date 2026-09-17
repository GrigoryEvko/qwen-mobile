#!/usr/bin/env bash
# Build the native libraries and the APK, then print the two hash tables.
#
#   scripts/build-all.sh
#
# The steps are scripts/build-native.sh, which applies the patch series
# first, and scripts/build-apk.sh. The environment variables of the two
# scripts (KEEP_BUILD, NATIVE_TARGETS) apply here too. The outputs:
#
#   android/snapdragon/jniLibs/arm64-v8a/*.so
#   build/hashes-native.txt
#   build/apk/app-release-unsigned.apk, build/apk/app-release.apk
#   build/apk/test-results/
#   build/hashes-apk.txt

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

cd "$REPO_ROOT"
scripts/build-native.sh
scripts/build-apk.sh

echo
echo "== build/hashes-native.txt"
cat build/hashes-native.txt
echo
echo "== build/hashes-apk.txt"
cat build/hashes-apk.txt
