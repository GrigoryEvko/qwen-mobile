#!/usr/bin/env bash
# Build the release APK in the pinned container of ci/Containerfile, run the
# unit tests there, and write build/hashes-apk.txt.
#
#   scripts/build-apk.sh
#
# Do not run this script on its own. Use scripts/build-all.sh, which runs
# scripts/build-native.sh first and stops at the first failure. This script
# packages whatever sits in android/snapdragon/jniLibs, thus a native build
# that failed leaves the libraries of the run before it. Step 0 refuses that
# case.
#
# The native libraries must exist: scripts/build-native.sh writes them to
# android/snapdragon/jniLibs. The steps:
#
#   1. Build the image from ci/Containerfile. The tag holds a hash of that
#      file, thus a change of the file gives a new image.
#   2. Stage the Android project into build/apk/src/android: the tracked files
#      of android/, the untracked files that git does not ignore, and the
#      jniLibs. The host copies of local.properties, .gradle and build/ stay
#      out, and the build does not touch the working tree of the host.
#   3. Write local.properties for the container: the SDK at /opt/android-sdk
#      and llama.cpp at /workspace/third_party/llama.cpp.
#   4. If ~/.android/debug.keystore exists on the host, copy it into
#      build/cache/home/.android, which is ANDROID_USER_HOME in the container.
#      Java reads user.home from the passwd file, not from HOME, and the uid
#      of the caller has no entry there, thus AGP needs that variable to find
#      the key. Without a key, AGP makes a new one on each run, and the signed
#      APK differs from run to run.
#   5. In the container: the unit tests (AGP 9 makes them for the debug build
#      type only), then assembleRelease with -Punsigned=true
#      (app-release-unsigned.apk, the reproducible artifact), then
#      assembleRelease with the debug key (app-release.apk, for adb install).
#   6. Copy the two APKs and the test reports to build/apk, and write
#      build/hashes-apk.txt.
#
# Determinism:
#   - The staged source is new on each run, thus each run is a clean build.
#     --no-build-cache keeps Gradle from a copy of a previous output.
#   - The Gradle distribution has a pinned checksum, the plugin classpath and
#     the app dependencies have lockfiles, and the SDK is in the image.
#   - -Pandroid.builder.sdkDownload=false stops AGP from a download of an SDK
#     component that the image does not have.
#   - AGP writes a constant time (1981-01-01 01:01) into each zip entry.
#
# The Gradle caches live in build/cache/gradle. Remove that directory to
# download the distribution and the dependencies again.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

cd "$REPO_ROOT"

jnilibs="android/snapdragon/jniLibs/arm64-v8a"
[[ -f "$jnilibs/libqwenmobile.so" ]] || die "no native libraries in $jnilibs: scripts/build-native.sh"

# 0. The libraries must come from the last native build.
#
# scripts/build-native.sh writes build/hashes-native.txt from the files that it
# put in jniLibs. This script packages whatever sits in that directory. Thus a
# native build that stops at a compile error leaves the libraries of the run
# before it, and an APK built after it carries code that nobody measured. That
# happened twice on 2026-09-19: the APK held libggml-hexagon.so d70a73b3 where
# the build output held 0566d66d, and the numbers of that session described the
# previous build.
#
# The check compares the two tables and stops when they differ. Run
# scripts/build-all.sh, which chains the two scripts and stops at the first
# failure.
native_hashes="build/hashes-native.txt"
[[ -f "$native_hashes" ]] || die "no $native_hashes: run scripts/build-native.sh (or scripts/build-all.sh)"
if ! diff -u "$native_hashes" <(sha256_table "$jnilibs"/*.so) > build/hashes-native.diff; then
    echo "apk: $jnilibs does not match $native_hashes" >&2
    echo "apk: the last native build did not produce these libraries, thus the APK would" >&2
    echo "apk: carry code that no measurement describes. The difference:" >&2
    cat build/hashes-native.diff >&2
    die "stale native libraries; run scripts/build-all.sh"
fi
rm -f build/hashes-native.diff
echo "apk: native libraries match $native_hashes"

# 1. The image.
image=$(ensure_apk_image)

# 2. The staged source.
stage="build/apk/src"
rm -rf "$stage" build/apk/app-release-unsigned.apk build/apk/app-release.apk build/apk/test-results
mkdir -p "$stage"
while IFS= read -r -d '' file; do
    [[ -e "$file" ]] || continue
    cp --parents -- "$file" "$stage/"
done < <(git ls-files -z --cached --others --exclude-standard -- android)
mkdir -p "$stage/android/snapdragon"
cp -r "android/snapdragon/jniLibs" "$stage/android/snapdragon/jniLibs"

# 3. The properties of the container.
printf 'sdk.dir=/opt/android-sdk\nllama.dir=/workspace/third_party/llama.cpp\n' > "$stage/android/local.properties"

# 4. The debug key.
mkdir -p build/cache/home/.android
if [[ -f "$HOME/.android/debug.keystore" ]]; then
    cp "$HOME/.android/debug.keystore" build/cache/home/.android/debug.keystore
    echo "apk: the debug key of this machine signs app-release.apk"
else
    echo "apk: no ~/.android/debug.keystore, AGP makes a new debug key"
fi

# 5. The build.
SOURCE_DATE_EPOCH=$(source_date_epoch)
mkdir -p build/cache/gradle
echo "apk: version $(version_name) ($(version_code))"
container_run \
    -v "$REPO_ROOT/$stage/android:/workspace/android" \
    -w /workspace/android \
    -e SOURCE_DATE_EPOCH="$SOURCE_DATE_EPOCH" \
    -e GRADLE_USER_HOME=/workspace/build/cache/gradle \
    -e ANDROID_USER_HOME=/workspace/build/cache/home/.android \
    -e VERSION_CODE="$(version_code)" \
    "$image" bash -euo pipefail -c '
common=(--no-daemon --no-build-cache --console=plain -Pprebuilt=true -Pandroid.builder.sdkDownload=false -PversionCode="$VERSION_CODE")
./gradlew "${common[@]}" -Punsigned=true :app:testDebugUnitTest :app:assembleRelease
cp app/build/outputs/apk/release/app-release-unsigned.apk /workspace/build/apk/
./gradlew "${common[@]}" :app:assembleRelease
cp app/build/outputs/apk/release/app-release.apk /workspace/build/apk/
cp -r app/build/test-results/testDebugUnitTest /workspace/build/apk/test-results
'

# 6. The hashes.
sha256_table build/apk/app-release-unsigned.apk build/apk/app-release.apk > build/hashes-apk.txt
echo "apk: build/hashes-apk.txt"
cat build/hashes-apk.txt

# An entry of the APK must not contain the container path.
scratch=$(mktemp -d "${TMPDIR:-/tmp}/build-apk.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
unzip -q -o build/apk/app-release-unsigned.apk -d "$scratch"
leaks=$(grep -r -a -l '/workspace' "$scratch" || true)
[[ -z "$leaks" ]] || echo "apk: note: these entries contain /workspace:" "$leaks"
