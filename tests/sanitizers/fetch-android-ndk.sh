#!/usr/bin/env bash
# Download and unpack the pinned Android NDK r30 (30.0.16248370) in the layout
# of the Android SDK: ~/Android/Sdk/ndk/30.0.16248370.
#
# The Android ASan runtime of tests/sanitizers/build-asan-android-runtime.sh
# is bit-identical only with the same NDK at the same path and the same
# repository path: the paths go into the runtime. The runtime that the phone
# runs (sha256 313e8c71...) comes from this NDK at
# /home/grigory/Android/Sdk/ndk/30.0.16248370, with the repository at
# /home/grigory/airi/qwen-mobile. build-asan-android-runtime.sh takes the
# newest NDK of ~/Android/Sdk/ndk when it gets no --ndk, thus this layout is
# the preset.
#
# Usage:
#   tests/sanitizers/fetch-android-ndk.sh [--sdk-ndk-dir DIR] [--zip-dir DIR]
#
#   --sdk-ndk-dir DIR   The ndk directory of the SDK layout. The NDK goes into
#                       DIR/30.0.16248370. The preset value is ~/Android/Sdk/ndk.
#   --zip-dir DIR       The directory of the downloaded zip. The preset value
#                       is build/cache/ndk of the repository.
#
# Output: the NDK root. The script prints it on the last line of stdout, thus
# a caller can do: ndk="$(tests/sanitizers/fetch-android-ndk.sh | tail -n 1)"
#
# Resources: 740 MB to download, 2.3 GB unpacked, less than 1 minute on a
# fast network. No container is necessary.
#
# Exit status: 0 if the NDK is there and correct, not 0 if not.

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly NDK_ZIP=android-ndk-r30-linux.zip
readonly NDK_URL="https://dl.google.com/android/repository/$NDK_ZIP"
readonly NDK_SHA256=753611f410d002cfcd3f3dc2ef49aad532089d3180b436c060a90bf0fcb64df2
readonly NDK_REVISION=30.0.16248370
SDK_NDK_DIR="$HOME/Android/Sdk/ndk"
ZIP_DIR="$REPO_ROOT/build/cache/ndk"

# Write a message to stderr with the name of the script.
log() {
    echo "[fetch-android-ndk] $*" >&2
}

# Write an error message to stderr and stop with exit status 1.
die() {
    log "ERROR: $*"
    exit 1
}

# Print the Pkg.Revision of an unpacked NDK, or nothing.
ndk_revision() {
    local line
    [[ -f "$1/source.properties" ]] || return 0
    while IFS= read -r line; do
        [[ "$line" == Pkg.Revision* ]] && { echo "${line#*= }"; return 0; }
    done < "$1/source.properties"
}

main() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --sdk-ndk-dir) SDK_NDK_DIR="$(realpath -m "$2")"; shift 2 ;;
            --zip-dir) ZIP_DIR="$(realpath -m "$2")"; shift 2 ;;
            -h|--help) head -29 "${BASH_SOURCE[0]}" | tail -28; exit 0 ;;
            *) die "The option '$1' is not known. Use --help for the usage." ;;
        esac
    done
    command -v curl > /dev/null || die "curl is not in PATH."
    command -v unzip > /dev/null || die "unzip is not in PATH."
    local root="$SDK_NDK_DIR/$NDK_REVISION" zip="$ZIP_DIR/$NDK_ZIP" have tmp
    if [[ "$(ndk_revision "$root")" == "$NDK_REVISION" ]]; then
        log "The NDK $NDK_REVISION is in $root."
        echo "$root"
        return 0
    fi
    mkdir -p "$ZIP_DIR" "$SDK_NDK_DIR"
    if [[ ! -f "$zip" ]] || [[ "$(sha256sum "$zip" | cut -d' ' -f1)" != "$NDK_SHA256" ]]; then
        log "Download $NDK_URL."
        curl -sSfL -o "$zip.part" "$NDK_URL" || die "The download of $NDK_URL failed."
        mv -f "$zip.part" "$zip"
    fi
    have="$(sha256sum "$zip" | cut -d' ' -f1)"
    [[ "$have" == "$NDK_SHA256" ]] || die "$zip has sha256 $have, not $NDK_SHA256."
    log "Unpack $zip into $root."
    tmp="$(mktemp -d "$SDK_NDK_DIR/.unpack.XXXXXX")"
    # The trap removes the temporary directory on each exit path.
    trap 'rm -rf "$tmp"' EXIT
    unzip -q "$zip" -d "$tmp" || die "The unzip of $zip failed."
    rm -rf "$root"
    mv "$tmp/android-ndk-r30" "$root"
    [[ "$(ndk_revision "$root")" == "$NDK_REVISION" ]] \
        || die "$root has the revision '$(ndk_revision "$root")', not $NDK_REVISION."
    log "The NDK $NDK_REVISION is in $root."
    echo "$root"
}

main "$@"
