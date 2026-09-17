#!/usr/bin/env bash
# Sign a release APK with the release key, in the container of ci/Containerfile.
#
#   scripts/sign-apk.sh <unsigned.apk> <signed.apk>
#
# The key comes from these variables:
#
#   ANDROID_KEYSTORE_BASE64    the keystore, base64, or
#   ANDROID_KEYSTORE_FILE      the path of the keystore
#   ANDROID_KEYSTORE_PASSWORD  the password of the store
#   ANDROID_KEY_ALIAS          the alias of the key
#   ANDROID_KEY_PASSWORD       the password of the key, the store password when absent
#
# Without ANDROID_KEYSTORE_BASE64 and ANDROID_KEYSTORE_FILE the script stops
# with the code 2, thus a caller can make the signature optional.
#
# The signature does not change the entries of the APK, thus the unsigned file
# stays the artifact that two machines compare.

set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

cd "$REPO_ROOT"

[[ $# -eq 2 ]] || die "usage: scripts/sign-apk.sh <unsigned.apk> <signed.apk>"
unsigned=$1
signed=$2
[[ -f "$unsigned" ]] || die "no such file: $unsigned"

keydir="build/cache/signing"
mkdir -p "$keydir"
chmod 700 "$keydir"
keystore="$keydir/release.jks"
trap 'rm -f "$keystore"' EXIT

if [[ -n "${ANDROID_KEYSTORE_BASE64:-}" ]]; then
    printf '%s' "$ANDROID_KEYSTORE_BASE64" | base64 -d > "$keystore"
elif [[ -n "${ANDROID_KEYSTORE_FILE:-}" ]]; then
    [[ -f "$ANDROID_KEYSTORE_FILE" ]] || die "no such keystore: $ANDROID_KEYSTORE_FILE"
    cp "$ANDROID_KEYSTORE_FILE" "$keystore"
else
    echo "sign: no key, the APK stays unsigned" >&2
    exit 2
fi
[[ -n "${ANDROID_KEYSTORE_PASSWORD:-}" ]] || die "ANDROID_KEYSTORE_PASSWORD is empty"
[[ -n "${ANDROID_KEY_ALIAS:-}" ]] || die "ANDROID_KEY_ALIAS is empty"

image=$(ensure_apk_image)
mkdir -p "$(dirname "$signed")"
cp "$unsigned" "$signed"

container_run \
    -e KS_PASS="$ANDROID_KEYSTORE_PASSWORD" \
    -e KEY_PASS="${ANDROID_KEY_PASSWORD:-$ANDROID_KEYSTORE_PASSWORD}" \
    -e KEY_ALIAS="$ANDROID_KEY_ALIAS" \
    -e KEYSTORE="/workspace/$keystore" \
    -e TARGET="/workspace/$signed" \
    "$image" bash -euo pipefail -c '
tools=$(ls -d "$ANDROID_HOME"/build-tools/* | sort -V | tail -1)
"$tools/zipalign" -c 4 "$TARGET" || {
    "$tools/zipalign" -p -f 4 "$TARGET" "$TARGET.aligned"
    mv "$TARGET.aligned" "$TARGET"
}
"$tools/apksigner" sign \
    --ks "$KEYSTORE" --ks-pass "env:KS_PASS" \
    --ks-key-alias "$KEY_ALIAS" --key-pass "env:KEY_PASS" \
    --v2-signing-enabled true --v3-signing-enabled true \
    "$TARGET"
"$tools/apksigner" verify --print-certs "$TARGET"
'
echo "sign: $signed"
