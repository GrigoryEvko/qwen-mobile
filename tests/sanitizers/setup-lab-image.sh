#!/usr/bin/env bash
# Make the Hexagon SDK image available under the two names that the scripts use.
#
# scripts/lib.sh pins SNAPDRAGON_IMAGE by the digest of its amd64 manifest
# (ghcr.io/snapdragon-toolchain/arm64-android@sha256:c012b817...). The kernel
# lab (tools/htp-lab/run.sh) uses the tag ghcr.io/snapdragon-toolchain/
# arm64-android:v0.7, whose multi-arch index contains that manifest. The
# script pulls the pinned digest if it is missing, and gives it the tag
# v0.7, thus the two names are the same image and no second download of the
# tag occurs.
#
# Usage:
#   tests/sanitizers/setup-lab-image.sh
#
# Requirements: podman (or docker, the engine of scripts/lib.sh), and a
# network connection to ghcr.io when the digest is not present. The image
# has approximately 10 GB.
#
# Exit status: 0 if the two names are present, not 0 if not.

set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/lib.sh"

readonly LAB_TAG=ghcr.io/snapdragon-toolchain/arm64-android:v0.7

main() {
    local engine id
    engine="$(container_engine)"
    if ! "$engine" image inspect "$SNAPDRAGON_IMAGE" > /dev/null 2>&1; then
        echo "[setup-lab-image] Pull $SNAPDRAGON_IMAGE." >&2
        "$engine" pull "$SNAPDRAGON_IMAGE" >&2
    fi
    id="$("$engine" image inspect --format '{{.Id}}' "$SNAPDRAGON_IMAGE")"
    if [[ "$("$engine" image inspect --format '{{.Id}}' "$LAB_TAG" 2> /dev/null || true)" != "$id" ]]; then
        "$engine" tag "$id" "$LAB_TAG"
    fi
    echo "[setup-lab-image] $LAB_TAG and $SNAPDRAGON_IMAGE are the image ${id:0:12}." >&2
}

main "$@"
