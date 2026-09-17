#!/bin/bash
# Take a screenshot of the phone and print its path.
# The directory comes from SHOT_DIR, or /tmp/qwen-shots.
#
#   android/scripts/shot.sh <name>
set -euo pipefail

NAME=${1:-shot}
DIR=${SHOT_DIR:-/tmp/qwen-shots}
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$DIR"
OUT="$DIR/$NAME.png"
flock "$HERE/.device-lock" -c "adb exec-out screencap -p > '$OUT'"
echo "$OUT"
