#!/bin/bash
# Install the release APK on the phone and accept the OxygenOS install dialog.
# The phone is shared, thus the install runs under the device lock.
#
#   android/scripts/install.sh [apk]
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
APK=${1:-$HERE/app/build/outputs/apk/release/app-release.apk}
test -f "$APK" || { echo "No APK at $APK. Build with ./gradlew assembleRelease first." >&2; exit 1; }

# The install waits for the dialog. The tap accepts it after 6 s.
flock "$HERE/.device-lock" -c "
adb install -r '$APK' &
sleep 6
adb shell input tap 398 2561
wait
"
