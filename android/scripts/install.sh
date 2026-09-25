#!/bin/bash
# Install the release APK on the phone. If OxygenOS shows its install dialog,
# press the accept button of that dialog, which the UI dump finds by its text.
# Never press a fixed screen position: when no dialog shows, a fixed tap lands
# on the launcher dock and opens a different app.
# The phone is shared, thus the install runs under the device lock.
#
#   android/scripts/install.sh [apk]
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)
# The default is the artifact of scripts/build-apk.sh. That build runs in the
# container on a staged copy, thus it never writes app/build/outputs, and a
# stale APK there would install silently over a fresh one.
APK=${1:-$HERE/../build/apk/app-release.apk}
test -f "$APK" || { echo "No APK at $APK. Build with scripts/build-apk.sh first." >&2; exit 1; }

# The longest time to look for the dialog, in seconds.
DIALOG_WAIT_S=${DIALOG_WAIT_S:-60}

# Print "x y", the center of the accept button of an install dialog on the
# screen, or print nothing. The button must have one of the accept labels and
# belong to an installer package, thus a launcher icon never matches.
accept_button() {
    adb shell uiautomator dump /data/local/tmp/install-ui.xml > /dev/null 2>&1 || return 0
    adb exec-out cat /data/local/tmp/install-ui.xml 2> /dev/null | python3 -c '
import re
import sys
import xml.etree.ElementTree as ET

LABELS = {"install", "continue installation", "continue install", "update",
          "установить", "продолжить установку", "обновить"}
PACKAGES = ("packageinstaller", "appdetail", "safecenter", "securitypermission")
try:
    root = ET.fromstring(sys.stdin.read())
except ET.ParseError:
    sys.exit(0)
for node in root.iter("node"):
    text = node.get("text", "").strip().lower()
    package = node.get("package", "")
    bounds = re.fullmatch(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", node.get("bounds", ""))
    if text in LABELS and any(p in package for p in PACKAGES) and bounds:
        x1, y1, x2, y2 = map(int, bounds.groups())
        print((x1 + x2) // 2, (y1 + y2) // 2)
        break
'
}

exec 9> "$HERE/.device-lock"
flock 9

adb install -r "$APK" &
install_pid=$!
pressed=0
for _ in $(seq 1 $((DIALOG_WAIT_S / 2))); do
    kill -0 "$install_pid" 2> /dev/null || break
    sleep 2
    if [ "$pressed" -eq 0 ]; then
        xy=$(accept_button)
        if [ -n "$xy" ]; then
            echo "install dialog: accept button at $xy"
            # shellcheck disable=SC2086 # xy holds two words, x and y.
            adb shell input tap $xy
            pressed=1
        fi
    fi
done
if kill -0 "$install_pid" 2> /dev/null; then
    echo "The install waits for a confirmation that this script cannot find. Confirm it on the phone." >&2
fi
wait "$install_pid"
