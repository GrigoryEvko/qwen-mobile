#!/bin/bash
# Print the visible nodes of the phone screen, one per line: text | resource-id | bounds.
# The dump can fail while the screen changes, thus it tries three times.
#
#   android/scripts/ui.sh
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)

dump() {
    for _ in 1 2 3; do
        if adb shell uiautomator dump /sdcard/ui.xml >/dev/null 2>&1 && adb shell cat /sdcard/ui.xml 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    return 1
}

flock "$HERE/../.device-lock" -c "$(declare -f dump); dump" | python3 "$HERE/ui_nodes.py"
