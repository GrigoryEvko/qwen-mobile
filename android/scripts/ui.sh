#!/bin/bash
# Print the visible nodes of the phone screen, one per line:
# text | resource-id | bounds. The dump can fail while the screen changes, thus it tries three times.
#
#   android/scripts/ui.sh
set -euo pipefail

HERE=$(cd "$(dirname "$0")/.." && pwd)

dump() {
    for _ in 1 2 3; do
        if adb shell uiautomator dump /sdcard/ui.xml >/dev/null 2>&1 && adb shell cat /sdcard/ui.xml 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    return 1
}

flock "$HERE/.device-lock" -c "$(declare -f dump); dump" | python3 -c '
import re, sys
xml = sys.stdin.read()
for m in re.finditer(r"<node[^>]*>", xml):
    node = m.group(0)
    text = re.search(r" text=\"([^\"]*)\"", node)
    rid = re.search(r" resource-id=\"([^\"]*)\"", node)
    bounds = re.search(r" bounds=\"(\[[^\"]*\])\"", node)
    t = text.group(1) if text else ""
    r = rid.group(1).split("/")[-1] if rid else ""
    if t or r:
        print(f"{t} | {r} | {bounds.group(1) if bounds else \"\"}")
'
