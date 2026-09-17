#!/bin/bash
# Tap the center of the first screen node whose text or resource-id contains the argument.
#
#   android/scripts/tap.sh <text-or-id>
set -euo pipefail

test $# -eq 1 || { echo "usage: tap.sh <text-or-id>" >&2; exit 1; }
HERE=$(cd "$(dirname "$0")" && pwd)
LINE=$("$HERE/ui.sh" | grep -F -- "$1" | head -1)
test -n "$LINE" || { echo "No node matches '$1'." >&2; exit 1; }
BOUNDS=${LINE##*| }
read -r X Y <<<"$(python3 -c "
import re, sys
x1, y1, x2, y2 = map(int, re.findall(r'\d+', sys.argv[1]))
print((x1 + x2) // 2, (y1 + y2) // 2)
" "$BOUNDS")"
flock "$HERE/../.device-lock" -c "adb shell input tap $X $Y"
echo "tapped '$1' at $X $Y"
