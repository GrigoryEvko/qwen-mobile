#!/usr/bin/env bash
# Record a Perfetto trace of the phone for a number of seconds and pull the file.
#
# Usage: capture.sh [SECONDS] [OUT_FILE]
#
# SECONDS is the length of the trace (the preset value is 10). OUT_FILE is the
# trace file on the host (the preset value is qwen-DATE-TIME.pftrace in the
# current directory). Start the work in the app during the trace. Set
# ANDROID_SERIAL when more than one device is attached. Open the file at
# https://ui.perfetto.dev.
#
# The script reads the thermal status before and after the trace, because a
# measurement is correct only at status 0.

set -euo pipefail

seconds="${1:-10}"
out="${2:-qwen-$(date +%Y%m%d-%H%M%S).pftrace}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config="$here/qwen.pbtxt"
device_path="/data/misc/perfetto-traces/qwen.pftrace"

if ! [[ "$seconds" =~ ^[0-9]+$ ]] || [ "$seconds" -lt 1 ]; then
    echo "The number of seconds must be a positive integer, not '$seconds'." >&2
    exit 2
fi
if [ "$seconds" -gt 120 ]; then
    echo "The trace must not be longer than 120 seconds. The phone becomes hot, and the thermal status is not 0 after that." >&2
    exit 2
fi

thermal_status() {
    adb shell dumpsys thermalservice 2>/dev/null | grep -m 1 "Thermal Status" || echo "Thermal Status: unknown"
}

echo "before: $(thermal_status)"

# perfetto reads the configuration from stdin (-c -) as text (--txt). The duration
# line goes in front of the configuration. The timeout is the trace time plus 30
# seconds for the flush.
{ echo "duration_ms: $((seconds * 1000))"; cat "$config"; } \
    | timeout -s KILL "$((seconds + 30))" adb shell perfetto -c - --txt -o "$device_path"

echo "after: $(thermal_status)"

timeout -s KILL 120 adb pull "$device_path" "$out"
adb shell rm -f "$device_path"
echo "trace: $out"
