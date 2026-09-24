#!/system/bin/sh
# The gate before one phone run that loads a real model.
#
#   sh bin/gate.sh MIN_AVAILABLE_KB
#
# The gate stops the Qwen app (it holds its model in NPU buffers) and wakes the screen,
# then requires: the screen on (mWakefulness=Awake), thermal status 0, no charger, no
# measurement process that still runs, and at least MIN_AVAILABLE_KB of MemAvailable. It
# prints the conditions of the run: the screen state, the thermal status, the clock caps of
# cpu0 and cpu7, the battery level and temperature, and MemAvailable. It exits with 1 when one
# condition fails, thus "gate.sh N && run" does not start the run.
#
# With the screen off the kernel suspends the SoC while a run waits for the DSP: a decode token
# then waits 0.3 to 1.2 s (on the global counter; CLOCK_MONOTONIC shows about 160 ms of it),
# and the CPU clock caps fall. KEEP_SCREEN_STATE=1 skips the wake and the screen condition,
# for a run that measures the screen-off state on purpose.
#
# The build recipes of the phone stages (tools/stages/*/build.sh, tools/memprobe/build-phone.sh)
# copy this file to bin/gate.sh of the stage. tools/prof/run.py does the charger and thermal
# checks from the host through adb, and this gate does them on the phone before each run.
min_kb=${1:?usage: gate.sh MIN_AVAILABLE_KB}

am force-stop ai.airi.qwenmobile
am kill-all
if [ "${KEEP_SCREEN_STATE:-0}" != "1" ]; then
    input keyevent KEYCODE_WAKEUP
fi
sleep 2

wake=$(dumpsys power | grep -o 'mWakefulness=[A-Za-z]*' | head -n 1 | cut -d= -f2)

status=$(dumpsys thermalservice | grep "Thermal Status" | head -n 1 | tr -dc '0-9')
battery=$(dumpsys battery)
ac=$(echo "$battery" | grep "AC powered" | grep -c true)
usb=$(echo "$battery" | grep "USB powered" | grep -c true)
wireless=$(echo "$battery" | grep "Wireless powered" | grep -c true)
level=$(echo "$battery" | grep "^  level:" | tr -dc '0-9')
temp=$(echo "$battery" | grep "temperature:" | tr -dc '0-9')
avail=$(grep MemAvailable /proc/meminfo | tr -dc '0-9')
cap0=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq)
cap7=$(cat /sys/devices/system/cpu/cpu7/cpufreq/scaling_max_freq)
# The kernel keeps 15 characters of a process name, and "pgrep -f" would match the
# command line of the adb shell that starts this gate, thus the names go to "pgrep -x".
busy=$( (pgrep -x memprobe; pgrep -x kvkl; pgrep -x llama-bench; pgrep -x llama-perplexit; pgrep -x llama-completio) | head -n 1)

echo "gate: screen=$wake thermal=$status cap0=$cap0 cap7=$cap7 battery=$level% temp=$temp charger=ac$ac/usb$usb/wl$wireless MemAvailable=${avail}kB need=${min_kb}kB"

if [ "${KEEP_SCREEN_STATE:-0}" != "1" ] && [ "$wake" != "Awake" ]; then
    echo "gate: STOP, the screen is $wake. A run with the screen off stalls when the SoC suspends."
    exit 1
fi
if [ "$status" != "0" ]; then
    echo "gate: STOP, the thermal status is $status. Wait until it is 0."
    exit 1
fi
if [ "$ac$usb$wireless" != "000" ]; then
    echo "gate: STOP, a charger is connected. Disconnect it."
    exit 1
fi
if [ -n "$busy" ]; then
    echo "gate: STOP, a measurement process runs (pid $busy). Kill it by its pid."
    exit 1
fi
if [ "$avail" -lt "$min_kb" ]; then
    echo "gate: STOP, MemAvailable ${avail}kB is less than ${min_kb}kB. Close apps and do the run again."
    exit 1
fi
echo "gate: OK"
exit 0
