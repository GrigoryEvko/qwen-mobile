package ai.airi.qwenmobile

import android.app.ActivityManager
import android.app.GameManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PerformanceHintManager
import android.os.PowerManager
import java.io.File

/** Readers for the thermal state, the clocks, and the memory of the phone. */
object SystemStatus {
    private val thermalNames = listOf("NONE", "LIGHT", "MODERATE", "SEVERE", "CRITICAL", "EMERGENCY", "SHUTDOWN")

    /** The Android thermal status, 0 to 6. */
    fun thermalStatus(context: Context): Int {
        val pm = context.getSystemService(PowerManager::class.java)
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) pm.currentThermalStatus else -1
    }

    /**
     * A multi-line text with the state that a benchmark record needs:
     * thermal status, clock caps, battery, memory, game mode, ADPF.
     */
    fun snapshot(context: Context): String {
        val lines = ArrayList<String>()
        val status = thermalStatus(context)
        val statusName = thermalNames.getOrNull(status) ?: "n/a"
        val headroom = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val pm = context.getSystemService(PowerManager::class.java)
            val h = pm.getThermalHeadroom(10)
            if (h.isNaN()) "n/a" else "%.2f".format(h)
        } else {
            "n/a"
        }
        lines += "thermal: status $status $statusName, headroom(10s) $headroom" +
            if (status == 0) "" else "   <- NOT 0, wait before a measurement"

        lines += "cpu0: ${mhz(cpuFreq(0, "scaling_cur_freq"))} / cap ${mhz(cpuFreq(0, "scaling_max_freq"))} MHz, " +
            "cpu7: ${mhz(cpuFreq(7, "scaling_cur_freq"))} / cap ${mhz(cpuFreq(7, "scaling_max_freq"))} MHz"

        val battery = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val temp = battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1
        val plugged = battery?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0
        val level = battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val source = when (plugged) {
            BatteryManager.BATTERY_PLUGGED_AC -> "AC"
            BatteryManager.BATTERY_PLUGGED_USB -> "USB"
            BatteryManager.BATTERY_PLUGGED_WIRELESS -> "wireless"
            else -> "battery"
        }
        lines += "battery: ${temp / 10.0} C, $level %, on $source"

        val am = context.getSystemService(ActivityManager::class.java)
        val mem = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        lines += "memory: ${mem.availMem shr 20} MB available of ${mem.totalMem shr 20} MB"

        val gameMode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            when (context.getSystemService(GameManager::class.java).gameMode) {
                GameManager.GAME_MODE_UNSUPPORTED -> "unsupported"
                GameManager.GAME_MODE_STANDARD -> "standard"
                GameManager.GAME_MODE_PERFORMANCE -> "performance"
                GameManager.GAME_MODE_BATTERY -> "battery"
                else -> "custom"
            }
        } else {
            "n/a"
        }
        val adpf = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            if (context.getSystemService(PerformanceHintManager::class.java) != null) "available" else "absent"
        } else {
            "n/a"
        }
        lines += "game mode: $gameMode, ADPF: $adpf, cgroup: ${cgroup()}"
        return lines.joinToString("\n")
    }

    private fun cpuFreq(cpu: Int, name: String): Long? =
        runCatching { File("/sys/devices/system/cpu/cpu$cpu/cpufreq/$name").readText().trim().toLong() }.getOrNull()

    private fun mhz(khz: Long?): String = khz?.let { (it / 1000).toString() } ?: "?"

    private fun cgroup(): String =
        runCatching {
            File("/proc/self/cgroup").readLines().firstOrNull { it.contains(":cpu:") }?.substringAfterLast(':')
        }.getOrNull() ?: "?"
}
