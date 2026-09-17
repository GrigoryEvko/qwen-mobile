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
import java.util.Locale
import kotlin.math.abs

/** Readers for the thermal state, the clocks, the battery, and the memory of the phone. */
object SystemStatus {
    private val thermalNames = listOf("NONE", "LIGHT", "MODERATE", "SEVERE", "CRITICAL", "EMERGENCY", "SHUTDOWN")

    /** One reading of the phone state. Clocks are in MHz, the temperature in degrees Celsius. */
    data class State(
        val thermalStatus: Int,
        val thermalName: String,
        val headroom: String,
        val cpu0Cur: String,
        val cpu0Cap: String,
        val cpu7Cur: String,
        val cpu7Cap: String,
        val batteryTemp: Double,
        val batteryLevel: Int,
        val source: String,
        val charging: Boolean,
        val availMb: Long,
        val totalMb: Long,
        val gameMode: String,
        val adpf: String,
        val cgroup: String,
        /** The instantaneous battery current in microamperes, with the sign of the phone, or [PowerSample.INVALID]. */
        val currentUa: Long,
        /** The battery voltage in millivolts. */
        val voltageMv: Int,
    ) {
        /** True at thermal status 0, the only state of the measurement protocol. */
        val cool: Boolean get() = thermalStatus == 0

        /** The reading of the battery current as a sample, for its power. */
        val power: PowerSample get() = PowerSample(0, currentUa, voltageMv)

        /** The instantaneous power of the phone, as a text with its direction. */
        fun powerText(): String {
            if (!power.valid) {
                return "n/a"
            }
            val direction = if (currentUa < 0) "discharge" else "charge"
            return String.format(
                Locale.US, "%.2f W (%.0f mA %s at %d mV)", power.watts, abs(currentUa) / 1000.0, direction, voltageMv,
            )
        }

        /** The lines of a benchmark record. */
        fun lines(): List<String> = listOf(
            "thermal: status $thermalStatus $thermalName, headroom(10s) $headroom" +
                if (cool) "" else "   <- NOT 0, wait before a measurement",
            "cpu0: $cpu0Cur / cap $cpu0Cap MHz, cpu7: $cpu7Cur / cap $cpu7Cap MHz",
            "battery: $batteryTemp C, $batteryLevel %, on $source",
            "power: ${powerText()}",
            "memory: $availMb MB available of $totalMb MB",
            "game mode: $gameMode, ADPF: $adpf, cgroup: $cgroup",
        )
    }

    /** The Android thermal status, 0 to 6. */
    fun thermalStatus(context: Context): Int {
        val pm = context.getSystemService(PowerManager::class.java)
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) pm.currentThermalStatus else -1
    }

    /** Read the phone state one time. */
    fun read(context: Context): State {
        val status = thermalStatus(context)
        val headroom = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            val h = context.getSystemService(PowerManager::class.java).getThermalHeadroom(10)
            if (h.isNaN()) "n/a" else "%.2f".format(h)
        } else {
            "n/a"
        }

        val battery = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val temp = battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1
        val plugged = battery?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0
        val level = battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val voltage = battery?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) ?: 0
        val current = BatteryReader(context).currentUa()
        val source = when (plugged) {
            BatteryManager.BATTERY_PLUGGED_AC -> "AC"
            BatteryManager.BATTERY_PLUGGED_USB -> "USB"
            BatteryManager.BATTERY_PLUGGED_WIRELESS -> "wireless"
            else -> "battery"
        }

        val am = context.getSystemService(ActivityManager::class.java)
        val mem = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }

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

        return State(
            thermalStatus = status,
            thermalName = thermalNames.getOrNull(status) ?: "n/a",
            headroom = headroom,
            cpu0Cur = mhz(cpuFreq(0, "scaling_cur_freq")),
            cpu0Cap = mhz(cpuFreq(0, "scaling_max_freq")),
            cpu7Cur = mhz(cpuFreq(7, "scaling_cur_freq")),
            cpu7Cap = mhz(cpuFreq(7, "scaling_max_freq")),
            batteryTemp = temp / 10.0,
            batteryLevel = level,
            source = source,
            charging = plugged != 0,
            availMb = mem.availMem shr 20,
            totalMb = mem.totalMem shr 20,
            gameMode = gameMode,
            adpf = adpf,
            cgroup = cgroup(),
            currentUa = current,
            voltageMv = voltage,
        )
    }

    /** The phone state as the lines of a benchmark record. */
    fun snapshot(context: Context): String = read(context).lines().joinToString("\n")

    private fun cpuFreq(cpu: Int, name: String): Long? =
        runCatching { File("/sys/devices/system/cpu/cpu$cpu/cpufreq/$name").readText().trim().toLong() }.getOrNull()

    private fun mhz(khz: Long?): String = khz?.let { (it / 1000).toString() } ?: "?"

    private fun cgroup(): String =
        runCatching {
            File("/proc/self/cgroup").readLines().firstOrNull { it.contains(":cpu:") }?.substringAfterLast(':')
        }.getOrNull() ?: "?"
}
