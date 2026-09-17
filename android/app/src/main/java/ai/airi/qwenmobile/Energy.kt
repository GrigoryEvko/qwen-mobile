package ai.airi.qwenmobile

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.SystemClock
import java.util.Locale
import kotlin.math.abs

/**
 * The energy of a benchmark run, from the battery of the phone.
 *
 * The shell user cannot read the current files of the power supply, but the
 * app reads the same values through the battery manager. One sample holds
 * the current and the voltage at a time, and a window of samples gives the
 * energy of the work that ran during the window.
 *
 * The arithmetic is in [PowerTrace] and [PowerSample], which touch no
 * Android class, thus a unit test measures them.
 */

/**
 * One reading of the battery.
 *
 * @param elapsedMs  The time of the reading, from a monotonic clock
 * @param currentUa  The instantaneous current in microamperes. The sign is a
 *   convention of the phone: most phones give a negative value while the
 *   battery supplies the load, some give a positive one
 * @param voltageMv  The battery voltage in millivolts
 */
data class PowerSample(val elapsedMs: Long, val currentUa: Long, val voltageMv: Int) {
    /** The power in watts. The magnitude of the current counts, not its sign. */
    val watts: Double get() = abs(currentUa) * 1e-6 * voltageMv * 1e-3

    /** True when the phone gave both values. */
    val valid: Boolean get() = currentUa != INVALID && currentUa != 0L && voltageMv > 0

    companion object {
        /** The battery manager gives this value for a property that the phone does not support. */
        const val INVALID = Long.MIN_VALUE
    }
}

/**
 * The samples of one measurement window, in the order of their time.
 *
 * The energy comes from the trapezoid rule over the samples, thus a sample
 * that arrives late does not bias the result. All methods are O(n) in the
 * number of samples, and O(1) for [size].
 */
class PowerTrace {
    private val samples = ArrayList<PowerSample>()

    /** Keep a sample. The caller gives the samples in the order of their time. */
    fun add(sample: PowerSample) {
        if (sample.valid) {
            samples += sample
        }
    }

    /** The number of samples. */
    val size: Int get() = samples.size

    /** True when the window holds no usable sample. */
    val empty: Boolean get() = samples.isEmpty()

    /** The time from the first sample to the last one. */
    fun durationMs(): Long = if (samples.size < 2) 0L else samples.last().elapsedMs - samples.first().elapsedMs

    /** The mean of the battery voltage over the samples, in millivolts. */
    fun meanVoltageMv(): Double = if (samples.isEmpty()) 0.0 else samples.sumOf { it.voltageMv.toDouble() } / samples.size

    /** The mean of the current over the samples, in microamperes, with its sign. */
    fun meanCurrentUa(): Double = if (samples.isEmpty()) 0.0 else samples.sumOf { it.currentUa.toDouble() } / samples.size

    /**
     * The energy of the window in joules. With one sample the window has no
     * duration, thus the energy is 0.
     */
    fun energyJoules(): Double {
        var joules = 0.0
        for (i in 1 until samples.size) {
            val dt = (samples[i].elapsedMs - samples[i - 1].elapsedMs) / 1000.0
            joules += 0.5 * (samples[i].watts + samples[i - 1].watts) * dt
        }
        return joules
    }

    /**
     * The mean power of the window in watts. With no duration it is the mean
     * of the samples, thus a short window still gives a value.
     */
    fun meanPowerW(): Double {
        val ms = durationMs()
        if (ms > 0) {
            return energyJoules() / (ms / 1000.0)
        }
        return if (samples.isEmpty()) 0.0 else samples.sumOf { it.watts } / samples.size
    }

    /** The energy of one token in millijoules, 0 without a token. */
    fun energyPerTokenMj(tokens: Int): Double = if (tokens > 0) energyJoules() * 1000.0 / tokens else 0.0

    /**
     * The energy of one token in millijoules above the idle power of the
     * phone. The screen, the radio and the system draw [idleW] also without
     * an answer, thus this value is the cost of the answer itself.
     */
    fun netEnergyPerTokenMj(idleW: Double, tokens: Int): Double {
        if (tokens <= 0) {
            return 0.0
        }
        val net = (meanPowerW() - idleW) * (durationMs() / 1000.0)
        return net * 1000.0 / tokens
    }
}

/** The arithmetic that needs no window of samples. */
object Energy {
    /**
     * The energy that a difference of the charge counter gives, in joules.
     * The counter is a hardware value in microampere hours, thus it measures
     * a long window better than the samples of the current.
     */
    fun chargeEnergyJoules(deltaUah: Long, meanVoltageMv: Double): Double =
        abs(deltaUah) * 3.6e-3 * meanVoltageMv * 1e-3

    /** The energy of one token in millijoules for an energy in joules. */
    fun perTokenMj(joules: Double, tokens: Int): Double = if (tokens > 0) joules * 1000.0 / tokens else 0.0

    /** A value with one decimal, for the report. */
    fun format(value: Double, unit: String): String = String.format(Locale.US, "%.1f %s", value, unit)
}

/**
 * The battery values of the phone. Each call is a binder call, thus the
 * sampling runs off the display thread.
 */
class BatteryReader(context: Context) {
    private val app = context.applicationContext
    private val manager = app.getSystemService(BatteryManager::class.java)

    /** The instantaneous current in microamperes, or [PowerSample.INVALID]. */
    fun currentUa(): Long = property(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)

    /** The remaining charge in microampere hours, or [PowerSample.INVALID]. */
    fun chargeCounterUah(): Long = property(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)

    /** The battery voltage in millivolts, from the sticky intent of the battery, or 0. */
    fun voltageMv(): Int {
        val battery = app.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        return battery?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) ?: 0
    }

    /** One reading of the current and the voltage at this time. */
    fun sample(): PowerSample = PowerSample(SystemClock.elapsedRealtime(), currentUa(), voltageMv())

    /** True when the phone gives the current of the battery. */
    fun available(): Boolean = sample().valid

    private fun property(id: Int): Long {
        val value = manager?.getLongProperty(id) ?: return PowerSample.INVALID
        // The manager gives Long.MIN_VALUE or Integer.MIN_VALUE for a property that the phone does not support.
        return if (value == Long.MIN_VALUE || value == Int.MIN_VALUE.toLong()) PowerSample.INVALID else value
    }
}
