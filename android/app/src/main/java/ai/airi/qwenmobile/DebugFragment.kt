package ai.airi.qwenmobile

import android.content.ClipData
import android.content.ClipboardManager
import android.os.Bundle
import android.os.SystemClock
import android.view.Gravity
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import ai.airi.qwenmobile.databinding.FragmentDebugBinding
import com.google.android.material.color.MaterialColors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.abs

/**
 * The debug screen: the llama-bench measurement with the energy of each
 * generated token, the ggml devices, the phone state every 2 s while
 * visible, the engine state, and the profiling export.
 */
class DebugFragment : Fragment() {
    private var binding: FragmentDebugBinding? = null
    private val stamp = SimpleDateFormat("HH:mm:ss", Locale.US)
    private var running = false

    /** The value views of the System card, in the order of [SYSTEM_LABELS]. */
    private val systemValues = ArrayList<TextView>()
    private lateinit var engineModel: TextView
    private lateinit var engineTurn: TextView
    private lateinit var battery: BatteryReader

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        val b = FragmentDebugBinding.inflate(inflater, container, false)
        binding = b
        battery = BatteryReader(requireContext())
        viewLifecycleOwner.lifecycleScope.launch {
            LlamaEngine.deviceList.collect {
                b.devicesList.removeAllViews()
                fillDevices(b.devicesList)
            }
        }
        for (label in SYSTEM_LABELS) {
            systemValues += addRow(b.systemRows, getString(label))
        }
        engineModel = addRow(b.engineRows, getString(R.string.debug_model))
        engineTurn = addRow(b.engineRows, getString(R.string.debug_turn))

        b.benchRun.setOnClickListener { onRun() }
        b.benchClear.setOnClickListener {
            b.benchResult.text = ""
            b.benchResult.visibility = View.GONE
        }
        b.benchCopy.setOnClickListener { onCopyReport() }
        b.profilingShare.setOnClickListener { onExportProfiling() }

        viewLifecycleOwner.lifecycleScope.launch {
            viewLifecycleOwner.repeatOnLifecycle(Lifecycle.State.RESUMED) {
                while (isActive) {
                    refreshSystem()
                    delay(REFRESH_MS)
                }
            }
        }
        viewLifecycleOwner.lifecycleScope.launch {
            viewLifecycleOwner.repeatOnLifecycle(Lifecycle.State.RESUMED) {
                while (isActive) {
                    refreshEngine()
                    delay(REFRESH_MS)
                }
            }
        }
        return b.root
    }

    override fun onDestroyView() {
        super.onDestroyView()
        systemValues.clear()
        binding = null
    }

    /** One row per ggml device. */
    private fun fillDevices(list: LinearLayout) {
        val lines = LlamaEngine.devices.lines().filter { it.isNotBlank() && !it.startsWith("opencl:") && !it.startsWith("extensions:") }
        if (lines.isEmpty()) {
            addRow(list, getString(R.string.debug_no_devices))
            return
        }
        for (line in lines) {
            val name = line.substringBefore(":")
            val detail = line.substringAfter(":").trim()
            addRow(list, name).text = detail
        }
    }

    /** A label on the left and a value on the right. Returns the value view. */
    private fun addRow(parent: LinearLayout, label: String): TextView {
        val context = requireContext()
        val row = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dp(6), 0, dp(6))
        }
        val labelView = TextView(context).apply {
            text = label
            setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_BodyMedium)
            setTextColor(MaterialColors.getColor(this, com.google.android.material.R.attr.colorOnSurfaceVariant))
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 2f)
        }
        val valueView = TextView(context).apply {
            setTextAppearance(com.google.android.material.R.style.TextAppearance_Material3_BodyMedium)
            gravity = Gravity.END
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 3f)
        }
        row.addView(labelView)
        row.addView(valueView)
        parent.addView(row)
        return valueView
    }

    /** The System card. The sysfs and proc files are read off the main thread. */
    private suspend fun refreshSystem() {
        val context = context ?: return
        val profile = profilingFile()
        val (s, hasProfile) = withContext(Dispatchers.IO) { SystemStatus.read(context) to profile.exists() }
        val b = binding ?: return
        val thermal = systemValues[0]
        thermal.text = if (s.cool) {
            "${s.thermalStatus} ${s.thermalName}, headroom ${s.headroom}"
        } else {
            getString(R.string.debug_thermal_warning, s.thermalStatus, s.thermalName)
        }
        val attr = if (s.cool) android.R.attr.colorPrimary else android.R.attr.colorError
        thermal.setTextColor(MaterialColors.getColor(thermal, attr))
        systemValues[1].text = "${s.batteryTemp} °C, ${s.batteryLevel} %"
        systemValues[2].text = if (s.charging) "yes, ${s.source}" else "no"
        systemValues[3].text = "cpu0 ${s.cpu0Cur} / ${s.cpu0Cap} MHz\ncpu7 ${s.cpu7Cur} / ${s.cpu7Cap} MHz"
        systemValues[4].text = s.powerText()
        systemValues[5].text = "${s.availMb} MB free of ${s.totalMb} MB"
        systemValues[6].text = "${s.gameMode}, ${s.adpf}"
        b.benchRun.isEnabled = LlamaEngine.state.value != null && !running && !ChatSession.generating.value
        b.profilingShare.isEnabled = hasProfile
    }

    /**
     * The model line and the turn statistics. The statistics call queues
     * behind a streamed answer on the engine thread, thus the row keeps its
     * last value while an answer streams.
     */
    private suspend fun refreshEngine() {
        val loaded = LlamaEngine.state.value
        engineModel.text = loaded?.info ?: getString(R.string.debug_no_engine)
        when {
            loaded == null -> engineTurn.text = ""
            ChatSession.generating.value -> return
            else -> engineTurn.text = runCatching { LlamaEngine.stats() }.getOrDefault("")
        }
    }

    /**
     * Run the benchmark and measure the energy of its decode.
     *
     * The prompt and the answer run as two calls, thus the samples of the
     * battery current belong to one of them. An idle window before the run
     * gives the power that the phone draws without an answer, and the
     * charge counter of the battery gives a second value of the energy.
     */
    private fun onRun() {
        val b = binding ?: return
        if (LlamaEngine.state.value == null) {
            append(getString(R.string.debug_no_model))
            return
        }
        val pp = b.benchPp.text.toString().toIntOrNull() ?: 512
        val tg = b.benchTg.text.toString().toIntOrNull() ?: 128
        val reps = b.benchReps.text.toString().toIntOrNull()?.coerceIn(1, 20) ?: 3
        running = true
        b.benchRun.isEnabled = false
        val context = requireContext()
        val thermalStart = SystemStatus.thermalStatus(context)
        append("--- ${stamp.format(Date())} start, pp=$pp tg=$tg reps=$reps")
        append(SystemStatus.snapshot(context))
        viewLifecycleOwner.lifecycleScope.launch {
            try {
                // The phone draws power also without an answer. This window measures it.
                val idle = sampleWindow(IDLE_MS)
                val prompt = PowerTrace()
                val ppOut = if (pp > 0) measure(prompt) { LlamaEngine.bench(pp, 0, reps).trim() } else ""
                if (ppOut.isNotEmpty()) {
                    append(ppOut)
                }
                val decode = PowerTrace()
                val charge0 = battery.chargeCounterUah()
                val tgOut = if (tg > 0) measure(decode) { LlamaEngine.bench(0, tg, reps).trim() } else ""
                val charge1 = battery.chargeCounterUah()
                // The two calls give the same first line, which names the model.
                val header = ppOut.lineSequence().firstOrNull()
                val tgBody = if (header != null && tgOut.startsWith(header)) tgOut.removePrefix(header).trim() else tgOut
                if (tgBody.isNotEmpty()) {
                    append(tgBody)
                }
                appendEnergy(idle, prompt, decode, charge0, charge1, tg * reps)
            } catch (e: Exception) {
                append("bench failed: ${e.message}")
            } finally {
                append("thermal: $thermalStart at the start, ${SystemStatus.thermalStatus(context)} at the end")
                append("--- ${stamp.format(Date())} end")
                append(SystemStatus.snapshot(context))
                running = false
                binding?.benchRun?.isEnabled = true
            }
        }
    }

    /**
     * Run [work] and fill [trace] with the battery samples of its window, at
     * [SAMPLE_MS] intervals. The samples come from a thread that is not the
     * display thread, and the work runs on the engine thread.
     */
    private suspend fun <T> measure(trace: PowerTrace, work: suspend () -> T): T {
        val sampler = viewLifecycleOwner.lifecycleScope.launch(Dispatchers.Default) {
            while (isActive) {
                trace.add(battery.sample())
                delay(SAMPLE_MS)
            }
        }
        try {
            return work()
        } finally {
            // The join gives the samples to the thread that reads them.
            sampler.cancelAndJoin()
        }
    }

    /** Sample the battery for a duration, off the display thread. */
    private suspend fun sampleWindow(durationMs: Long): PowerTrace = withContext(Dispatchers.Default) {
        val trace = PowerTrace()
        val end = SystemClock.elapsedRealtime() + durationMs
        while (SystemClock.elapsedRealtime() < end) {
            trace.add(battery.sample())
            delay(SAMPLE_MS)
        }
        trace
    }

    /** The rows of the energy: the current, the power, and the energy of one generated token. */
    private fun appendEnergy(
        idle: PowerTrace,
        prompt: PowerTrace,
        decode: PowerTrace,
        charge0: Long,
        charge1: Long,
        tokens: Int,
    ) {
        if (decode.empty) {
            append("energy: the phone gives no battery current")
            return
        }
        val direction = if (decode.meanCurrentUa() < 0) "discharge" else "charge"
        append(
            String.format(
                Locale.US,
                "battery: %.0f mA %s at %.0f mV, %d samples in %.1f s",
                abs(decode.meanCurrentUa()) / 1000.0, direction, decode.meanVoltageMv(),
                decode.size, decode.durationMs() / 1000.0,
            ),
        )
        val idleW = idle.meanPowerW()
        append(
            String.format(
                Locale.US, "power: idle %.2f W, prompt %.2f W, decode %.2f W, decode above idle %.2f W",
                idleW, prompt.meanPowerW(), decode.meanPowerW(), decode.meanPowerW() - idleW,
            ),
        )
        var line = String.format(
            Locale.US, "energy: %d tok in %.1f s, %.0f mJ/token, %.0f mJ/token above idle",
            tokens, decode.durationMs() / 1000.0, decode.energyPerTokenMj(tokens),
            decode.netEnergyPerTokenMj(idleW, tokens),
        )
        // The charge counter is a hardware value, thus it measures a long window better than the samples.
        if (charge0 != PowerSample.INVALID && charge1 != PowerSample.INVALID && charge0 != charge1) {
            val joules = Energy.chargeEnergyJoules(charge1 - charge0, decode.meanVoltageMv())
            line += String.format(
                Locale.US, ", %.0f mJ/token by the charge counter (%d uAh)",
                Energy.perTokenMj(joules, tokens), abs(charge1 - charge0),
            )
        }
        append(line)
        if (direction == "charge") {
            append("energy: the phone is on a charger, thus these values are not the energy of the answer")
        }
    }

    /** Put the whole report on the clipboard. */
    private fun onCopyReport() {
        val text = binding?.benchResult?.text?.toString().orEmpty()
        if (text.isBlank()) {
            return
        }
        val clipboard = requireContext().getSystemService(ClipboardManager::class.java)
        clipboard?.setPrimaryClip(ClipData.newPlainText(getString(R.string.debug_benchmark), text))
        Toast.makeText(requireContext(), getString(R.string.debug_copied), Toast.LENGTH_SHORT).show()
    }

    /** Copy cl_profiling.csv to the external files directory, where adb pull reaches it. */
    private fun onExportProfiling() {
        val source = profilingFile()
        val dir = requireContext().getExternalFilesDir(null)
        try {
            val target = File(dir, source.name)
            source.copyTo(target, overwrite = true)
            Toast.makeText(requireContext(), getString(R.string.debug_exported, target.absolutePath), Toast.LENGTH_LONG).show()
        } catch (e: Exception) {
            Toast.makeText(requireContext(), getString(R.string.debug_export_failed, e.message), Toast.LENGTH_LONG).show()
        }
    }

    private fun profilingFile(): File = File(requireContext().filesDir, PROFILING_CSV)

    private fun append(text: String) {
        val b = binding ?: return
        b.benchResult.visibility = View.VISIBLE
        b.benchResult.append(text + "\n")
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()

    private companion object {
        const val REFRESH_MS = 2000L

        /** The interval between two battery samples: 4 Hz. */
        const val SAMPLE_MS = 250L

        /** The duration of the window that measures the power of the phone without an answer. */
        const val IDLE_MS = 1500L

        const val PROFILING_CSV = "cl_profiling.csv"
        val SYSTEM_LABELS = listOf(
            R.string.debug_thermal, R.string.debug_battery, R.string.debug_charging,
            R.string.debug_cpu, R.string.debug_power, R.string.debug_memory, R.string.debug_mode,
        )
    }
}
