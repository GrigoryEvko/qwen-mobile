package ai.airi.qwenmobile

import android.os.Bundle
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
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * The debug screen: the llama-bench measurement, the ggml devices, the phone
 * state every 2 s while visible, the engine state, and the profiling export.
 */
class DebugFragment : Fragment() {
    private var binding: FragmentDebugBinding? = null
    private val stamp = SimpleDateFormat("HH:mm:ss", Locale.US)
    private var running = false

    /** The value views of the System card, in the order of [SYSTEM_LABELS]. */
    private val systemValues = ArrayList<TextView>()
    private lateinit var engineModel: TextView
    private lateinit var engineTurn: TextView

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        val b = FragmentDebugBinding.inflate(inflater, container, false)
        binding = b
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

    private fun refreshSystem() {
        val b = binding ?: return
        val s = SystemStatus.read(requireContext())
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
        systemValues[4].text = "${s.availMb} MB free of ${s.totalMb} MB"
        systemValues[5].text = "${s.gameMode}, ${s.adpf}"
        b.benchRun.isEnabled = LlamaEngine.state.value != null && !running
        b.profilingShare.isEnabled = profilingFile().exists()
    }

    /** The model line and the turn statistics. The statistics call waits behind a running generation. */
    private suspend fun refreshEngine() {
        val loaded = LlamaEngine.state.value
        engineModel.text = loaded?.info ?: getString(R.string.debug_no_engine)
        engineTurn.text = if (loaded == null) "" else runCatching { LlamaEngine.stats() }.getOrDefault("")
    }

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
        append("--- ${stamp.format(Date())} start, pp=$pp tg=$tg reps=$reps")
        append(SystemStatus.snapshot(requireContext()))
        viewLifecycleOwner.lifecycleScope.launch {
            try {
                append(LlamaEngine.bench(pp, tg, reps).trim())
            } catch (e: Exception) {
                append("bench failed: ${e.message}")
            } finally {
                append("--- ${stamp.format(Date())} end")
                append(SystemStatus.snapshot(requireContext()))
                running = false
                binding?.benchRun?.isEnabled = true
            }
        }
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
        const val PROFILING_CSV = "cl_profiling.csv"
        val SYSTEM_LABELS = listOf(
            R.string.debug_thermal, R.string.debug_battery, R.string.debug_charging,
            R.string.debug_cpu, R.string.debug_memory, R.string.debug_mode,
        )
    }
}
