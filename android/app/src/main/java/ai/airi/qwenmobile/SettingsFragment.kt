package ai.airi.qwenmobile

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import ai.airi.qwenmobile.databinding.FragmentSettingsBinding
import ai.airi.qwenmobile.databinding.ItemModelBinding
import com.google.android.material.chip.Chip
import kotlinx.coroutines.launch
import java.io.File
import java.util.Locale

/**
 * The settings screen: model, compute unit, generation, about.
 * Every change goes to the [SettingsStore] at once, and the screen follows the store.
 */
class SettingsFragment : Fragment() {
    private var binding: FragmentSettingsBinding? = null
    private lateinit var store: SettingsStore

    /** The rows of the model list, by file path. */
    private val modelRows = LinkedHashMap<String, ItemModelBinding>()

    /** True while the screen copies the store into the views, thus the listeners stay quiet. */
    private var applying = false

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        val b = FragmentSettingsBinding.inflate(inflater, container, false)
        binding = b
        store = SettingsStore.of(requireContext())
        setupModel(b)
        setupCompute(b)
        setupGeneration(b)
        setupAbout(b)
        viewLifecycleOwner.lifecycleScope.launch {
            store.state.collect { apply(it) }
        }
        viewLifecycleOwner.lifecycleScope.launch {
            LlamaEngine.state.collect { loaded ->
                b.modelInfoText.text = loaded?.info ?: getString(R.string.settings_no_model_loaded)
            }
        }
        return b.root
    }

    override fun onResume() {
        super.onResume()
        refreshModels()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        modelRows.clear()
        binding = null
    }

    // --- Model ---

    private fun setupModel(b: FragmentSettingsBinding) {
        b.modelHint.text = getString(R.string.settings_model_hint, ModelFiles.publicDir().absolutePath)
        b.grantButton.setOnClickListener { startActivity(ModelFiles.allFilesAccessIntent(requireContext())) }
        b.refreshButton.setOnClickListener { refreshModels() }
    }

    /** Read the model directories again and rebuild the list. */
    private fun refreshModels() {
        val b = binding ?: return
        b.grantButton.visibility = if (ModelFiles.hasAllFilesAccess()) View.GONE else View.VISIBLE
        val models = ModelFiles.list(requireContext())
        b.modelList.removeAllViews()
        modelRows.clear()
        b.noModelsText.visibility = if (models.isEmpty()) View.VISIBLE else View.GONE
        for (file in models) {
            val row = ItemModelBinding.inflate(layoutInflater, b.modelList, true)
            row.nameText.text = file.name
            row.sizeText.text = formatSize(file.length())
            val select = View.OnClickListener { selectModel(file) }
            row.root.setOnClickListener(select)
            row.radio.setOnClickListener(select)
            modelRows[file.absolutePath] = row
        }
        applyModel(store.state.value.modelPath)
    }

    private fun selectModel(file: File) {
        store.update { it.copy(modelPath = file.absolutePath) }
    }

    private fun applyModel(path: String?) {
        for ((rowPath, row) in modelRows) {
            row.radio.isChecked = rowPath == path
        }
    }

    // --- Compute ---

    private fun setupCompute(b: FragmentSettingsBinding) {
        b.cpuChip.isEnabled = LlamaEngine.has(Backend.CPU)
        b.gpuChip.isEnabled = LlamaEngine.has(Backend.GPU)
        b.npuChip.isEnabled = LlamaEngine.has(Backend.NPU)
        b.hybridChip.isEnabled = LlamaEngine.has(Backend.HYBRID)
        b.backendGroup.setOnCheckedStateChangeListener { _, checkedIds ->
            if (applying) return@setOnCheckedStateChangeListener
            val backend = when (checkedIds.firstOrNull()) {
                b.gpuChip.id -> Backend.GPU
                b.npuChip.id -> Backend.NPU
                b.hybridChip.id -> Backend.HYBRID
                else -> Backend.CPU
            }
            store.update { it.copy(backend = backend) }
        }
        b.threadsSlider.valueFrom = SettingsStore.MIN_THREADS.toFloat()
        b.threadsSlider.valueTo = SettingsStore.MAX_THREADS.toFloat()
        b.threadsSlider.stepSize = 1f
        b.threadsSlider.addOnChangeListener { _, value, fromUser ->
            b.threadsValue.text = value.toInt().toString()
            if (fromUser) {
                store.update { it.copy(threads = value.toInt()) }
            }
        }
        for (length in SettingsStore.CONTEXT_LENGTHS) {
            val chip = layoutInflater.inflate(R.layout.item_choice_chip, b.contextGroup, false) as Chip
            chip.id = View.generateViewId()
            chip.text = length.toString()
            chip.tag = length
            b.contextGroup.addView(chip)
        }
        b.contextGroup.setOnCheckedStateChangeListener { group, checkedIds ->
            if (applying) return@setOnCheckedStateChangeListener
            val chip = checkedIds.firstOrNull()?.let { group.findViewById<Chip>(it) } ?: return@setOnCheckedStateChangeListener
            store.update { it.copy(nCtx = chip.tag as Int) }
        }
    }

    // --- Generation ---

    private fun setupGeneration(b: FragmentSettingsBinding) {
        b.thinkingSwitch.setOnCheckedChangeListener { _, checked ->
            if (applying) return@setOnCheckedChangeListener
            store.update { it.copy(thinking = checked) }
        }
        b.temperatureSlider.valueFrom = 0f
        b.temperatureSlider.valueTo = SettingsStore.MAX_TEMPERATURE
        b.temperatureSlider.stepSize = 0.05f
        b.temperatureSlider.addOnChangeListener { _, value, fromUser ->
            b.temperatureValue.text = String.format(Locale.US, "%.2f", value)
            if (fromUser) {
                store.update { it.copy(temperature = value) }
            }
        }
        b.topPSlider.valueFrom = SettingsStore.MIN_TOP_P
        b.topPSlider.valueTo = 1f
        b.topPSlider.stepSize = 0.01f
        b.topPSlider.addOnChangeListener { _, value, fromUser ->
            b.topPValue.text = String.format(Locale.US, "%.2f", value)
            if (fromUser) {
                store.update { it.copy(topP = value) }
            }
        }
    }

    // --- About ---

    private fun setupAbout(b: FragmentSettingsBinding) {
        val context = requireContext()
        val version = context.packageManager.getPackageInfo(context.packageName, 0).versionName ?: "?"
        b.versionText.text = getString(R.string.settings_version, version)
        b.deviceText.text = LlamaEngine.devices.lineSequence().firstOrNull { it.isNotBlank() }
            ?: getString(R.string.settings_no_devices)
    }

    // --- Store to views ---

    private fun apply(s: AppSettings) {
        val b = binding ?: return
        applying = true
        try {
            applyModel(s.modelPath)
            b.backendGroup.check(
                when (s.backend) {
                    Backend.CPU -> b.cpuChip.id
                    Backend.GPU -> b.gpuChip.id
                    Backend.NPU -> b.npuChip.id
                    Backend.HYBRID -> b.hybridChip.id
                },
            )
            b.threadsSlider.value = s.threads.toFloat()
            b.threadsValue.text = s.threads.toString()
            for (i in 0 until b.contextGroup.childCount) {
                val chip = b.contextGroup.getChildAt(i) as Chip
                if (chip.tag == s.nCtx) {
                    b.contextGroup.check(chip.id)
                }
            }
            b.thinkingSwitch.isChecked = s.thinking
            b.temperatureSlider.value = snap(s.temperature, 0f, 0.05f)
            b.temperatureValue.text = String.format(Locale.US, "%.2f", s.temperature)
            b.topPSlider.value = snap(s.topP, SettingsStore.MIN_TOP_P, 0.01f)
            b.topPValue.text = String.format(Locale.US, "%.2f", s.topP)
        } finally {
            applying = false
        }
    }

    /** The nearest slider stop. The slider rejects a value between two stops. */
    private fun snap(value: Float, from: Float, step: Float): Float =
        from + Math.round((value - from) / step) * step

    private fun formatSize(bytes: Long): String {
        val mib = bytes / 1048576.0
        return if (mib >= 1024) String.format(Locale.US, "%.2f GB", mib / 1024) else String.format(Locale.US, "%.0f MB", mib)
    }
}
