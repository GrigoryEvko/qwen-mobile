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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
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
                applySpeculative(b, loaded)
            }
        }
        viewLifecycleOwner.lifecycleScope.launch {
            LlamaEngine.deviceList.collect {
                enableBackendChips(b)
                b.deviceText.text = firstDeviceLine()
            }
        }
        return b.root
    }

    /** The OpenCL device line, or the text for no device. */
    private fun firstDeviceLine(): String =
        LlamaEngine.devices.lineSequence().firstOrNull { it.isNotBlank() } ?: getString(R.string.settings_no_devices)

    /** The chips of the visible devices. All but the CPU wait for the backends. */
    private fun enableBackendChips(b: FragmentSettingsBinding) {
        b.cpuChip.isEnabled = LlamaEngine.has(Backend.CPU)
        b.gpuChip.isEnabled = LlamaEngine.has(Backend.GPU)
        b.npuChip.isEnabled = LlamaEngine.has(Backend.NPU)
        b.hybridChip.isEnabled = LlamaEngine.has(Backend.HYBRID)
        b.visionGpuSwitch.isEnabled = LlamaEngine.has(Backend.GPU)
    }

    override fun onResume() {
        super.onResume()
        refreshModels()
    }

    override fun onPause() {
        binding?.let { saveSystemPrompt(it) }
        super.onPause()
    }

    override fun onDestroyView() {
        binding?.let { saveSystemPrompt(it) }
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

    /** Read the model directories again, off the main thread, and rebuild the list. */
    private fun refreshModels() {
        val b = binding ?: return
        b.grantButton.visibility = if (ModelFiles.hasAllFilesAccess()) View.GONE else View.VISIBLE
        val context = requireContext()
        viewLifecycleOwner.lifecycleScope.launch {
            // The shared directory goes through FUSE, thus the listing and the sizes stay off the main thread.
            val models = withContext(Dispatchers.IO) { ModelFiles.list(context).map { it to it.length() } }
            val list = binding ?: return@launch
            list.modelList.removeAllViews()
            modelRows.clear()
            list.noModelsText.visibility = if (models.isEmpty()) View.VISIBLE else View.GONE
            for ((file, size) in models) {
                val row = ItemModelBinding.inflate(layoutInflater, list.modelList, true)
                row.nameText.text = ModelFiles.displayName(file)
                row.sizeText.text = formatSize(size)
                val select = View.OnClickListener { selectModel(file) }
                row.root.setOnClickListener(select)
                row.radio.setOnClickListener(select)
                modelRows[file.absolutePath] = row
            }
            applyModel(store.state.value.modelPath)
        }
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
        enableBackendChips(b)
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
        b.visionGpuSwitch.setOnCheckedChangeListener { _, checked ->
            if (applying) return@setOnCheckedChangeListener
            store.update { it.copy(visionOnGpu = checked) }
        }
        for (detail in ImageDetail.entries) {
            val chip = layoutInflater.inflate(R.layout.item_choice_chip, b.imageDetailGroup, false) as Chip
            chip.id = View.generateViewId()
            chip.text = getString(R.string.settings_image_detail_chip, getString(detailLabel(detail)), detail.tokens)
            chip.tag = detail
            b.imageDetailGroup.addView(chip)
        }
        b.imageDetailGroup.setOnCheckedStateChangeListener { group, checkedIds ->
            if (applying) return@setOnCheckedStateChangeListener
            val chip = checkedIds.firstOrNull()?.let { group.findViewById<Chip>(it) } ?: return@setOnCheckedStateChangeListener
            store.update { it.copy(imageDetail = chip.tag as ImageDetail) }
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

    /**
     * The speculative switch is available only for a loaded model that holds
     * the MTP tensors and runs on a compute unit that can draft with them.
     * The hint tells the user which of the three conditions is missing.
     */
    private fun applySpeculative(b: FragmentSettingsBinding, loaded: LoadedModel?) {
        b.speculativeSwitch.isEnabled = loaded?.hasMtp == true
        b.speculativeHint.text = when {
            loaded == null -> getString(R.string.settings_speculative_no_model)
            !loaded.hasMtp -> getString(R.string.settings_speculative_absent)
            else -> getString(R.string.settings_speculative_hint)
        }
    }

    /** The label of an image detail level. */
    private fun detailLabel(detail: ImageDetail): Int = when (detail) {
        ImageDetail.FAST -> R.string.settings_image_detail_fast
        ImageDetail.STANDARD -> R.string.settings_image_detail_standard
        ImageDetail.DETAILED -> R.string.settings_image_detail_detailed
    }

    // --- Generation ---

    private fun setupGeneration(b: FragmentSettingsBinding) {
        b.thinkingSwitch.setOnCheckedChangeListener { _, checked ->
            if (applying) return@setOnCheckedChangeListener
            store.update { it.copy(thinking = checked) }
        }
        b.speculativeSwitch.setOnCheckedChangeListener { _, checked ->
            if (applying) return@setOnCheckedChangeListener
            store.update { it.copy(speculative = checked) }
        }
        applySpeculative(b, LlamaEngine.state.value)
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
        // The prompt is saved when the field loses the focus and when the
        // screen stops, not on every keystroke: a write per character would
        // touch the disk and rebuild the prompt of the next turn each time.
        b.systemPromptInput.setOnFocusChangeListener { _, focused ->
            if (!focused) saveSystemPrompt(b)
        }
    }

    /** Copy the system prompt of the field into the store, when it changed. */
    private fun saveSystemPrompt(b: FragmentSettingsBinding) {
        if (applying) return
        val text = b.systemPromptInput.text?.toString().orEmpty().trim()
        if (text != store.state.value.systemPrompt) {
            store.update { it.copy(systemPrompt = text) }
        }
    }

    // --- About ---

    private fun setupAbout(b: FragmentSettingsBinding) {
        val context = requireContext()
        val version = context.packageManager.getPackageInfo(context.packageName, 0).versionName ?: "?"
        b.versionText.text = getString(R.string.settings_version, version)
        b.deviceText.text = firstDeviceLine()
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
            b.visionGpuSwitch.isChecked = s.visionOnGpu
            for (i in 0 until b.imageDetailGroup.childCount) {
                val chip = b.imageDetailGroup.getChildAt(i) as Chip
                if (chip.tag == s.imageDetail) {
                    b.imageDetailGroup.check(chip.id)
                }
            }
            b.threadsSlider.value = s.threads.toFloat()
            b.threadsValue.text = s.threads.toString()
            for (i in 0 until b.contextGroup.childCount) {
                val chip = b.contextGroup.getChildAt(i) as Chip
                if (chip.tag == s.nCtx) {
                    b.contextGroup.check(chip.id)
                }
            }
            b.thinkingSwitch.isChecked = s.thinking
            b.speculativeSwitch.isChecked = s.speculative
            b.temperatureSlider.value = snap(s.temperature, 0f, 0.05f)
            b.temperatureValue.text = String.format(Locale.US, "%.2f", s.temperature)
            b.topPSlider.value = snap(s.topP, SettingsStore.MIN_TOP_P, 0.01f)
            b.topPValue.text = String.format(Locale.US, "%.2f", s.topP)
            // A running edit wins over the store, thus typing is never cut off.
            if (!b.systemPromptInput.hasFocus() &&
                b.systemPromptInput.text?.toString() != s.systemPrompt
            ) {
                b.systemPromptInput.setText(s.systemPrompt)
            }
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
