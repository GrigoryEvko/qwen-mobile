package ai.airi.qwenmobile

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import ai.airi.qwenmobile.databinding.FragmentChatBinding
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import java.io.File

/** The chat screen: model selection, backend, and the conversation. */
class ChatFragment : Fragment() {
    private var binding: FragmentChatBinding? = null
    private val messages = ArrayList<ChatMessage>()
    private val adapter = MessageAdapter(messages)
    private var models: List<File> = emptyList()
    private var generation: Job? = null

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        val b = FragmentChatBinding.inflate(inflater, container, false)
        binding = b
        b.messages.layoutManager = LinearLayoutManager(requireContext()).apply { stackFromEnd = true }
        b.messages.adapter = adapter
        b.backendGroup.check(if (LlamaEngine.hasGpu) b.gpuButton.id else b.cpuButton.id)
        b.gpuButton.isEnabled = LlamaEngine.hasGpu

        b.loadButton.setOnClickListener { onLoadOrUnload() }
        b.sendButton.setOnClickListener { onSend() }
        b.stopButton.setOnClickListener { generation?.cancel() }
        b.grantButton.setOnClickListener { startActivity(ModelFiles.allFilesAccessIntent(requireContext())) }
        b.clearButton.setOnClickListener { onClearChat() }

        viewLifecycleOwner.lifecycleScope.launch {
            LlamaEngine.state.collect { loaded ->
                b.loadButton.text = getString(if (loaded == null) R.string.load else R.string.unload)
                b.statusText.text = loaded?.info ?: getString(R.string.no_model)
                b.sendButton.isEnabled = loaded != null
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
        binding = null
    }

    /** Read the model directories again and fill the spinner. */
    private fun refreshModels() {
        val b = binding ?: return
        b.grantButton.visibility = if (ModelFiles.hasAllFilesAccess()) View.GONE else View.VISIBLE
        models = ModelFiles.list(requireContext())
        val names = models.map { "${it.name} (${it.length() shr 20} MB)" }
        b.modelSpinner.adapter = ArrayAdapter(requireContext(), android.R.layout.simple_spinner_dropdown_item, names)
    }

    private fun onLoadOrUnload() {
        val b = binding ?: return
        if (LlamaEngine.state.value != null) {
            generation?.cancel()
            viewLifecycleOwner.lifecycleScope.launch { LlamaEngine.unload() }
            return
        }
        val file = models.getOrNull(b.modelSpinner.selectedItemPosition)
        if (file == null) {
            toast(getString(R.string.no_model))
            return
        }
        val config = EngineConfig(
            path = file.absolutePath,
            gpu = b.backendGroup.checkedButtonId == b.gpuButton.id,
            threads = b.threadsInput.text.toString().toIntOrNull()?.coerceIn(1, 16) ?: 4,
        )
        b.loadButton.isEnabled = false
        b.statusText.text = "Loading ${file.name} on ${if (config.gpu) "GPU" else "CPU"}..."
        viewLifecycleOwner.lifecycleScope.launch {
            try {
                LlamaEngine.load(config)
                messages.clear()
                adapter.notifyDataSetChanged()
            } catch (e: Exception) {
                toast(e.message ?: "load failed")
                b.statusText.text = "Load failed: ${e.message}"
            } finally {
                b.loadButton.isEnabled = true
            }
        }
    }

    private fun onSend() {
        val b = binding ?: return
        val text = b.input.text.toString().trim()
        if (text.isEmpty() || generation?.isActive == true) {
            return
        }
        b.input.text?.clear()
        messages += ChatMessage("user", text)
        val history = messages.toList()
        val answer = ChatMessage("assistant", "")
        messages += answer
        adapter.notifyItemRangeInserted(messages.size - 2, 2)
        b.messages.scrollToPosition(messages.size - 1)
        setGenerating(true)

        generation = viewLifecycleOwner.lifecycleScope.launch {
            try {
                LlamaEngine.generate(history, b.thinkingSwitch.isChecked).collect { piece ->
                    answer.content += piece
                    adapter.notifyItemChanged(messages.size - 1)
                    b.messages.scrollToPosition(messages.size - 1)
                }
                b.statusText.text = LlamaEngine.stats()
            } catch (e: Exception) {
                if (e !is kotlinx.coroutines.CancellationException) {
                    answer.content += "\n[error: ${e.message}]"
                    adapter.notifyItemChanged(messages.size - 1)
                }
            } finally {
                setGenerating(false)
            }
        }
    }

    private fun onClearChat() {
        generation?.cancel()
        messages.clear()
        adapter.notifyDataSetChanged()
        viewLifecycleOwner.lifecycleScope.launch { LlamaEngine.resetChat() }
    }

    private fun setGenerating(active: Boolean) {
        val b = binding ?: return
        b.sendButton.visibility = if (active) View.GONE else View.VISIBLE
        b.stopButton.visibility = if (active) View.VISIBLE else View.GONE
        b.loadButton.isEnabled = !active
    }

    private fun toast(message: String) {
        Toast.makeText(requireContext(), message, Toast.LENGTH_LONG).show()
    }
}
