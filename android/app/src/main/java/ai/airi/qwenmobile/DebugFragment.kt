package ai.airi.qwenmobile

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.lifecycleScope
import androidx.lifecycle.repeatOnLifecycle
import ai.airi.qwenmobile.databinding.FragmentDebugBinding
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** The debug screen: the phone state each second, and the llama-bench measurement. */
class DebugFragment : Fragment() {
    private var binding: FragmentDebugBinding? = null
    private val stamp = SimpleDateFormat("HH:mm:ss", Locale.US)

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        val b = FragmentDebugBinding.inflate(inflater, container, false)
        binding = b
        b.devicesText.text = LlamaEngine.devices.trim()
        b.runButton.setOnClickListener { onRun() }
        b.clearButton.setOnClickListener { b.resultsText.text = "" }

        viewLifecycleOwner.lifecycleScope.launch {
            viewLifecycleOwner.repeatOnLifecycle(Lifecycle.State.RESUMED) {
                while (isActive) {
                    b.statusText.text = SystemStatus.snapshot(requireContext())
                    b.runButton.isEnabled = LlamaEngine.state.value != null && !running
                    delay(1000)
                }
            }
        }
        return b.root
    }

    override fun onDestroyView() {
        super.onDestroyView()
        binding = null
    }

    private var running = false

    private fun onRun() {
        val b = binding ?: return
        val loaded = LlamaEngine.state.value
        if (loaded == null) {
            append("No model is loaded. Load one on the Chat tab.")
            return
        }
        val pp = b.ppInput.text.toString().toIntOrNull() ?: 512
        val tg = b.tgInput.text.toString().toIntOrNull() ?: 128
        val reps = b.repsInput.text.toString().toIntOrNull()?.coerceIn(1, 20) ?: 3
        running = true
        b.runButton.isEnabled = false
        append("--- ${stamp.format(Date())} start, pp=$pp tg=$tg reps=$reps")
        append(SystemStatus.snapshot(requireContext()))
        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val result = LlamaEngine.bench(pp, tg, reps)
                append(result.trim())
            } catch (e: Exception) {
                append("bench failed: ${e.message}")
            } finally {
                append("--- ${stamp.format(Date())} end")
                append(SystemStatus.snapshot(requireContext()))
                running = false
                b.runButton.isEnabled = true
            }
        }
    }

    private fun append(text: String) {
        val b = binding ?: return
        b.resultsText.append(text + "\n")
        b.resultsScroll.post { b.resultsScroll.fullScroll(View.FOCUS_DOWN) }
    }
}
