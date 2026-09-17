package ai.airi.qwenmobile

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

/**
 * One message of the conversation. The role is "user", "assistant", or "system".
 * An image is the encoded file (JPEG) that goes to the vision projector.
 */
class ChatMessage(val role: String, var content: String, val image: ByteArray? = null) {
    /** The decoded image for the message list, decoded one time. */
    val bitmap: Bitmap? by lazy { image?.let { BitmapFactory.decodeByteArray(it, 0, it.size) } }
}

/** The compute unit of the model. The ggml device name is null for the CPU. */
enum class Backend(val deviceName: String?, val label: String) {
    CPU(null, "CPU"),
    GPU("GPUOpenCL", "GPU"),
    NPU("HTP0", "NPU"),
}

/** The settings of one loaded model. */
data class EngineConfig(
    val path: String,
    val backend: Backend,
    val threads: Int,
    val nCtx: Int = 8192,
    val mmproj: String? = null,
)

/** A loaded model with the description line from the native side. */
data class LoadedModel(val config: EngineConfig, val info: String)

/**
 * The single owner of the native engine. Every native call runs on one
 * dedicated thread, thus the ADPF session and the mutex see one caller.
 */
object LlamaEngine {
    /** The maximum number of tokens in one answer. */
    private const val MAX_ANSWER_TOKENS = 4096

    private val dispatcher = Executors.newSingleThreadExecutor { runnable ->
        Thread(runnable, "llama-engine")
    }.asCoroutineDispatcher()

    private var handle = 0L

    private val stateFlow = MutableStateFlow<LoadedModel?>(null)

    /** The loaded model, or null. */
    val state: StateFlow<LoadedModel?> = stateFlow

    /** The ggml backend devices, one per line. */
    val devices: String by lazy { LlamaNative.devices() }

    /** Tell if the device of a backend is visible. */
    fun has(backend: Backend): Boolean =
        backend.deviceName == null || devices.lines().any { it.startsWith(backend.deviceName + ":") }

    /** Load a model. A model that is loaded already is released first. */
    suspend fun load(config: EngineConfig): LoadedModel = withContext(dispatcher) {
        releaseLocked()
        handle = LlamaNative.load(
            config.path,
            config.mmproj,
            config.backend.deviceName,
            if (config.backend == Backend.CPU) 0 else 999,
            config.threads,
            config.nCtx,
        )
        val loaded = LoadedModel(config, LlamaNative.modelInfo(handle))
        stateFlow.value = loaded
        loaded
    }

    /** Release the loaded model. */
    suspend fun unload() = withContext(dispatcher) {
        releaseLocked()
    }

    /**
     * Generate the answer to the conversation. Each emitted string is a
     * piece of the answer. The flow stops at the end token, at the token
     * limit, or when the collector cancels.
     */
    fun generate(messages: List<ChatMessage>, thinking: Boolean): Flow<String> = flow {
        val h = requireHandle()
        LlamaNative.chatStart(
            h,
            messages.map { it.role }.toTypedArray(),
            messages.map { it.content }.toTypedArray(),
            messages.map { it.image }.toTypedArray(),
            thinking,
        )
        var count = 0
        while (count < MAX_ANSWER_TOKENS) {
            val bytes = LlamaNative.generateNext(h) ?: break
            count += 1
            if (bytes.isNotEmpty()) {
                emit(String(bytes, Charsets.UTF_8))
            }
        }
    }.flowOn(dispatcher)

    /** The speed line of the current turn. */
    suspend fun stats(): String = withContext(dispatcher) {
        LlamaNative.stats(requireHandle())
    }

    /** Clear the model memory. The next turn starts from an empty state. */
    suspend fun resetChat() = withContext(dispatcher) {
        if (handle != 0L) {
            LlamaNative.resetChat(handle)
        }
    }

    /** Run the benchmark on the loaded model. */
    suspend fun bench(pp: Int, tg: Int, reps: Int): String = withContext(dispatcher) {
        LlamaNative.bench(requireHandle(), pp, tg, reps)
    }

    private fun requireHandle(): Long {
        check(handle != 0L) { "No model is loaded" }
        return handle
    }

    private fun releaseLocked() {
        if (handle != 0L) {
            LlamaNative.free(handle)
            handle = 0L
        }
        stateFlow.value = null
    }
}
