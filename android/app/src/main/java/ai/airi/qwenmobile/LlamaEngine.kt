package ai.airi.qwenmobile

import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

/** One message of the conversation. The role is "user", "assistant", or "system". */
data class ChatMessage(val role: String, var content: String)

/** The settings of one loaded model. */
data class EngineConfig(
    val path: String,
    val gpu: Boolean,
    val threads: Int,
    val nCtx: Int = 8192,
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

    /** Tell if a GPU backend is compiled in and visible. */
    val hasGpu: Boolean by lazy { devices.contains("(GPU)") }

    /** Load a model. A model that is loaded already is released first. */
    suspend fun load(config: EngineConfig): LoadedModel = withContext(dispatcher) {
        releaseLocked()
        handle = LlamaNative.load(
            config.path,
            if (config.gpu) 999 else 0,
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
