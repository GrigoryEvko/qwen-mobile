package ai.airi.qwenmobile

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.filterNotNull
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

/**
 * One message of the conversation. The role is "user", "assistant", or "system".
 * An image is the encoded file (JPEG) that goes to the vision projector.
 *
 * A streamed answer grows by [appendContent] and [appendThinking], which
 * cost O(piece). The first read of [content] or [thinking] after a change
 * copies the text one time, O(length), and the next reads are O(1).
 * Read and write the text on the main thread only.
 */
class ChatMessage(val role: String, content: String, val image: ByteArray? = null) {
    /**
     * The phase of an answer. A user message is always DONE. INTERRUPTED is
     * a stop before any answer text, FAILED is an error before any answer
     * text. The two are not saved and the model does not see them.
     */
    enum class Phase { THINKING, ANSWERING, DONE, INTERRUPTED, FAILED }

    var phase: Phase = Phase.DONE

    private val contentText = StringBuilder(content)
    private var contentCache: String? = content

    /** The text of the message. */
    var content: String
        get() = contentCache ?: contentText.toString().also { contentCache = it }
        set(value) {
            contentText.setLength(0)
            contentText.append(value)
            contentCache = value
        }

    /** True when [content] has text. O(1). */
    val hasContent: Boolean get() = contentText.isNotEmpty()

    private val thinkingText = StringBuilder()
    private var thinkingCache: String? = ""

    /** The thinking of an answer, empty without one. It is not part of [content]. */
    var thinking: String
        get() = thinkingCache ?: thinkingText.toString().also { thinkingCache = it }
        set(value) {
            thinkingText.setLength(0)
            thinkingText.append(value)
            thinkingCache = value
        }

    /** True when [thinking] has text. O(1). */
    val hasThinking: Boolean get() = thinkingText.isNotEmpty()

    /** Add a piece of the streamed answer to [content]. */
    fun appendContent(piece: CharSequence) {
        contentText.append(piece)
        contentCache = null
    }

    /** Add a piece of the streamed thinking to [thinking]. */
    fun appendThinking(piece: CharSequence) {
        thinkingText.append(piece)
        thinkingCache = null
    }

    /** The duration of the thinking, 0 without one. */
    var thinkingMs: Long = 0

    /** The choice of the user on the thinking field, null for the default of the phase. */
    var thinkingExpanded: Boolean? = null

    /** The decoded image for the message list, decoded one time. */
    val bitmap: Bitmap? by lazy { image?.let { BitmapFactory.decodeByteArray(it, 0, it.size) } }
}

/** One step of a streamed answer. */
sealed class Piece {
    class Text(val text: String) : Piece()
    object ThinkOpen : Piece()
    object ThinkClose : Piece()
}

/**
 * The compute unit of the model. The ggml device name is null for the CPU.
 * The GPU is the default, the NPU is opt-in: its single-token decode is
 * experimental. The hybrid one prefills on the NPU and decodes on the GPU.
 */
enum class Backend(val deviceName: String?, val prefillDeviceName: String?, val label: String) {
    CPU(null, null, "CPU"),
    GPU("GPUOpenCL", null, "GPU"),
    NPU("HTP0", null, "NPU"),
    HYBRID("GPUOpenCL", "HTP0", "NPU+GPU");

    /** The ggml devices the backend needs. */
    val devices: List<String> get() = listOfNotNull(deviceName, prefillDeviceName)

    companion object {
        /** The backend of a fresh install. */
        val DEFAULT = GPU
    }
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
        // The engine thread runs at the background priority, thus the display thread stays smooth.
        Thread({
            android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_BACKGROUND)
            runnable.run()
        }, "llama-engine")
    }.asCoroutineDispatcher()

    private val engineScope = CoroutineScope(dispatcher + SupervisorJob())

    private var handle = 0L

    private val stateFlow = MutableStateFlow<LoadedModel?>(null)

    /** The loaded model, or null. */
    val state: StateFlow<LoadedModel?> = stateFlow

    private val devicesFlow = MutableStateFlow<String?>(null)

    /**
     * The ggml backend devices, one per line, null until the backends are
     * initialized. The initialization opens the NPU session, which can take
     * seconds on a hot phone, thus it never runs on the display thread.
     */
    val deviceList: StateFlow<String?> = devicesFlow

    /** The device list, or an empty text before the backends are initialized. */
    val devices: String get() = devicesFlow.value ?: ""

    /** True when the backends are initialized and the device list is known. */
    val ready: Boolean get() = devicesFlow.value != null

    /**
     * Initialize the backends on the engine thread. Every native call that
     * follows queues behind it on the same thread. Call one time from the
     * application.
     */
    fun start(libDir: String, workDir: String) {
        engineScope.launch {
            LlamaNative.initialize(libDir)
            LlamaNative.setWorkingDirectory(workDir)
            devicesFlow.value = LlamaNative.devices()
        }
    }

    /** Suspend until the backends are initialized. */
    suspend fun awaitReady() {
        devicesFlow.filterNotNull().first()
    }

    /** Tell if every device of a backend is visible. False before the backends are initialized. */
    fun has(backend: Backend): Boolean =
        ready && backend.devices.all { name -> devices.lines().any { it.startsWith("$name:") } }

    /**
     * Load a model. A model that is loaded already is released first. With an
     * accelerator the CPU only feeds it, thus the threads stay at half the cores.
     */
    suspend fun load(config: EngineConfig): LoadedModel = withContext(dispatcher) {
        awaitReady()
        releaseLocked()
        val threads = if (config.backend == Backend.CPU) {
            config.threads
        } else {
            config.threads.coerceAtMost(maxOf(1, Runtime.getRuntime().availableProcessors() / 2))
        }
        handle = LlamaNative.load(
            config.path,
            config.mmproj,
            config.backend.deviceName,
            config.backend.prefillDeviceName,
            if (config.backend == Backend.CPU) 0 else 999,
            threads,
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
     * Generate the answer to the conversation. Each emitted piece is a part
     * of the answer or a thinking tag. The flow stops at the end token, at
     * the token limit, or when the collector cancels. The messages are
     * copied on the thread of the caller, thus the engine thread reads no
     * shared text.
     */
    fun generate(
        messages: List<ChatMessage>,
        thinking: Boolean,
        temperature: Float = 0.7f,
        topP: Float = 0.8f,
    ): Flow<Piece> {
        val roles = messages.map { it.role }.toTypedArray()
        val contents = messages.map { it.content }.toTypedArray()
        val images = messages.map { it.image }.toTypedArray()
        return flow {
            val h = requireHandle()
            LlamaNative.chatStart(h, roles, contents, images, thinking, temperature, topP)
            var count = 0
            while (count < MAX_ANSWER_TOKENS) {
                val bytes = LlamaNative.generateNext(h) ?: break
                count += 1
                if (bytes.size > 1) {
                    emit(Piece.Text(String(bytes, 1, bytes.size - 1, Charsets.UTF_8)))
                }
                when (bytes[0].toInt()) {
                    1 -> emit(Piece.ThinkOpen)
                    2 -> emit(Piece.ThinkClose)
                }
            }
        }.flowOn(dispatcher)
    }

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
