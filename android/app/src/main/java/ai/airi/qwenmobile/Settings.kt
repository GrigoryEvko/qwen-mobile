package ai.airi.qwenmobile

import android.content.Context
import android.content.SharedPreferences
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

/**
 * The token budget of one image. More tokens read small text better. The
 * encoder and the prompt grow with the count: the encode is quadratic in
 * the patches, the prefill is linear in the tokens.
 */
enum class ImageDetail(val tokens: Int) {
    FAST(256),
    STANDARD(576),

    /** 768 is the ceiling until the encoder is correct on the NPU at 1024 tokens. */
    DETAILED(768);

    companion object {
        /** The budget of a fresh install. */
        val DEFAULT = STANDARD
    }
}

/** The user settings. The model path is null until the user selects a model. */
data class AppSettings(
    val modelPath: String?,
    val backend: Backend,
    val threads: Int = 4,
    val nCtx: Int = 8192,
    /** Keep the image encoder on the GPU also when the prompt runs on the NPU. */
    val visionOnGpu: Boolean = false,
    val imageDetail: ImageDetail = ImageDetail.DEFAULT,
    val thinking: Boolean = false,
    val temperature: Float = SettingsStore.DEFAULT_TEMPERATURE,
    val topP: Float = SettingsStore.DEFAULT_TOP_P,
    /**
     * Draft the answer with the MTP block of the model. A change loads the
     * model again, because the MTP block loads with the weights and the
     * context keeps the state snapshots of a draft.
     */
    val speculative: Boolean = false,
    /**
     * A standing instruction that leads every conversation. The chat
     * template of Qwen3.5 puts it in a system turn. An empty value sends
     * no system turn at all, thus the model keeps its own default.
     */
    val systemPrompt: String = "",
)

/**
 * The process-wide store of the settings, backed by SharedPreferences.
 *
 * Reads come from [state]. Writes go through [update], reach the disk at
 * once, and every collector of [state] sees the new value.
 */
class SettingsStore private constructor(context: Context) {
    private val prefs: SharedPreferences = context.getSharedPreferences(FILE, Context.MODE_PRIVATE)
    private val stateFlow = MutableStateFlow(read())
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)

    /** The current settings. */
    val state: StateFlow<AppSettings> = stateFlow

    init {
        // The device list comes after the first read. A backend without a device changes to the default at that time.
        scope.launch {
            LlamaEngine.awaitReady()
            update { it }
        }
    }

    /** Replace the settings by the result of [transform] and save them. */
    @Synchronized
    fun update(transform: (AppSettings) -> AppSettings) {
        val next = sanitize(transform(stateFlow.value))
        write(next)
        stateFlow.value = next
    }

    /**
     * The value of one key, or [default] when the file holds a value of
     * another type there. SharedPreferences then throws ClassCastException,
     * and read() runs in the constructor of the store, thus one such value
     * stops the app at each start (task #94).
     */
    private inline fun <T> readOr(key: String, default: T, get: () -> T): T = try {
        get()
    } catch (e: ClassCastException) {
        Log.w(FILE, "The setting $key has a value of a wrong type, the default is used", e)
        default
    }

    private fun read(): AppSettings = sanitize(
        AppSettings(
            modelPath = readOr(KEY_MODEL, null) { prefs.getString(KEY_MODEL, null) },
            backend = readOr(KEY_BACKEND, null) { prefs.getString(KEY_BACKEND, null) }?.let { name ->
                Backend.entries.firstOrNull { it.name == name }
            } ?: defaultBackend(),
            threads = readOr(KEY_THREADS, 4) { prefs.getInt(KEY_THREADS, 4) },
            nCtx = readOr(KEY_N_CTX, 8192) { prefs.getInt(KEY_N_CTX, 8192) },
            visionOnGpu = readOr(KEY_VISION_GPU, false) { prefs.getBoolean(KEY_VISION_GPU, false) },
            imageDetail = readOr(KEY_IMAGE_DETAIL, null) { prefs.getString(KEY_IMAGE_DETAIL, null) }?.let { name ->
                ImageDetail.entries.firstOrNull { it.name == name }
            } ?: ImageDetail.DEFAULT,
            thinking = readOr(KEY_THINKING, false) { prefs.getBoolean(KEY_THINKING, false) },
            temperature = readOr(KEY_TEMPERATURE, DEFAULT_TEMPERATURE) { prefs.getFloat(KEY_TEMPERATURE, DEFAULT_TEMPERATURE) },
            topP = readOr(KEY_TOP_P, DEFAULT_TOP_P) { prefs.getFloat(KEY_TOP_P, DEFAULT_TOP_P) },
            speculative = readOr(KEY_SPECULATIVE, false) { prefs.getBoolean(KEY_SPECULATIVE, false) },
            systemPrompt = readOr(KEY_SYSTEM_PROMPT, null) { prefs.getString(KEY_SYSTEM_PROMPT, null) } ?: "",
        ),
    )

    private fun write(s: AppSettings) {
        prefs.edit()
            .putString(KEY_MODEL, s.modelPath)
            .putString(KEY_BACKEND, s.backend.name)
            .putInt(KEY_THREADS, s.threads)
            .putInt(KEY_N_CTX, s.nCtx)
            .putBoolean(KEY_VISION_GPU, s.visionOnGpu)
            .putString(KEY_IMAGE_DETAIL, s.imageDetail.name)
            .putBoolean(KEY_THINKING, s.thinking)
            .putFloat(KEY_TEMPERATURE, s.temperature)
            .putFloat(KEY_TOP_P, s.topP)
            .putBoolean(KEY_SPECULATIVE, s.speculative)
            .putString(KEY_SYSTEM_PROMPT, s.systemPrompt)
            .apply()
    }

    companion object {
        private const val FILE = "settings"
        private const val KEY_MODEL = "model_path"
        private const val KEY_BACKEND = "backend"
        private const val KEY_THREADS = "threads"
        private const val KEY_N_CTX = "n_ctx"
        private const val KEY_VISION_GPU = "vision_on_gpu"
        private const val KEY_IMAGE_DETAIL = "image_detail"
        private const val KEY_THINKING = "thinking"
        private const val KEY_TEMPERATURE = "temperature"
        private const val KEY_TOP_P = "top_p"
        private const val KEY_SPECULATIVE = "speculative"
        private const val KEY_SYSTEM_PROMPT = "system_prompt"

        /** The permitted context lengths, in tokens. */
        val CONTEXT_LENGTHS: List<Int> = listOf(2048, 4096, 8192, 16384)

        const val MIN_THREADS = 1
        const val MAX_THREADS = 8
        const val MAX_TEMPERATURE = 1.5f
        const val MIN_TOP_P = 0.5f

        /** The sampling of a fresh install: the values that Qwen3.5 recommends without thinking. */
        const val DEFAULT_TEMPERATURE = 0.7f
        const val DEFAULT_TOP_P = 0.8f

        /** The character ceiling of the system prompt, about 500 tokens. */
        const val MAX_SYSTEM_PROMPT = 2000

        @Volatile
        private var instance: SettingsStore? = null

        /** The store of the process. The first call creates it on the application context. */
        fun of(context: Context): SettingsStore =
            instance ?: synchronized(this) {
                instance ?: SettingsStore(context.applicationContext).also { instance = it }
            }

        /** The GPU when it is visible or not known yet, else the CPU. The NPU is experimental, thus the user selects it. */
        fun defaultBackend(): Backend = if (!LlamaEngine.ready || LlamaEngine.has(Backend.GPU)) Backend.GPU else Backend.CPU

        /**
         * Keep every value inside its permitted range. A backend without a
         * visible device falls back, but only when the device list is known.
         */
        fun sanitize(s: AppSettings): AppSettings = s.copy(
            backend = if (!LlamaEngine.ready || LlamaEngine.has(s.backend)) s.backend else defaultBackend(),
            threads = s.threads.coerceIn(MIN_THREADS, MAX_THREADS),
            nCtx = if (s.nCtx in CONTEXT_LENGTHS) s.nCtx else 8192,
            // coerceIn keeps NaN, and NaN stops the sampler of the engine, thus NaN takes the default (task #164).
            temperature = if (s.temperature.isNaN()) DEFAULT_TEMPERATURE else s.temperature.coerceIn(0f, MAX_TEMPERATURE),
            topP = if (s.topP.isNaN()) DEFAULT_TOP_P else s.topP.coerceIn(MIN_TOP_P, 1f),
            // The prompt costs context on every turn and is prefilled again
            // whenever it changes, thus it is bounded. The limit counts UTF-16
            // units: a cut between the two units of a surrogate pair leaves a
            // high surrogate that is not UTF-8 in the engine, thus the cut
            // goes before the pair (task #165).
            systemPrompt = s.systemPrompt.trim().let { p ->
                if (p.length > MAX_SYSTEM_PROMPT && p[MAX_SYSTEM_PROMPT - 1].isHighSurrogate()) {
                    p.take(MAX_SYSTEM_PROMPT - 1)
                } else {
                    p.take(MAX_SYSTEM_PROMPT)
                }
            },
        )
    }
}
