package ai.airi.qwenmobile

import android.content.Context
import android.content.SharedPreferences
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

/** The user settings. The model path is null until the user selects a model. */
data class AppSettings(
    val modelPath: String?,
    val backend: Backend,
    val threads: Int = 4,
    val nCtx: Int = 8192,
    val thinking: Boolean = false,
    val temperature: Float = 0.7f,
    val topP: Float = 0.8f,
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

    /** The current settings. */
    val state: StateFlow<AppSettings> = stateFlow

    /** Replace the settings by the result of [transform] and save them. */
    @Synchronized
    fun update(transform: (AppSettings) -> AppSettings) {
        val next = sanitize(transform(stateFlow.value))
        write(next)
        stateFlow.value = next
    }

    private fun read(): AppSettings = sanitize(
        AppSettings(
            modelPath = prefs.getString(KEY_MODEL, null),
            backend = prefs.getString(KEY_BACKEND, null)?.let { name ->
                Backend.entries.firstOrNull { it.name == name }
            } ?: defaultBackend(),
            threads = prefs.getInt(KEY_THREADS, 4),
            nCtx = prefs.getInt(KEY_N_CTX, 8192),
            thinking = prefs.getBoolean(KEY_THINKING, false),
            temperature = prefs.getFloat(KEY_TEMPERATURE, 0.7f),
            topP = prefs.getFloat(KEY_TOP_P, 0.8f),
        ),
    )

    private fun write(s: AppSettings) {
        prefs.edit()
            .putString(KEY_MODEL, s.modelPath)
            .putString(KEY_BACKEND, s.backend.name)
            .putInt(KEY_THREADS, s.threads)
            .putInt(KEY_N_CTX, s.nCtx)
            .putBoolean(KEY_THINKING, s.thinking)
            .putFloat(KEY_TEMPERATURE, s.temperature)
            .putFloat(KEY_TOP_P, s.topP)
            .apply()
    }

    companion object {
        private const val FILE = "settings"
        private const val KEY_MODEL = "model_path"
        private const val KEY_BACKEND = "backend"
        private const val KEY_THREADS = "threads"
        private const val KEY_N_CTX = "n_ctx"
        private const val KEY_THINKING = "thinking"
        private const val KEY_TEMPERATURE = "temperature"
        private const val KEY_TOP_P = "top_p"

        /** The permitted context lengths, in tokens. */
        val CONTEXT_LENGTHS: List<Int> = listOf(2048, 4096, 8192, 16384)

        const val MIN_THREADS = 1
        const val MAX_THREADS = 8
        const val MAX_TEMPERATURE = 1.5f
        const val MIN_TOP_P = 0.5f

        @Volatile
        private var instance: SettingsStore? = null

        /** The store of the process. The first call creates it on the application context. */
        fun of(context: Context): SettingsStore =
            instance ?: synchronized(this) {
                instance ?: SettingsStore(context.applicationContext).also { instance = it }
            }

        /** The GPU when it is visible, else the CPU. The NPU is experimental, thus the user selects it. */
        fun defaultBackend(): Backend = if (LlamaEngine.has(Backend.GPU)) Backend.GPU else Backend.CPU

        /** Keep every value inside its permitted range. A backend without a visible device falls back. */
        fun sanitize(s: AppSettings): AppSettings = s.copy(
            backend = if (LlamaEngine.has(s.backend)) s.backend else defaultBackend(),
            threads = s.threads.coerceIn(MIN_THREADS, MAX_THREADS),
            nCtx = if (s.nCtx in CONTEXT_LENGTHS) s.nCtx else 8192,
            temperature = s.temperature.coerceIn(0f, MAX_TEMPERATURE),
            topP = s.topP.coerceIn(MIN_TOP_P, 1f),
        )
    }
}
