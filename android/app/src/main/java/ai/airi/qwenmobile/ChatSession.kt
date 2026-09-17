package ai.airi.qwenmobile

import android.content.Context
import android.os.SystemClock
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.IdentityHashMap

/**
 * The conversation of the process: the messages, the running answer, and
 * the model life cycle. It lives in an application scope, thus an answer
 * continues while the app is in the background or on another tab. The
 * fragment is a view over it and reads the three flows.
 *
 * All mutations of [messages] happen on the main thread.
 */
object ChatSession {
    /** The messages, oldest first. Read on the main thread only. */
    val messages = ArrayList<ChatMessage>()

    /** The speed line of each answer, by message identity. */
    val metaByMessage = IdentityHashMap<ChatMessage, String>()

    /** Counts the structural changes: inserts and clears. */
    val structure: StateFlow<Long> get() = structureFlow

    /** Counts the text changes of the last message while an answer streams. */
    val revision: StateFlow<Long> get() = revisionFlow

    /** True while an answer streams. */
    val generating: StateFlow<Boolean> get() = generatingFlow

    /** True while the model loads. */
    val loading: StateFlow<Boolean> get() = loadingFlow

    /** The problems of the automatic loads, for the screen to show. */
    val problems: SharedFlow<String> get() = problemFlow

    private val structureFlow = MutableStateFlow(0L)
    private val revisionFlow = MutableStateFlow(0L)
    private val generatingFlow = MutableStateFlow(false)
    private val loadingFlow = MutableStateFlow(false)
    private val problemFlow = MutableSharedFlow<String>(extraBufferCapacity = 4)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)

    private lateinit var app: Context
    private lateinit var store: ConversationStore
    private var generation: Job? = null
    private var restored = false
    private var autoLoaded = false

    /** True when the settings changed the model while an answer streams. The reload waits for the end. */
    private var reloadPending = false

    /** The interval between two saves of a streaming answer. */
    private const val SAVE_INTERVAL_MS = 3000L

    /** Restore the conversation from disk one time. */
    fun init(context: Context) {
        if (restored) {
            return
        }
        restored = true
        app = context.applicationContext
        store = ConversationStore(app)
        scope.launch {
            val entries = withContext(Dispatchers.IO) { store.load() }
            if (messages.isEmpty() && entries.isNotEmpty()) {
                for (entry in entries) {
                    messages += entry.message
                    entry.meta?.let { metaByMessage[entry.message] = it }
                }
                structureFlow.value += 1
            }
        }
        // A model change in the settings is a switch: the messages stay, the engine reloads.
        scope.launch {
            SettingsStore.of(app).state.collect { s ->
                val loaded = LlamaEngine.state.value ?: return@collect
                val c = loaded.config
                if (s.modelPath != c.path || s.backend != c.backend || s.threads != c.threads || s.nCtx != c.nCtx) {
                    reloadPending = true
                    if (!generatingFlow.value) {
                        reloadNow()
                    }
                }
            }
        }
    }

    /** The queued reload of the settings model. The problems go to [problems]. */
    private fun reloadNow() {
        reloadPending = false
        scope.launch {
            loadFromSettings()?.let { problemFlow.emit(it) }
        }
    }

    // --- the model ---

    /**
     * Load the model of the settings. Returns null when a model is loaded at
     * the end, or the text of the problem.
     */
    suspend fun loadFromSettings(): String? {
        LlamaEngine.awaitReady()
        val s = SettingsStore.of(app).state.value
        val path = s.modelPath
        if (path == null || !File(path).isFile) {
            return app.getString(R.string.no_model_selected)
        }
        var config = EngineConfig(
            path = path,
            backend = s.backend,
            threads = s.threads,
            nCtx = s.nCtx,
            mmproj = ModelFiles.mmprojFor(File(path))?.absolutePath,
        )
        if (LlamaEngine.state.value?.config == config) {
            return null
        }
        // A load that does not fit in memory gets killed while it runs. The
        // hybrid backend holds the weights twice, thus it falls back to the GPU.
        var problem: String? = null
        if (!MemoryBudget.fits(app, config)) {
            val fallback = if (config.backend == Backend.HYBRID) Backend.GPU else Backend.CPU
            problem = app.getString(
                R.string.load_too_large,
                MemoryBudget.format(MemoryBudget.weightBytes(config)),
                MemoryBudget.format(MemoryBudget.availableBytes(app)),
                fallback.label,
            )
            config = config.copy(backend = fallback)
            if (!MemoryBudget.fits(app, config)) {
                return app.getString(R.string.load_no_memory, MemoryBudget.format(MemoryBudget.weightBytes(config)))
            }
        }
        loadingFlow.value = true
        return try {
            LlamaEngine.load(config)
            LlamaEngine.resetChat()
            // The fallback is a notice, not a failure: the model is loaded.
            problem?.let { problemFlow.emit(it) }
            null
        } catch (e: Exception) {
            app.getString(R.string.load_failed, e.message ?: "?")
        } finally {
            loadingFlow.value = false
        }
    }

    /** Load the model one time per process when the settings name one. Returns the problem text or null. */
    suspend fun autoLoad(): String? {
        if (autoLoaded || LlamaEngine.state.value != null) {
            return null
        }
        autoLoaded = true
        if (SettingsStore.of(app).state.value.modelPath == null) {
            return null
        }
        return loadFromSettings()
    }

    fun unload() {
        stop()
        scope.launch { LlamaEngine.unload() }
    }

    /** Start an empty conversation and clear the model memory. */
    fun clear() {
        stop()
        messages.clear()
        metaByMessage.clear()
        structureFlow.value += 1
        scope.launch {
            LlamaEngine.resetChat()
            withContext(Dispatchers.IO) { store.clear() }
        }
    }

    // --- the turn ---

    /** Cancel the running answer, if any. */
    fun stop() {
        generation?.cancel()
    }

    /**
     * Add the user message and stream the answer into a new assistant
     * message. The answer runs in the application scope and survives the
     * fragment view. The result of the load, when one is necessary, goes to
     * [onProblem].
     */
    fun send(text: String, image: ByteArray?, onProblem: (String) -> Unit) {
        if (generatingFlow.value) {
            return
        }
        messages += ChatMessage("user", text, image)
        val history = messages.toList()
        val answer = ChatMessage("assistant", "")
        messages += answer
        structureFlow.value += 1
        generatingFlow.value = true
        GenerationService.start(app)
        generation = scope.launch {
            var lastSave = SystemClock.elapsedRealtime()
            try {
                if (LlamaEngine.state.value == null) {
                    val problem = loadFromSettings()
                    if (problem != null) {
                        onProblem(problem)
                        return@launch
                    }
                }
                val s = SettingsStore.of(app).state.value
                LlamaEngine.generate(history, s.thinking, s.temperature, s.topP).collect { piece ->
                    answer.content += piece
                    revisionFlow.value += 1
                    val now = SystemClock.elapsedRealtime()
                    if (now - lastSave > SAVE_INTERVAL_MS) {
                        lastSave = now
                        save()
                    }
                }
                metaByMessage[answer] = LlamaEngine.stats()
            } catch (e: Exception) {
                if (e !is kotlinx.coroutines.CancellationException) {
                    answer.content += "\n\n*${e.message}*"
                }
            } finally {
                revisionFlow.value += 1
                generatingFlow.value = false
                GenerationService.stop(app)
                save()
                if (reloadPending) {
                    reloadNow()
                }
            }
        }
    }

    /** Write the conversation to disk, off the main thread. */
    private fun save() {
        val entries = messages.map { ConversationStore.Entry(it, metaByMessage[it]) }
        scope.launch(Dispatchers.IO) { store.save(entries) }
    }
}
