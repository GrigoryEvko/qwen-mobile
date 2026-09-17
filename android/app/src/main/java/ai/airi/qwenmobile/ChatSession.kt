package ai.airi.qwenmobile

import android.content.Context
import android.os.SystemClock
import android.os.Trace
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.util.IdentityHashMap

/**
 * The conversation of the process: the messages, the running answer, and
 * the model life cycle. It lives in an application scope, thus an answer
 * continues while the app is in the background or on another tab. The
 * fragment is a view over it and reads the flows.
 *
 * All mutations of [messages] happen on the main thread. The engine sees
 * one load or one answer at a time, in the order of the requests.
 */
object ChatSession {
    /** The messages, oldest first. Read on the main thread only. */
    val messages = ArrayList<ChatMessage>()

    /** The speed line of each answer, or the error of a failed one, by message identity. */
    val metaByMessage = IdentityHashMap<ChatMessage, String>()

    /** Counts the structural changes: inserts and clears. */
    val structure: StateFlow<Long> get() = structureFlow

    /** Counts the text changes of the last message while an answer streams. */
    val revision: StateFlow<Long> get() = revisionFlow

    /** True while an answer streams. */
    val generating: StateFlow<Boolean> get() = generatingFlow

    /** True while the model loads. */
    val loading: StateFlow<Boolean> get() = loadingFlow

    /** The problem of the last load, or null after a load that succeeded or an unload. */
    val error: StateFlow<String?> get() = errorFlow

    /** The problems of the loads that the user did not start from a message, for the screen to show. */
    val problems: SharedFlow<String> get() = problemFlow

    private val structureFlow = MutableStateFlow(0L)
    private val revisionFlow = MutableStateFlow(0L)
    private val generatingFlow = MutableStateFlow(false)
    private val loadingFlow = MutableStateFlow(false)
    private val errorFlow = MutableStateFlow<String?>(null)
    private val problemFlow = MutableSharedFlow<String>(extraBufferCapacity = 4)
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)

    /** The disk work runs in order on one thread, thus the restore reads the file before the first save writes it. */
    private val diskDispatcher: CoroutineDispatcher = Dispatchers.IO.limitedParallelism(1)

    /** One load or one answer at a time on the engine. The lock is not re-entrant. */
    private val engineLock = Mutex()

    private lateinit var app: Context
    private lateinit var store: ConversationStore
    private var generation: Job? = null
    private var restore: Job? = null
    private var reload: Job? = null
    private var load: Job? = null
    private var initialized = false
    private var autoLoaded = false

    /** The configuration of the last load attempt. A settings change compares against it. */
    private var wantedConfig: EngineConfig? = null

    /** True when the settings changed the model while an answer streams. The reload waits for the end. */
    private var reloadPending = false

    /** The interval between two saves of a streamed answer. */
    private const val SAVE_INTERVAL_MS = 3000L

    /** The wait after a settings change before the reload, thus a slider drag loads one time. */
    private const val RELOAD_DELAY_MS = 500L

    private const val TAG = "ChatSession"

    /** Restore the conversation from disk one time and follow the settings. */
    fun init(context: Context) {
        if (initialized) {
            return
        }
        initialized = true
        app = context.applicationContext
        store = ConversationStore(app)
        restore = scope.launch {
            val entries = withContext(diskDispatcher) { store.load() }
            if (entries.isEmpty()) {
                return@launch
            }
            // A message sent during the restore stays after the restored ones.
            val restored = entries.map { it.toMessage() }
            messages.addAll(0, restored)
            for ((entry, message) in entries.zip(restored)) {
                entry.meta?.let { metaByMessage[message] = it }
            }
            structureFlow.value += 1
        }
        // A model change in the settings is a switch: the messages stay, the engine reloads.
        scope.launch {
            SettingsStore.of(app).state.collect { s ->
                val c = wantedConfig ?: LlamaEngine.state.value?.config ?: return@collect
                if (s.modelPath != c.path || s.backend != c.backend || s.threads != c.threads || s.nCtx != c.nCtx) {
                    scheduleReload()
                }
            }
        }
    }

    /** Reload after a short wait when no answer streams. A running answer defers it to its end. */
    private fun scheduleReload() {
        reloadPending = true
        if (generatingFlow.value) {
            return
        }
        reload?.cancel()
        reload = scope.launch {
            delay(RELOAD_DELAY_MS)
            reloadPending = false
            loadAndReport()
        }
    }

    // --- the model ---

    /**
     * Load the model of the settings. Returns null when a model is loaded at
     * the end, or the text of the problem. The problem also goes to [error].
     */
    suspend fun loadFromSettings(): String? = engineLock.withLock { loadLocked() }

    /** Load the model of the settings in the session scope. The problem, if any, goes to [problems]. */
    fun requestLoad() {
        load?.cancel()
        load = scope.launch { loadAndReport() }
    }

    /** Load the model one time per process when the settings name one. */
    fun autoLoad() {
        if (autoLoaded || LlamaEngine.state.value != null) {
            return
        }
        autoLoaded = true
        if (SettingsStore.of(app).state.value.modelPath == null) {
            return
        }
        requestLoad()
    }

    private suspend fun loadAndReport() {
        loadFromSettings()?.let { problemFlow.emit(it) }
    }

    /** The load of the settings model. The caller holds [engineLock]. */
    private suspend fun loadLocked(): String? {
        val s = SettingsStore.of(app).state.value
        val path = s.modelPath ?: return app.getString(R.string.no_model_selected).also { errorFlow.value = it }
        // The chip shows the load from the request on, also while the backends start at a cold start.
        loadingFlow.value = true
        val problem = try {
            LlamaEngine.awaitReady()
            // A load runs to its end: the native call cannot stop, and the flags stay correct.
            withContext(NonCancellable) { loadModel(path, s) }
        } finally {
            loadingFlow.value = false
        }
        errorFlow.value = problem
        return problem
    }

    /** The load itself. The directory listing and the memory check run off the main thread. */
    private suspend fun loadModel(path: String, s: AppSettings): String? {
        val config = withContext(Dispatchers.IO) {
            val file = File(path)
            if (!file.isFile) {
                null
            } else {
                EngineConfig(path, s.backend, s.threads, s.nCtx, ModelFiles.mmprojFor(file)?.absolutePath, s.visionOnGpu)
            }
        } ?: return app.getString(R.string.no_model_selected)
        wantedConfig = config
        if (LlamaEngine.state.value?.config == config) {
            return null
        }
        // A load that does not fit in memory gets killed while it runs. The
        // hybrid backend holds the weights twice. No other backend is taken
        // in its place: the user selects the compute unit.
        val shortage = withContext(Dispatchers.IO) {
            if (MemoryBudget.fits(app, config)) {
                null
            } else {
                app.getString(
                    R.string.load_no_memory,
                    MemoryBudget.format(MemoryBudget.weightBytes(config)),
                    MemoryBudget.format(MemoryBudget.availableBytes(app)),
                    config.backend.label,
                )
            }
        }
        if (shortage != null) {
            return shortage
        }
        return try {
            LlamaEngine.load(config)
            LlamaEngine.resetChat()
            null
        } catch (e: Exception) {
            Log.e(TAG, "The model did not load: $path", e)
            app.getString(R.string.load_failed, e.message ?: e.javaClass.simpleName)
        }
    }

    /** Release the model. A pending reload is dropped, and a running answer stops. */
    fun unload() {
        reloadPending = false
        reload?.cancel()
        load?.cancel()
        wantedConfig = null
        errorFlow.value = null
        stop()
        scope.launch { engineLock.withLock { LlamaEngine.unload() } }
    }

    /** Start an empty conversation and clear the model memory. */
    fun clear() {
        restore?.cancel()
        stop()
        messages.clear()
        metaByMessage.clear()
        structureFlow.value += 1
        // The engine thread runs its tasks in order, thus the reset follows the stopped answer.
        scope.launch { LlamaEngine.resetChat() }
        persist { store.clear() }
    }

    // --- the turn ---

    /** Cancel the running answer, if any. The engine stops before the coroutine cancels, thus the stop is immediate. */
    fun stop() {
        if (generatingFlow.value) {
            LlamaEngine.stop()
        }
        generation?.cancel()
    }

    /**
     * Add the user message and stream the answer into a new assistant
     * message. The answer runs in the application scope and survives the
     * fragment view. A load problem becomes the text of a failed answer.
     */
    fun send(text: String, image: ByteArray?) {
        if (generatingFlow.value) {
            return
        }
        messages += ChatMessage("user", text, image)
        // An interrupted or failed answer is not part of the conversation the model sees.
        val history = messages.filter { it.phase == ChatMessage.Phase.DONE }
        val s = SettingsStore.of(app).state.value
        val answer = ChatMessage("assistant", "")
        // In thinking mode the chat template opens the thinking, thus the answer starts in it.
        answer.phase = if (s.thinking) ChatMessage.Phase.THINKING else ChatMessage.Phase.ANSWERING
        messages += answer
        structureFlow.value += 1
        generatingFlow.value = true
        GenerationService.start(app)
        generation = scope.launch {
            val stream = AnswerStream(answer) { SystemClock.elapsedRealtime() }
            var error: String? = null
            var cancelled = false
            try {
                engineLock.withLock {
                    if (LlamaEngine.state.value == null) {
                        val problem = loadLocked()
                        if (problem != null) {
                            error = problem
                            return@withLock
                        }
                    }
                    stream.markStart()
                    var lastSave = SystemClock.elapsedRealtime()
                    LlamaEngine.generate(history, s.thinking, s.temperature, s.topP).collect { piece ->
                        // The piece arrives on the main thread here: the section shows the hand-over in a system trace.
                        Trace.beginSection("piece-post")
                        try {
                            stream.accept(piece)
                            revisionFlow.value += 1
                        } finally {
                            Trace.endSection()
                        }
                        val now = SystemClock.elapsedRealtime()
                        if (now - lastSave > SAVE_INTERVAL_MS) {
                            lastSave = now
                            save()
                        }
                    }
                    metaByMessage[answer] = LlamaEngine.stats()
                }
            } catch (e: CancellationException) {
                cancelled = true
            } catch (e: Exception) {
                Log.e(TAG, "The answer failed", e)
                error = e.message ?: e.javaClass.simpleName
            } finally {
                stream.finish(error, cancelled)
                val failure = error
                if (failure != null && answer.phase == ChatMessage.Phase.DONE) {
                    // The answer text stays, and the error goes in the place of the speed line.
                    metaByMessage[answer] = failure
                }
                revisionFlow.value += 1
                generatingFlow.value = false
                GenerationService.stop(app)
                save()
                if (reloadPending) {
                    scheduleReload()
                }
            }
        }
    }

    /**
     * Write the conversation to disk, in order, off the main thread. A
     * completed or a streamed answer is written, a thinking without an
     * answer, an interrupted or a failed one is not.
     */
    private fun save() {
        val entries = messages
            .filter { it.phase == ChatMessage.Phase.DONE || it.phase == ChatMessage.Phase.ANSWERING }
            .map { ConversationStore.Entry.of(it, metaByMessage[it]) }
        persist { store.save(entries) }
    }

    /** Run disk work in order, off the main thread. A failure is logged and does not stop the process. */
    private fun persist(work: () -> Unit) {
        scope.launch(diskDispatcher) {
            try {
                work()
            } catch (e: Exception) {
                Log.e(TAG, "The write of the conversation failed", e)
            }
        }
    }
}
