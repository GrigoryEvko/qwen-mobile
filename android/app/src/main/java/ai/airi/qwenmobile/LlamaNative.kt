package ai.airi.qwenmobile

/**
 * The JNI entry points of libqwenmobile.so. All calls that take a handle
 * must come from one thread. [LlamaEngine] gives that thread.
 */
object LlamaNative {
    init {
        System.loadLibrary("qwenmobile")
        init()
    }

    /** Initialize the llama.cpp backends and route the log to logcat. */
    @JvmStatic external fun init()

    /** One line per ggml backend device: name, description, type, memory. */
    @JvmStatic external fun devices(): String

    /**
     * Load a GGUF model and make its context.
     *
     * @param path       The GGUF file
     * @param gpuLayers  The number of layers on the GPU, 0 for the CPU only
     * @param threads    The number of CPU threads
     * @param nCtx       The context length in tokens
     * @return The engine handle
     * @throws RuntimeException If the model or the context does not load
     */
    @JvmStatic external fun load(path: String, gpuLayers: Int, threads: Int, nCtx: Int): Long

    /** Release the engine. The handle is not valid after this call. */
    @JvmStatic external fun free(handle: Long)

    /** One line with the model description, its size, and the engine settings. */
    @JvmStatic external fun modelInfo(handle: Long): String

    /**
     * Apply the chat template and decode the prompt.
     *
     * @return The number of prompt tokens that this call decoded
     * @throws RuntimeException If the template or the decode fails
     */
    @JvmStatic external fun chatStart(
        handle: Long,
        roles: Array<String>,
        contents: Array<String>,
        thinking: Boolean,
    ): Int

    /**
     * Sample and decode one token.
     *
     * @return The UTF-8 bytes of the complete characters so far, or null at the end of the answer
     */
    @JvmStatic external fun generateNext(handle: Long): ByteArray?

    /** One line with the prefill and generation speed of the current turn. */
    @JvmStatic external fun stats(handle: Long): String

    /** Clear the model memory and the turn statistics. */
    @JvmStatic external fun resetChat(handle: Long)

    /**
     * Run the llama-bench measurement. The chat memory is empty afterwards.
     *
     * @param pp    The number of prompt tokens in one batch
     * @param tg    The number of tokens to generate one at a time
     * @param reps  The number of repetitions
     * @return The result lines with mean and standard deviation
     */
    @JvmStatic external fun bench(handle: Long, pp: Int, tg: Int, reps: Int): String
}
