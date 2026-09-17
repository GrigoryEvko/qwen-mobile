package ai.airi.qwenmobile

/**
 * The JNI entry points of libqwenmobile.so. All calls that take a handle
 * must come from one thread. [LlamaEngine] gives that thread.
 */
object LlamaNative {
    private var initialized = false

    init {
        System.loadLibrary("qwenmobile")
    }

    /**
     * Initialize the backends one time. [libDir] is the native library
     * directory of the app: the dynamic backends load from it and the DSP
     * library of the NPU is found there.
     */
    @Synchronized
    fun initialize(libDir: String) {
        if (!initialized) {
            init(libDir)
            initialized = true
        }
    }

    /** Initialize the llama.cpp backends from the library directory and route the log to logcat. */
    @JvmStatic private external fun init(libDir: String)

    /** Make the directory current, thus a profiling build writes its CSV there. */
    @JvmStatic external fun setWorkingDirectory(path: String)

    /** The OpenCL device properties, then one line per ggml backend device. */
    @JvmStatic external fun devices(): String

    /**
     * Load a GGUF model and make its context.
     *
     * @param path       The GGUF file
     * @param mmproj     The vision projector GGUF, or null for a text-only engine
     * @param device         The ggml device name (GPUOpenCL, HTP0), or null for the CPU
     * @param prefillDevice  The device of a second model copy for the prompt (HTP0), or null
     * @param gpuLayers      The number of layers on the device, 0 for the CPU only
     * @param threads        The number of CPU threads
     * @param nCtx           The context length in tokens
     * @return The engine handle
     * @throws RuntimeException If a device, the model or a context does not load
     */
    @JvmStatic external fun load(
        path: String,
        mmproj: String?,
        device: String?,
        prefillDevice: String?,
        gpuLayers: Int,
        threads: Int,
        nCtx: Int,
    ): Long

    /** Release the engine. The handle is not valid after this call. */
    @JvmStatic external fun free(handle: Long)

    /** One line with the model description, its size, and the engine settings. */
    @JvmStatic external fun modelInfo(handle: Long): String

    /**
     * Apply the chat template and decode the prompt.
     *
     * @param images       One entry per message: the encoded image bytes (JPEG, PNG) or null
     * @param thinking     Enable the thinking mode of the chat template
     * @param temperature  The sampling temperature, 0 for greedy
     * @param topP         The nucleus probability mass
     * @return The number of prompt tokens that this call decoded
     * @throws RuntimeException If the template, an image, or the decode fails
     */
    @JvmStatic external fun chatStart(
        handle: Long,
        roles: Array<String>,
        contents: Array<String>,
        images: Array<ByteArray?>,
        thinking: Boolean,
        temperature: Float,
        topP: Float,
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
