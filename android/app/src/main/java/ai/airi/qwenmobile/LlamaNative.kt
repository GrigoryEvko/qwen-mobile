package ai.airi.qwenmobile

import androidx.annotation.Keep
import java.nio.ByteBuffer

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
     * @param visionDevice   The device of the image encoder (HTP0, GPUOpenCL), or null for the
     *                       GPU when it is present, else the CPU
     * @param gpuLayers      The number of layers on the device, 0 for the CPU only
     * @param threads        The number of CPU threads
     * @param nCtx           The context length in tokens
     * @param imageMaxTokens The maximum number of vision tokens of one image, between 64 and 768
     * @param cacheDir       The cache directory of the app for the prompt states and the encoded
     *                       images, or null to keep them in RAM only
     * @return The engine handle
     * @throws RuntimeException If a device, the model or a context does not load, or a limit is out of range
     */
    @JvmStatic external fun load(
        path: String,
        mmproj: String?,
        device: String?,
        prefillDevice: String?,
        visionDevice: String?,
        gpuLayers: Int,
        threads: Int,
        nCtx: Int,
        imageMaxTokens: Int,
        cacheDir: String?,
    ): Long

    /**
     * The pixels of an image file for the vision encoder. The engine calls
     * this on its thread from [chatStart], for an image that its cache does
     * not know, with the token limit of the load. Refer to
     * [ImageBytes.decodeForModel] for the contract.
     */
    @Keep
    @JvmStatic
    fun decodeImage(bytes: ByteArray, maxTokens: Int, dims: IntArray): ByteBuffer? =
        ImageBytes.decodeForModel(bytes, maxTokens, dims)

    /** Release the engine. The handle is not valid after this call. */
    @JvmStatic external fun free(handle: Long)

    /** One line with the model description, its size, and the engine settings. */
    @JvmStatic external fun modelInfo(handle: Long): String

    /**
     * Apply the chat template and decode the prompt. The engine restores
     * the longest prefix of the prompt that a snapshot of an earlier turn
     * holds, and decodes only the rest.
     *
     * @param images       One entry per message: the encoded image bytes (JPEG, PNG) or null. A new
     *                     image decodes through [decodeImage], a known one is not decoded
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
     * @return Byte 0 is the kind of the token (0 text, 1 the thinking opens, 2 the thinking closes),
     *   then the UTF-8 bytes of the complete characters so far. Null at the end of the answer. An
     *   end token inside an open thinking gives the close kind one time, then null.
     */
    @JvmStatic external fun generateNext(handle: Long): ByteArray?

    /**
     * Ask the answer to stop. Any thread can call this while [generateNext]
     * runs on the engine thread: the next [generateNext] gives null without
     * a sample. [chatStart] clears the request.
     */
    @JvmStatic external fun requestStop(handle: Long)

    /** One line with the prefill and generation speed of the current turn, and what came from the caches. */
    @JvmStatic external fun stats(handle: Long): String

    /** Clear the model memory, the caches in RAM, and the turn statistics. The files of the caches stay. */
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
