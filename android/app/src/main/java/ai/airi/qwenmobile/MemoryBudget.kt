package ai.airi.qwenmobile

import android.app.ActivityManager
import android.content.Context
import java.io.File
import java.util.Locale

/**
 * The memory a model load needs against what the phone can give. A load
 * that does not fit gets killed by the low-memory killer while it runs,
 * thus the check happens before the load. Call [fits] and [availableBytes]
 * off the main thread: they read files.
 */
object MemoryBudget {
    /**
     * The memory the runtime needs next to the weights: context, graph, buffers.
     *
     * The ceiling is the 4B model at the longest context of the settings
     * (16384 tokens) with an F16 KV cache on the NPU. Measured on the phone at
     * 8192 tokens: 4768 MiB of rpcmem buffers and 802 MiB of resident memory
     * after a prompt of 4096 tokens, against 4516 MiB of weights in the file,
     * thus 1055 MiB. The resident part holds a CPU copy of the token
     * embedding (644 MiB), because a model loads without a file mapping when
     * one of its devices cannot load from a mapping. 8192 more positions add
     * 256 MiB of KV cache and about 16 MiB of compute buffer: 1327 MiB. The
     * RAM tiers of the state store (256 MiB) and of the image cache (64 MiB)
     * fill during a conversation: 1647 MiB. The 2B model needs about 1260 MiB.
     */
    const val RUNTIME_BYTES = 1_650L shl 20

    /**
     * The bytes of the KV cache of the MTP draft context for each position of
     * the context. The MTP block is one attention layer: K and V of 4 heads
     * of 256 values in F16 is 4 KB for each position of the 4B model, and
     * 2 KB for the 2B model.
     */
    const val DRAFT_KV_BYTES_PER_POSITION = 4L shl 10

    /**
     * The bytes of the recurrent state snapshots that a draft needs. The
     * context keeps one snapshot for each drafted position (kDraftMax = 4 in
     * spec_policy.h, the n_rs_seq of the context), next to the state itself.
     * One recurrent layer of the 4B model holds a state of 128 x 128 x 32
     * values and a convolution state of 8192 x 3 values in F32, which is
     * 2.094 MiB, and the model has 24 recurrent layers: 50.25 MiB for one
     * snapshot and 201 MiB for the four. The 2B model needs 4 x 19.27 MiB =
     * 77 MiB, thus this value is the ceiling of the two models. llama.cpp
     * logs the total with the state itself: "RS buffer size = 251.25 MiB"
     * for the 4B.
     */
    const val DRAFT_STATE_BYTES = 201L shl 20

    /**
     * The bytes of the compute buffer of the MTP draft context: 100 to 130 MiB
     * for the 2B and the 4B model. This value is the ceiling of the two.
     */
    const val DRAFT_COMPUTE_BYTES = 130L shl 20

    /**
     * The bytes of the second context of the hybrid backend, next to its second
     * copy of the weights: the KV cache, the recurrent state and the compute
     * buffer of the prefill context. The ceiling is the 4B model at the longest
     * context of the settings (16384 tokens), about 750 MiB.
     */
    const val HYBRID_CONTEXT_BYTES = 750L shl 20

    /**
     * The bytes of the weights that the backend holds in memory. The hybrid
     * backend holds one copy for the NPU and one for the GPU.
     */
    fun weightBytes(config: EngineConfig): Long {
        val model = File(config.path).length()
        val projector = config.mmproj?.let { File(it).length() } ?: 0L
        val copies = if (config.backend.prefillDeviceName != null) 2 else 1
        return model * copies + projector
    }

    /**
     * The bytes that speculative decoding adds: the KV cache of the MTP draft
     * context over the whole context length, the compute buffer of that
     * context, and the recurrent state snapshots of the target context. Zero
     * without it.
     */
    fun speculativeBytes(config: EngineConfig): Long =
        if (config.speculativeReady) {
            config.nCtx * DRAFT_KV_BYTES_PER_POSITION + DRAFT_COMPUTE_BYTES + DRAFT_STATE_BYTES
        } else {
            0L
        }

    /** The bytes of the second context of the hybrid backend, zero for the other backends. */
    fun hybridContextBytes(config: EngineConfig): Long =
        if (config.backend.prefillDeviceName != null) HYBRID_CONTEXT_BYTES else 0L

    /** The bytes that the load needs next to [RUNTIME_BYTES]. */
    fun loadBytes(config: EngineConfig): Long = weightBytes(config) + speculativeBytes(config) + hybridContextBytes(config)

    /**
     * The bytes the kernel can give without swapping: MemAvailable of
     * /proc/meminfo, which counts the reclaimable caches. The activity
     * manager's figure leaves those out and refuses loads that fit.
     */
    fun availableBytes(context: Context): Long {
        val fromKernel = runCatching {
            File("/proc/meminfo").useLines { parseMemAvailable(it) }
        }.getOrNull()
        if (fromKernel != null) {
            return fromKernel
        }
        val info = ActivityManager.MemoryInfo()
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        am.getMemoryInfo(info)
        return info.availMem
    }

    /** The MemAvailable value of the lines of /proc/meminfo, in bytes, or null when the line is missing. */
    fun parseMemAvailable(lines: Sequence<String>): Long? =
        lines.firstOrNull { it.startsWith("MemAvailable:") }
            ?.split(Regex("\\s+"))?.getOrNull(1)?.toLongOrNull()?.times(1024)

    /**
     * True when the load fits with the runtime margin. The model that is
     * loaded at the moment is released first, thus its bytes count as free.
     */
    fun fits(context: Context, config: EngineConfig): Boolean {
        val loaded = LlamaEngine.state.value?.config?.let { loadBytes(it) } ?: 0L
        return fits(loadBytes(config), availableBytes(context), loaded)
    }

    /** True when [weights] plus [RUNTIME_BYTES] fit in [available] plus the [loaded] bytes that a release frees. */
    fun fits(weights: Long, available: Long, loaded: Long): Boolean = weights + RUNTIME_BYTES <= available + loaded

    /** A text like "7.3 GB". */
    fun format(bytes: Long): String = String.format(Locale.US, "%.1f GB", bytes / 1e9)
}
