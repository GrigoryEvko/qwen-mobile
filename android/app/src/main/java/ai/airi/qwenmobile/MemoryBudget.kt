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
    /** The memory the runtime needs next to the weights: context, graph, buffers. */
    const val RUNTIME_BYTES = 1_200L shl 20

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
        val loaded = LlamaEngine.state.value?.config?.let { weightBytes(it) } ?: 0L
        return fits(weightBytes(config), availableBytes(context), loaded)
    }

    /** True when [weights] plus [RUNTIME_BYTES] fit in [available] plus the [loaded] bytes that a release frees. */
    fun fits(weights: Long, available: Long, loaded: Long): Boolean = weights + RUNTIME_BYTES <= available + loaded

    /** A text like "7.3 GB". */
    fun format(bytes: Long): String = String.format(Locale.US, "%.1f GB", bytes / 1e9)
}
