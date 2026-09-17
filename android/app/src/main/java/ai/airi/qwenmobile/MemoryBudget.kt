package ai.airi.qwenmobile

import android.app.ActivityManager
import android.content.Context
import java.io.File

/**
 * The memory a model load needs against what the phone can give. A load
 * that does not fit gets killed by the low-memory killer while it runs,
 * thus the check happens before the load.
 */
object MemoryBudget {
    /** The memory the runtime needs next to the weights: context, graph, buffers. */
    private const val RUNTIME_BYTES = 1_200L shl 20

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

    /** The bytes the system reports as available without swapping. */
    fun availableBytes(context: Context): Long {
        val info = ActivityManager.MemoryInfo()
        val am = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        am.getMemoryInfo(info)
        return info.availMem
    }

    /** True when the load fits with the runtime margin. */
    fun fits(context: Context, config: EngineConfig): Boolean =
        weightBytes(config) + RUNTIME_BYTES <= availableBytes(context)

    /** A text like "7.3 GB". */
    fun format(bytes: Long): String = "%.1f GB".format(bytes / 1e9)
}
