package ai.airi.qwenmobile

import android.content.Context
import android.os.PowerManager

/**
 * The hold of the engine on the CPU while it computes.
 *
 * With the screen off, the system suspends the SoC while the engine thread
 * waits for the NPU. Each suspend delays one token by 0.3 to 1.3 s of wall
 * time. CLOCK_MONOTONIC does not count the suspend, thus the speed line of
 * the app does not show the delay. The foreground service does not prevent a
 * suspend, but a partial wake lock does.
 *
 * The hold counts its holders, and the system lock is not reference counted:
 * the first holder acquires the lock and the last holder releases it. Each
 * acquire has a timeout, thus a native call that does not return loses the
 * lock after [timeoutMs]. A holder that computes for a longer time calls
 * [renew]. All calls are thread safe.
 *
 * @param lock The system lock
 * @param timeoutMs The timeout of one acquire
 * @param clockMs A monotonic clock in milliseconds, for the interval of [renew]
 */
class WakeHold(
    private val lock: SystemLock,
    private val timeoutMs: Long = TIMEOUT_MS,
    private val clockMs: () -> Long = { System.nanoTime() / 1_000_000 },
) {
    /** The system lock: a partial wake lock on the phone, a fake in the unit tests. */
    interface SystemLock {
        /** Hold the CPU for a maximum of [timeoutMs]. A second call starts the timeout again. */
        fun acquire(timeoutMs: Long)

        /** Let the CPU sleep. A call without a hold does nothing. */
        fun release()
    }

    private var holders = 0
    private var armedAtMs = 0L

    /** The number of holders. */
    val count: Int
        @Synchronized get() = holders

    /** Add a holder, and acquire the lock with a new timeout. */
    @Synchronized
    fun enter() {
        holders += 1
        arm()
    }

    /**
     * Start the timeout again when half of it has passed. A call without a
     * holder does nothing. Thus a call for each token makes a maximum of one
     * system call in each half of the timeout.
     */
    @Synchronized
    fun renew() {
        if (holders > 0 && clockMs() - armedAtMs >= timeoutMs / 2) {
            arm()
        }
    }

    /** Remove a holder. The last holder releases the lock. */
    @Synchronized
    fun exit() {
        check(holders > 0) { "The wake hold has no holder to remove" }
        holders -= 1
        if (holders == 0) {
            lock.release()
        }
    }

    private fun arm() {
        lock.acquire(timeoutMs)
        armedAtMs = clockMs()
    }

    companion object {
        /** The timeout of one acquire. A load or a benchmark takes less time. */
        const val TIMEOUT_MS = 10 * 60 * 1000L

        /** The tag of the lock in "dumpsys power". */
        const val TAG = "qwenmobile:engine"

        /** A hold on one partial wake lock of the app, which is not reference counted. */
        fun forEngine(context: Context): WakeHold {
            val power = context.getSystemService(PowerManager::class.java)
            val lock = power.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, TAG).apply { setReferenceCounted(false) }
            return WakeHold(object : SystemLock {
                override fun acquire(timeoutMs: Long) = lock.acquire(timeoutMs)

                override fun release() = lock.release()
            })
        }
    }
}

/**
 * Run [block] with a holder on this hold, or without a holder for a null
 * hold. Each return path of the block removes the holder: the end, an
 * exception and a cancel.
 */
inline fun <T> WakeHold?.around(block: () -> T): T {
    if (this == null) {
        return block()
    }
    enter()
    try {
        return block()
    } finally {
        exit()
    }
}
