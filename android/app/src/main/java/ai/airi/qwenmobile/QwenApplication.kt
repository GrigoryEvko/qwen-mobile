package ai.airi.qwenmobile

import android.app.Application

/**
 * Starts the native backends on the engine thread, thus the first screen
 * opens at once. The engine gets the partial wake lock of the app.
 */
class QwenApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        LlamaEngine.start(
            applicationInfo.nativeLibraryDir,
            filesDir.absolutePath,
            cacheDir.absolutePath,
            WakeHold.forEngine(this),
        )
    }
}
