package ai.airi.qwenmobile

import android.app.Application

/** Starts the native backends on the engine thread, thus the first screen opens at once. */
class QwenApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        LlamaEngine.start(applicationInfo.nativeLibraryDir, filesDir.absolutePath, cacheDir.absolutePath)
    }
}
