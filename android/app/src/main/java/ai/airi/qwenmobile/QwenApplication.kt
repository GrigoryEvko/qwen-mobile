package ai.airi.qwenmobile

import android.app.Application

/** Initializes the native backends before any screen uses them. */
class QwenApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        LlamaNative.initialize(applicationInfo.nativeLibraryDir)
    }
}
