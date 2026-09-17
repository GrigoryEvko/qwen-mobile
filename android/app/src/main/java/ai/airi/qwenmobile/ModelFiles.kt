package ai.airi.qwenmobile

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.Settings
import java.io.File

/** Where the app finds GGUF files, and the all-files permission around it. */
object ModelFiles {
    /** The shared directory. adb can push files here. */
    fun publicDir(): File = File(Environment.getExternalStorageDirectory(), "qwen/models")

    /** The app-private external directory. No permission is necessary. */
    fun privateDir(context: Context): File? = context.getExternalFilesDir("models")

    /** All GGUF language models from the two directories, sorted by name. Projector files are not models. */
    fun list(context: Context): List<File> {
        val dirs = listOfNotNull(publicDir(), privateDir(context))
        return dirs.flatMap { dir ->
            dir.listFiles { file -> file.isFile && isGguf(file) && !isProjector(file) }?.toList().orEmpty()
        }.sortedBy { it.name }
    }

    /**
     * The vision projector of a model: a projector GGUF in the same directory
     * whose name starts with the model family, the first two dash-separated
     * parts of the model name ("Qwen3.5-2B"). F16 comes first.
     */
    fun mmprojFor(model: File): File? {
        val family = model.name.split("-").take(2).joinToString("-")
        return model.parentFile
            ?.listFiles { file -> file.isFile && isGguf(file) && isProjector(file) && file.name.startsWith(family) }
            ?.sortedBy { if (it.name.contains("F16")) 0 else 1 }
            ?.firstOrNull()
    }

    private fun isGguf(file: File): Boolean = file.name.endsWith(".gguf")

    private fun isProjector(file: File): Boolean = file.name.contains("mmproj")

    /** Tell if the app can read the shared directory. */
    fun hasAllFilesAccess(): Boolean =
        Build.VERSION.SDK_INT < Build.VERSION_CODES.R || Environment.isExternalStorageManager()

    /** The settings page where the user grants the all-files access. */
    fun allFilesAccessIntent(context: Context): Intent = Intent(
        Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
        Uri.parse("package:${context.packageName}"),
    )
}
