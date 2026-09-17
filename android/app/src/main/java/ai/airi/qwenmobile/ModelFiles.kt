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
     * The vision projector of a model. A projector named after the model
     * file ("<model>.mmproj.gguf") pairs with it: a rotated model needs the
     * projector rotated with it. Else the projector of the model family, the
     * first two dash-separated parts of the name ("Qwen3.5-2B"), F16 first.
     */
    fun mmprojFor(model: File): File? {
        val paired = File(model.parentFile, model.nameWithoutExtension + ".mmproj.gguf")
        if (paired.isFile) {
            return paired
        }
        val family = model.name.split("-").take(2).joinToString("-")
        return model.parentFile
            ?.listFiles { file -> file.isFile && isGguf(file) && isProjector(file) && file.name.startsWith(family) }
            ?.sortedBy { if (it.name.contains("F16")) 0 else 1 }
            ?.firstOrNull()
    }

    /** The recipe tags of a file name and their long forms. */
    private val TAGS = mapOf(
        "HAD" to "Hadamard",
        "CS" to "col-scales",
        "PM" to "MLP-perm",
        "GPTQ" to "GPTQ",
        "QR" to "Qronos",
        "BO" to "block-opt",
        "NQ" to "NeUQI",
        "LR" to "low-rank",
        "CB4" to "codebook",
    )

    /**
     * The display name of a model file: the family, then the recipe tags in
     * long form. "Qwen3.5-2B-Q4_0-HAD-CS-GPTQ-BO.gguf" reads
     * "Qwen3.5-2B · Q4_0 · Hadamard · col-scales · GPTQ · block-opt".
     */
    fun displayName(file: File): String {
        val base = file.nameWithoutExtension.removeSuffix(".mmproj")
        val parts = base.split("-")
        if (parts.size <= 2) {
            return base
        }
        val family = parts.take(2).joinToString("-")
        val tags = parts.drop(2).map { TAGS[it] ?: it }
        return (listOf(family) + tags).joinToString(" · ")
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
