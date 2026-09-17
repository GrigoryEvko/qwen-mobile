package ai.airi.qwenmobile

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.security.MessageDigest

/**
 * The conversation on disk: ``<dir>/conversation.json`` with the messages
 * and ``<dir>/images/<hash>.jpg`` with the attached images.
 *
 * A save writes the whole file through a temporary file and a rename, adds
 * the images that are missing, and removes the images that no message
 * refers to. Complexity is O(messages). Call the methods off the main
 * thread, one at a time.
 */
class ConversationStore(private val dir: File) {
    constructor(context: Context) : this(context.filesDir)

    private val file = File(dir, "conversation.json")
    private val imageDir = File(dir, "images")

    /**
     * One saved message: a copy of the fields of the message and its speed
     * line. The copy is made on the thread that owns the message, thus the
     * write reads no text that changes.
     */
    class Entry(
        val role: String,
        val content: String,
        val thinking: String,
        val thinkingMs: Long,
        val image: ByteArray?,
        val meta: String?,
    ) {
        /** The message of the entry, in the phase DONE. */
        fun toMessage(): ChatMessage = ChatMessage(role, content, image).also {
            it.thinking = thinking
            it.thinkingMs = thinkingMs
        }

        companion object {
            /** The entry of [message] with its speed line [meta]. */
            fun of(message: ChatMessage, meta: String?): Entry =
                Entry(message.role, message.content, message.thinking, message.thinkingMs, message.image, meta)
        }
    }

    /**
     * Write the messages.
     *
     * @throws IOException If the directory is not writable or the rename fails
     */
    @Synchronized
    fun save(entries: List<Entry>) {
        dir.mkdirs()
        imageDir.mkdirs()
        val used = HashSet<String>()
        val array = JSONArray()
        for (entry in entries) {
            val obj = JSONObject()
                .put("role", entry.role)
                .put("content", entry.content)
                .put("thinking", entry.thinking)
                .put("thinkingMs", entry.thinkingMs)
            entry.image?.let { bytes ->
                val name = hashName(bytes)
                val target = File(imageDir, name)
                if (!target.exists()) {
                    target.writeBytes(bytes)
                }
                used += name
                obj.put("image", name)
            }
            entry.meta?.let { obj.put("meta", it) }
            array.put(obj)
        }
        val tmp = File(dir, file.name + ".tmp")
        FileOutputStream(tmp).use { out ->
            out.write(JSONObject().put("messages", array).toString().toByteArray(Charsets.UTF_8))
            // The bytes are on the disk before the rename, thus a power loss keeps the previous file.
            out.fd.sync()
        }
        if (!tmp.renameTo(file)) {
            throw IOException("The rename of ${tmp.name} to ${file.name} failed in $dir")
        }
        imageDir.listFiles()?.forEach { image ->
            if (image.name !in used) {
                image.delete()
            }
        }
    }

    /** Read the messages, or an empty list when there is no file or it is not readable. */
    @Synchronized
    fun load(): List<Entry> {
        if (!file.isFile) {
            return emptyList()
        }
        return try {
            val array = JSONObject(file.readText()).getJSONArray("messages")
            List(array.length()) { i ->
                val obj = array.getJSONObject(i)
                val image = obj.optString("image", "").takeIf { it.isNotEmpty() }
                    ?.let { File(imageDir, it) }?.takeIf { it.isFile }?.readBytes()
                Entry(
                    role = obj.getString("role"),
                    content = obj.getString("content"),
                    thinking = obj.optString("thinking", ""),
                    thinkingMs = obj.optLong("thinkingMs", 0L),
                    image = image,
                    meta = obj.optString("meta", "").takeIf { it.isNotEmpty() },
                )
            }
        } catch (e: Exception) {
            Log.w(TAG, "The conversation file is not readable, the conversation starts empty", e)
            emptyList()
        }
    }

    /** Remove the file and the images. */
    @Synchronized
    fun clear() {
        file.delete()
        imageDir.listFiles()?.forEach { it.delete() }
    }

    private fun hashName(bytes: ByteArray): String {
        val digest = MessageDigest.getInstance("SHA-1").digest(bytes)
        return digest.joinToString("") { "%02x".format(it) } + ".jpg"
    }

    private companion object {
        const val TAG = "ConversationStore"
    }
}
