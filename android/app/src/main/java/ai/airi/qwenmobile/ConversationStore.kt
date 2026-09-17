package ai.airi.qwenmobile

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

/**
 * The conversation on disk: ``filesDir/conversation.json`` with the messages
 * and ``filesDir/images/<hash>.jpg`` with the attached images.
 *
 * A save writes the whole file, adds the images that are missing, and
 * removes the images that no message refers to. Complexity is O(messages).
 */
class ConversationStore(context: Context) {
    private val file = File(context.filesDir, "conversation.json")
    private val imageDir = File(context.filesDir, "images")

    /** One saved message with its speed line. */
    data class Entry(val message: ChatMessage, val meta: String?)

    /** Write the messages. Call it off the main thread. */
    @Synchronized
    fun save(entries: List<Entry>) {
        imageDir.mkdirs()
        val used = HashSet<String>()
        val array = JSONArray()
        for (entry in entries) {
            val obj = JSONObject()
                .put("role", entry.message.role)
                .put("content", entry.message.content)
                .put("thinking", entry.message.thinking)
                .put("thinkingMs", entry.message.thinkingMs)
            entry.message.image?.let { bytes ->
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
        val tmp = File(file.parentFile, file.name + ".tmp")
        tmp.writeText(JSONObject().put("messages", array).toString())
        tmp.renameTo(file)
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
                val message = ChatMessage(obj.getString("role"), obj.getString("content"), image)
                message.thinking = obj.optString("thinking", "")
                message.thinkingMs = obj.optLong("thinkingMs", 0L)
                Entry(message, obj.optString("meta", "").takeIf { it.isNotEmpty() })
            }
        } catch (e: Exception) {
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
}
