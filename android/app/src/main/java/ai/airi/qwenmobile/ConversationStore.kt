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

    /**
     * Read the messages, or an empty list when there is no file. A message
     * that is not readable is skipped, and the others load (task #94, D9).
     *
     * A file that did not load in full stays on the disk as
     * conversation.json.bad (refer to [keepDamaged]). A file that gives no
     * message goes there, thus the next save does not write an empty
     * conversation over it. A file that gives some messages stays, and a
     * copy of it goes there. O(size of the file).
     */
    @Synchronized
    fun load(): List<Entry> {
        if (!file.isFile) {
            return emptyList()
        }
        val bytes = try {
            file.readBytes()
        } catch (e: IOException) {
            Log.w(TAG, "The conversation file is not readable, the conversation starts empty", e)
            return emptyList()
        }
        val array = try {
            JSONObject(String(bytes, Charsets.UTF_8)).getJSONArray("messages")
        } catch (e: Exception) {
            Log.w(TAG, "The conversation file does not parse, the conversation starts empty", e)
            keepDamaged(bytes, move = true)
            return emptyList()
        } catch (e: StackOverflowError) {
            // The org.json parser of Android is recursive and has no limit of the depth.
            Log.w(TAG, "The conversation file is nested too deeply, the conversation starts empty", e)
            keepDamaged(bytes, move = true)
            return emptyList()
        }
        val out = ArrayList<Entry>(array.length())
        for (i in 0 until array.length()) {
            try {
                out += entryOf(array.getJSONObject(i))
            } catch (e: Exception) {
                Log.w(TAG, "Message $i of the conversation file is not readable, it is skipped", e)
            }
        }
        if (out.size < array.length()) {
            keepDamaged(bytes, move = out.isEmpty())
        }
        return out
    }

    /**
     * The entry of one message object of the file. The image name must have
     * the form that [hashName] gives, thus a name such as
     * "../conversation.json" does not read a file outside of the image
     * directory (task #166).
     */
    private fun entryOf(obj: JSONObject): Entry {
        val image = obj.optString("image", "").takeIf { IMAGE_NAME.matches(it) }
            ?.let { File(imageDir, it) }?.takeIf { it.isFile }?.readBytes()
        return Entry(
            role = obj.getString("role"),
            content = obj.getString("content"),
            thinking = obj.optString("thinking", ""),
            thinkingMs = obj.optLong("thinkingMs", 0L),
            image = image,
            meta = obj.optString("meta", "").takeIf { it.isNotEmpty() },
        )
    }

    /**
     * Keep the bytes of a conversation file that did not load in full. The
     * name is conversation.json.bad, or conversation.json.bad-<hash> when
     * that name holds other bytes, thus no kept file is overwritten and the
     * same bytes are kept one time. With [move], the file goes to that name,
     * else it stays and a copy goes there. The images of the file stay only
     * until the next save.
     */
    private fun keepDamaged(bytes: ByteArray, move: Boolean) {
        try {
            val first = File(dir, file.name + ".bad")
            val target = if (!first.exists() || sameBytes(first, bytes)) {
                first
            } else {
                File(dir, file.name + ".bad-" + hashName(bytes).take(12))
            }
            // A target that exists holds the same bytes already.
            if (!target.exists() && !(move && file.renameTo(target))) {
                target.writeBytes(bytes)
            }
            if (move) {
                file.delete()
            }
        } catch (e: IOException) {
            Log.w(TAG, "The damaged conversation file was not kept, it stays as ${file.name}", e)
        }
    }

    /** True when the file holds exactly the bytes. */
    private fun sameBytes(f: File, bytes: ByteArray): Boolean =
        f.length() == bytes.size.toLong() && f.readBytes().contentEquals(bytes)

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

        /** The name of an image file: the SHA-1 of its bytes in hexadecimal, then .jpg (refer to hashName). */
        val IMAGE_NAME = Regex("[0-9a-f]{40}\\.jpg")
    }
}
