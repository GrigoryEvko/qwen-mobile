package ai.airi.qwenmobile

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File
import kotlin.random.Random

/**
 * The fuzz tests of the load path of [ConversationStore]: random files,
 * damaged files, files with wrong key types, and saved conversations with
 * random text and images. Refer to [FuzzSwitch] for the iteration count, the
 * seed, and the switch of the finding tests.
 *
 * The properties of load():
 * - It returns and does not throw, whatever the file holds.
 * - It gives no more messages than the file holds.
 * - A save of random entries followed by a load gives the same entries.
 */
class ConversationStoreFuzzTest {
    @get:Rule
    val folder = TemporaryFolder()

    private val file get() = File(folder.root, "conversation.json")

    /** A random string: ASCII, Cyrillic, CJK, emoji (surrogate pairs), controls, JSON syntax and template tags. */
    private fun randomText(rnd: Random, max: Int): String {
        val pieces = listOf("a", "Z", " ", "\n", "\t", "\"", "\\", "{", "}", "[", "]", ":", ",", "/", "\u0000", "\u001f",
            "Привет", "日本", "😀", " ", "﻿", "<think>", "</think>", "<|im_end|>", "null", "true", "1e999")
        val n = rnd.nextInt(max + 1)
        val sb = StringBuilder()
        while (sb.length < n) {
            sb.append(pieces[rnd.nextInt(pieces.size)])
        }
        return sb.toString()
    }

    /** A random JSON value of the depth: every type, thus a key gets a wrong type often. */
    private fun randomValue(rnd: Random, depth: Int): Any = when (if (depth <= 0) rnd.nextInt(6) else rnd.nextInt(8)) {
        0 -> randomText(rnd, 16)
        1 -> rnd.nextLong()
        2 -> rnd.nextDouble() * 1e6
        3 -> rnd.nextBoolean()
        4 -> JSONObject.NULL
        5 -> rnd.nextInt()
        6 -> JSONArray().apply { repeat(rnd.nextInt(4)) { put(randomValue(rnd, depth - 1)) } }
        else -> JSONObject().apply { repeat(rnd.nextInt(4)) { put(randomText(rnd, 6), randomValue(rnd, depth - 1)) } }
    }

    /** One message object: each key present with the right type, a wrong type, or absent. */
    private fun randomMessage(rnd: Random): JSONObject {
        val obj = JSONObject()
        fun maybe(key: String, right: () -> Any) {
            when (rnd.nextInt(6)) {
                0 -> Unit
                1 -> obj.put(key, randomValue(rnd, 2))
                else -> obj.put(key, right())
            }
        }
        maybe("role") { listOf("user", "assistant", "system")[rnd.nextInt(3)] }
        maybe("content") { randomText(rnd, 40) }
        maybe("thinking") { randomText(rnd, 20) }
        maybe("thinkingMs") { rnd.nextLong() }
        maybe("image") { "%040x.jpg".format(rnd.nextLong() and Long.MAX_VALUE) }
        maybe("meta") { randomText(rnd, 20) }
        if (rnd.nextInt(8) == 0) {
            obj.put(randomText(rnd, 8), randomValue(rnd, 2))
        }
        return obj
    }

    /** The text of a random conversation file: a valid shape, a wrong shape, or bytes that are not JSON. */
    private fun randomFile(rnd: Random): String = when (rnd.nextInt(6)) {
        0 -> randomText(rnd, 200)
        1 -> randomValue(rnd, 4).toString()
        2 -> JSONObject().put("messages", randomValue(rnd, 3)).toString()
        else -> JSONObject().put("messages", JSONArray().apply { repeat(rnd.nextInt(6)) { put(randomMessage(rnd)) } }).toString()
    }

    /** The number of objects in the messages array of the text, or -1 when the text has no such array. */
    private fun messageCount(text: String): Int = try {
        JSONObject(text).getJSONArray("messages").length()
    } catch (e: Exception) {
        -1
    }

    @Test
    fun aRandomFileLoadsAndNeverThrows() {
        val rnd = Random(FuzzSwitch.seed)
        val store = ConversationStore(folder.root)
        repeat(FuzzSwitch.iterations) { i ->
            val text = randomFile(rnd)
            file.writeText(text)
            val loaded = try {
                store.load()
            } catch (t: Throwable) {
                fail("iteration $i: load() threw $t for the file: ${text.take(300)}")
                return
            }
            val count = messageCount(text)
            assertTrue("iteration $i: ${loaded.size} messages loaded from a file of $count", loaded.size <= maxOf(count, 0))
        }
    }

    @Test
    fun aDamagedSavedFileLoadsAndNeverThrows() {
        val rnd = Random(FuzzSwitch.seed + 1)
        val store = ConversationStore(folder.root)
        repeat(FuzzSwitch.iterations) { i ->
            store.save(randomEntries(rnd))
            val bytes = file.readBytes()
            when (rnd.nextInt(3)) {
                0 -> file.writeBytes(bytes.copyOf(rnd.nextInt(bytes.size + 1)))
                1 -> {
                    repeat(1 + rnd.nextInt(4)) { bytes[rnd.nextInt(bytes.size)] = rnd.nextInt(256).toByte() }
                    file.writeBytes(bytes)
                }
                else -> file.writeBytes(bytes + rnd.nextBytes(rnd.nextInt(1, 16)))
            }
            try {
                store.load()
            } catch (t: Throwable) {
                fail("iteration $i: load() threw $t for a damaged file")
            }
        }
    }

    private fun randomEntries(rnd: Random): List<ConversationStore.Entry> {
        val images = List(3) { rnd.nextBytes(rnd.nextInt(1, 64)) }
        return List(rnd.nextInt(0, 8)) {
            ConversationStore.Entry(
                role = listOf("user", "assistant", "system")[rnd.nextInt(3)],
                content = randomText(rnd, 60),
                thinking = randomText(rnd, 30),
                thinkingMs = rnd.nextLong(),
                image = if (rnd.nextInt(3) == 0) images[rnd.nextInt(images.size)] else null,
                meta = if (rnd.nextBoolean()) randomText(rnd, 20).ifEmpty { null } else null,
            )
        }
    }

    @Test
    fun aSaveAndALoadGiveTheSameRandomEntries() {
        val rnd = Random(FuzzSwitch.seed + 2)
        val store = ConversationStore(folder.root)
        repeat(FuzzSwitch.iterations) { i ->
            val entries = randomEntries(rnd)
            store.save(entries)
            val loaded = store.load()
            assertEquals("iteration $i: the count", entries.size, loaded.size)
            for ((a, b) in entries.zip(loaded)) {
                assertEquals("iteration $i: role", a.role, b.role)
                assertEquals("iteration $i: content", a.content, b.content)
                assertEquals("iteration $i: thinking", a.thinking, b.thinking)
                assertEquals("iteration $i: thinkingMs", a.thinkingMs, b.thinkingMs)
                assertEquals("iteration $i: meta", a.meta, b.meta)
                if (a.image == null) assertNull(b.image) else assertArrayEquals("iteration $i: image", a.image, b.image)
            }
            // Only the images that a message holds stay on disk.
            val names = File(folder.root, "images").listFiles().orEmpty().map { it.name }.toSet()
            assertEquals("iteration $i: the image files", entries.mapNotNull { it.image?.toList() }.toSet().size, names.size)
        }
    }

    /**
     * A deeply nested file loads as an empty conversation. The org.json of
     * the tests stops at a nesting depth of 512 with a JSONException. The
     * org.json of Android has no such limit and its parser is recursive, thus
     * there the same file gives a StackOverflowError, which load() catches
     * (finding conversation-deep-nesting, task #94 D9, from the source, not
     * from a run on a phone).
     */
    @Test
    fun aDeeplyNestedFileLoadsAsAnEmptyConversation() {
        val depth = 100_000
        file.writeText("{\"messages\":" + "[".repeat(depth) + "]".repeat(depth) + "}")
        val loaded = try {
            ConversationStore(folder.root).load()
        } catch (t: Throwable) {
            fail("load() threw $t for a file with $depth nested arrays")
            return
        }
        assertTrue(loaded.isEmpty())
    }

    /**
     * An image name that is not a name of save() reads no file (finding
     * conversation-image-path, task #166): "../secret.bin" would read a file
     * outside of the image directory as the image of the message.
     */
    @Test
    fun anImageNameOutsideTheImageDirectoryIsNotRead() {
        File(folder.root, "images").mkdirs()
        File(folder.root, "secret.bin").writeBytes(byteArrayOf(1, 2, 3))
        val msg = JSONObject().put("role", "user").put("content", "x").put("image", "../secret.bin")
        file.writeText(JSONObject().put("messages", JSONArray().put(msg)).toString())
        val loaded = ConversationStore(folder.root).load()
        assertEquals(1, loaded.size)
        assertNull("the image of the message came from outside the image directory", loaded[0].image)
    }

    /**
     * One message without a role or a content does not lose the other
     * messages, and the file stays whole as conversation.json.bad (finding
     * conversation-all-or-nothing, task #94 D9).
     */
    @Test
    fun oneBadMessageDoesNotLoseTheOthers() {
        val good = JSONObject().put("role", "user").put("content", "kept")
        val bad = JSONObject().put("content", "no role")
        val text = JSONObject().put("messages", JSONArray().put(good).put(bad).put(good)).toString()
        file.writeText(text)
        val loaded = ConversationStore(folder.root).load()
        assertEquals(2, loaded.size)
        assertEquals("the file stays", text, file.readText())
        assertEquals("a copy is kept", text, File(folder.root, "conversation.json.bad").readText())
    }

    /**
     * A file that does not parse goes to conversation.json.bad, thus the next
     * save does not write over it. A second damaged file does not overwrite
     * the first one, and the same bytes are kept one time (task #94 D9).
     */
    @Test
    fun aFileThatDoesNotParseIsKeptAndNotOverwritten() {
        val store = ConversationStore(folder.root)
        val bad = File(folder.root, "conversation.json.bad")
        file.writeText("{\"messages\": [")
        assertTrue(store.load().isEmpty())
        assertTrue("the damaged file moved", !file.exists())
        assertEquals("{\"messages\": [", bad.readText())
        store.save(listOf(ConversationStore.Entry("user", "new", "", 0, null, null)))
        assertEquals("the save did not touch the damaged file", "{\"messages\": [", bad.readText())

        file.writeText("not json")
        assertTrue(store.load().isEmpty())
        assertEquals("the first damaged file stays", "{\"messages\": [", bad.readText())
        val others = folder.root.listFiles().orEmpty().filter { it.name.startsWith("conversation.json.bad-") }
        assertEquals(1, others.size)
        assertEquals("not json", others[0].readText())

        file.writeText("not json")
        assertTrue(store.load().isEmpty())
        assertEquals("the same bytes are kept one time", 2,
            folder.root.listFiles().orEmpty().count { it.name.startsWith("conversation.json.bad") })
    }
}
