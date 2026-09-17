package ai.airi.qwenmobile

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/** The conversation file and the image directory in a temporary directory. */
class ConversationStoreTest {
    @get:Rule
    val folder = TemporaryFolder()

    private val image = byteArrayOf(1, 2, 3, 4, 5)

    private fun entries(): List<ConversationStore.Entry> {
        val user = ChatMessage("user", "What is in the picture?", image)
        val answer = ChatMessage("assistant", "A cat.")
        answer.thinking = "Fur, whiskers."
        answer.thinkingMs = 1_234
        return listOf(
            ConversationStore.Entry.of(user, null),
            ConversationStore.Entry.of(answer, "12.3 t/s"),
        )
    }

    @Test
    fun aSaveAndALoadGiveTheSameMessages() {
        val store = ConversationStore(folder.root)
        store.save(entries())
        val loaded = store.load()
        assertEquals(2, loaded.size)

        val user = loaded[0]
        assertEquals("user", user.role)
        assertEquals("What is in the picture?", user.content)
        assertArrayEquals(image, user.image)
        assertNull(user.meta)
        assertEquals("", user.thinking)

        val answer = loaded[1]
        assertEquals("assistant", answer.role)
        assertEquals("A cat.", answer.content)
        assertEquals("Fur, whiskers.", answer.thinking)
        assertEquals(1_234L, answer.thinkingMs)
        assertEquals("12.3 t/s", answer.meta)
        assertNull(answer.image)

        val message = answer.toMessage()
        assertEquals(ChatMessage.Phase.DONE, message.phase)
        assertEquals("Fur, whiskers.", message.thinking)
    }

    @Test
    fun aSaveLeavesNoTemporaryFile() {
        val store = ConversationStore(folder.root)
        store.save(entries())
        assertTrue(File(folder.root, "conversation.json").isFile)
        assertFalse(File(folder.root, "conversation.json.tmp").exists())
    }

    @Test
    fun anImageWithoutAMessageIsRemovedByTheNextSave() {
        val store = ConversationStore(folder.root)
        store.save(entries())
        val images = File(folder.root, "images").listFiles().orEmpty()
        assertEquals(1, images.size)
        assertTrue(images[0].name.endsWith(".jpg"))

        store.save(listOf(ConversationStore.Entry.of(ChatMessage("user", "text only"), null)))
        assertEquals(0, File(folder.root, "images").listFiles().orEmpty().size)
    }

    @Test
    fun aClearRemovesTheFileAndTheImages() {
        val store = ConversationStore(folder.root)
        store.save(entries())
        store.clear()
        assertTrue(store.load().isEmpty())
        assertEquals(0, File(folder.root, "images").listFiles().orEmpty().size)
    }

    @Test
    fun aMissingFileLoadsAsAnEmptyConversation() {
        assertTrue(ConversationStore(folder.root).load().isEmpty())
    }

    @Test
    fun aDamagedFileLoadsAsAnEmptyConversation() {
        File(folder.root, "conversation.json").writeText("{ not json")
        assertTrue(ConversationStore(folder.root).load().isEmpty())
    }

    @Test
    fun aMissingImageFileLoadsAsAMessageWithoutAnImage() {
        val store = ConversationStore(folder.root)
        store.save(entries())
        File(folder.root, "images").listFiles().orEmpty().forEach { it.delete() }
        val loaded = store.load()
        assertEquals(2, loaded.size)
        assertNull(loaded[0].image)
        assertEquals("What is in the picture?", loaded[0].content)
    }

    @Test
    fun theSameImageIsWrittenOneTime() {
        val store = ConversationStore(folder.root)
        val first = ChatMessage("user", "one", image)
        val second = ChatMessage("user", "two", image.copyOf())
        store.save(listOf(ConversationStore.Entry.of(first, null), ConversationStore.Entry.of(second, null)))
        assertEquals(1, File(folder.root, "images").listFiles().orEmpty().size)
        val loaded = store.load()
        assertArrayEquals(image, loaded[0].image)
        assertArrayEquals(image, loaded[1].image)
    }
}
