package ai.airi.qwenmobile

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.random.Random

/**
 * The fuzz tests of the pure parts of the chat session: the history that
 * goes to the model (modelHistory in ChatSession.kt), the phase machine of a
 * streamed answer (AnswerStream), and the round trip of a finished message
 * through the conversation store entry. The sequences are random with a
 * fixed seed. Refer to [FuzzSwitch].
 */
class ChatSessionFuzzTest {
    private val phases = ChatMessage.Phase.values()

    private fun randomMessage(rnd: Random): ChatMessage {
        val role = listOf("user", "user", "assistant", "assistant", "system", "tool")[rnd.nextInt(6)]
        val m = ChatMessage(role, "m${rnd.nextInt(1000)}", if (rnd.nextInt(5) == 0) byteArrayOf(1) else null)
        m.phase = if (role == "assistant") phases[rnd.nextInt(phases.size)] else ChatMessage.Phase.DONE
        return m
    }

    /**
     * The properties of the history: the system prompt leads it when it is
     * not blank, the order stays, no two user messages follow each other, an
     * assistant message is DONE, and the last message of the list stays.
     */
    @Test
    fun theHistoryOfARandomConversationKeepsItsProperties() {
        val rnd = Random(FuzzSwitch.seed + 20)
        repeat(FuzzSwitch.iterations * 5) { i ->
            val messages = List(rnd.nextInt(0, 12)) { randomMessage(rnd) }
            val system = listOf("", "  ", "Be brief.")[rnd.nextInt(3)]
            val out = modelHistory(messages, system)
            val body = if (system.isNotBlank()) {
                assertEquals("iteration $i: the system prompt leads", "system", out.first().role)
                assertEquals("iteration $i", system, out.first().content)
                out.drop(1)
            } else {
                out
            }
            // The order stays: the body is a subsequence of the messages, by identity.
            var at = 0
            for (m in body) {
                while (at < messages.size && messages[at] !== m) at++
                assertTrue("iteration $i: the history changed the order", at < messages.size)
                at++
            }
            for ((a, b) in body.zipWithNext()) {
                assertFalse("iteration $i: two user messages follow each other", a.role == "user" && b.role == "user")
            }
            for (m in body) {
                assertTrue("iteration $i: an assistant message that is not DONE", m.role != "assistant" || m.phase == ChatMessage.Phase.DONE)
            }
            val last = messages.lastOrNull()
            if (last != null && last.role != "assistant") {
                assertTrue("iteration $i: the last message went", body.lastOrNull() === last)
            }
        }
    }

    /**
     * The properties of a streamed answer after its end: the phase is DONE,
     * FAILED or INTERRUPTED; an answer with text is DONE; the text starts
     * without a blank; a failed answer shows the error; the thinking time is
     * not negative.
     */
    @Test
    fun aRandomStreamEndsInAConsistentPhase() {
        val rnd = Random(FuzzSwitch.seed + 21)
        repeat(FuzzSwitch.iterations * 5) { i ->
            var now = 1_000L
            val answer = ChatMessage("assistant", "")
            answer.phase = if (rnd.nextBoolean()) ChatMessage.Phase.THINKING else ChatMessage.Phase.ANSWERING
            val stream = AnswerStream(answer) { now }
            stream.markStart()
            repeat(rnd.nextInt(0, 20)) {
                now += rnd.nextLong(0, 500)
                stream.accept(
                    when (rnd.nextInt(6)) {
                        0 -> Piece.ThinkOpen
                        1 -> Piece.ThinkClose
                        2 -> Piece.Text(listOf(" ", "\n", "\t ")[rnd.nextInt(3)])
                        else -> Piece.Text(listOf("a", " b", "\nc", "Привет", "😀")[rnd.nextInt(5)])
                    },
                )
            }
            val error = if (rnd.nextInt(4) == 0) "boom" else null
            val cancelled = rnd.nextInt(4) == 0
            val hadText = answer.hasContent
            stream.finish(error, cancelled)
            val p = answer.phase
            assertTrue("iteration $i: phase $p", p == ChatMessage.Phase.DONE || p == ChatMessage.Phase.FAILED || p == ChatMessage.Phase.INTERRUPTED)
            if (hadText) {
                assertEquals("iteration $i: an answer with text is DONE", ChatMessage.Phase.DONE, p)
                assertFalse("iteration $i: the text starts with a blank", answer.content[0].isWhitespace())
            }
            if (p == ChatMessage.Phase.FAILED) {
                assertEquals("iteration $i: a failed answer shows its error", error, answer.content)
            }
            assertTrue("iteration $i: thinking time ${answer.thinkingMs}", answer.thinkingMs >= 0)
        }
    }

    /** A finished message goes into a store entry and back with its fields. */
    @Test
    fun aMessageSurvivesTheEntryRoundTrip() {
        val rnd = Random(FuzzSwitch.seed + 22)
        repeat(FuzzSwitch.iterations) { i ->
            val m = randomMessage(rnd)
            m.thinking = "t${rnd.nextInt()}"
            m.thinkingMs = rnd.nextLong(0, Long.MAX_VALUE)
            val back = ConversationStore.Entry.of(m, "meta").toMessage()
            assertEquals("iteration $i", m.role, back.role)
            assertEquals("iteration $i", m.content, back.content)
            assertEquals("iteration $i", m.thinking, back.thinking)
            assertEquals("iteration $i", m.thinkingMs, back.thinkingMs)
            assertEquals("iteration $i: a restored message is DONE", ChatMessage.Phase.DONE, back.phase)
        }
    }
}
