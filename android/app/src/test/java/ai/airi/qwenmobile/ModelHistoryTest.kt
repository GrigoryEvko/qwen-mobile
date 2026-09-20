package ai.airi.qwenmobile

import org.junit.Assert.assertEquals
import org.junit.Test

/** The messages that reach the chat template after interrupted, failed and restored turns. */
class ModelHistoryTest {
    private fun user(text: String) = ChatMessage("user", text)

    private fun answer(text: String, phase: ChatMessage.Phase = ChatMessage.Phase.DONE): ChatMessage {
        val m = ChatMessage("assistant", text)
        m.phase = phase
        return m
    }

    private fun roles(messages: List<ChatMessage>) = modelHistory(messages).map { "${it.role}:${it.content}" }

    @Test
    fun completedExchangesAndTheCurrentUserMessageStay() {
        val messages = listOf(user("q1"), answer("a1"), user("q2"), answer("a2"), user("q3"))
        assertEquals(listOf("user:q1", "assistant:a1", "user:q2", "assistant:a2", "user:q3"), roles(messages))
    }

    @Test
    fun anInterruptedAnswerTakesItsUserMessageWithIt() {
        val messages = listOf(user("q1"), answer("", ChatMessage.Phase.INTERRUPTED), user("q2"))
        assertEquals(listOf("user:q2"), roles(messages))
    }

    @Test
    fun aFailedAnswerTakesItsUserMessageWithIt() {
        val messages = listOf(user("q1"), answer("a1"), user("q2"), answer("boom", ChatMessage.Phase.FAILED), user("q3"))
        assertEquals(listOf("user:q1", "assistant:a1", "user:q3"), roles(messages))
    }

    @Test
    fun aUserMessageWithoutAnAnswerAfterARestartGoes() {
        val messages = listOf(user("q1"), user("q2"), answer("a2"), user("q3"))
        assertEquals(listOf("user:q2", "assistant:a2", "user:q3"), roles(messages))
    }

    @Test
    fun aStoppedAnswerWithTextIsAnExchange() {
        val messages = listOf(user("q1"), answer("partial"), user("q2"))
        assertEquals(listOf("user:q1", "assistant:partial", "user:q2"), roles(messages))
    }

    @Test
    fun aSystemMessageStays() {
        val messages = listOf(ChatMessage("system", "be brief"), user("q1"), answer("", ChatMessage.Phase.INTERRUPTED), user("q2"))
        assertEquals(listOf("system:be brief", "user:q2"), roles(messages))
    }

    @Test
    fun aSystemPromptLeadsTheHistory() {
        val messages = listOf(user("hello"))
        val out = modelHistory(messages, "You answer in Russian.")
        assertEquals(2, out.size)
        assertEquals("system", out[0].role)
        assertEquals("You answer in Russian.", out[0].content)
        assertEquals("user", out[1].role)
    }

    @Test
    fun aBlankSystemPromptSendsNoSystemTurn() {
        // An empty field must not send an empty system turn: that would cost
        // tokens and change the behaviour of the model for no reason.
        for (blank in listOf("", "   ", "\n")) {
            val out = modelHistory(listOf(user("hello")), blank)
            assertEquals(1, out.size)
            assertEquals("user", out[0].role)
        }
    }

    @Test
    fun theSystemPromptIsNotPartOfTheStoredMessages() {
        // modelHistory is pure: the caller's list is untouched, thus the
        // conversation store never persists the prompt.
        val messages = listOf(user("hello"))
        modelHistory(messages, "Be brief.")
        assertEquals(1, messages.size)
        assertEquals("user", messages[0].role)
    }
}
