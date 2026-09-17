package ai.airi.qwenmobile

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** The split of raw answers into the thinking and the visible answer. */
class ThinkingTest {
    @Test
    fun plainAnswerHasNoThinking() {
        val parts = Thinking.split("The answer is 42.")
        assertNull(parts.thinking)
        assertEquals("The answer is 42.", parts.answer)
        assertTrue(parts.thinkingDone)
    }

    @Test
    fun completeThinkingSplitsFromTheAnswer() {
        val parts = Thinking.split("<think>\nLet me count.\n</think>\n\n**42**")
        assertEquals("Let me count.", parts.thinking)
        assertEquals("**42**", parts.answer)
        assertTrue(parts.thinkingDone)
    }

    @Test
    fun openThinkingIsInProgress() {
        val parts = Thinking.split("<think>\nLet me")
        assertEquals("Let me", parts.thinking)
        assertEquals("", parts.answer)
        assertFalse(parts.thinkingDone)
    }

    @Test
    fun emptyThinkingCountsAsNone() {
        val parts = Thinking.split("<think>\n\n</think>\n\nHello")
        assertNull(parts.thinking)
        assertEquals("Hello", parts.answer)
        assertTrue(parts.thinkingDone)
    }

    @Test
    fun whitespaceBeforeTheTagIsPermitted() {
        val parts = Thinking.split("\n<think>a</think>b")
        assertEquals("a", parts.thinking)
        assertEquals("b", parts.answer)
    }

    @Test
    fun aTagInsideTheTextIsNotThinking() {
        val parts = Thinking.split("Use <think> as a tag.")
        assertNull(parts.thinking)
        assertEquals("Use <think> as a tag.", parts.answer)
    }

    @Test
    fun emptyTextIsAnEmptyAnswer() {
        val parts = Thinking.split("")
        assertNull(parts.thinking)
        assertEquals("", parts.answer)
        assertTrue(parts.thinkingDone)
    }
}
