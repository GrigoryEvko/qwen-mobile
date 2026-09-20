package ai.airi.qwenmobile

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** The phase machine of a streamed answer, with a clock that the test moves. */
class AnswerStreamTest {
    private var now = 1_000L

    private fun stream(thinkingMode: Boolean): AnswerStream {
        val answer = ChatMessage("assistant", "")
        answer.phase = if (thinkingMode) ChatMessage.Phase.THINKING else ChatMessage.Phase.ANSWERING
        return AnswerStream(answer) { now }
    }

    @Test
    fun thinkingModeSplitsTheThinkingFromTheAnswer() {
        val s = stream(thinkingMode = true)
        s.markStart()
        s.accept(Piece.Text("Let me"))
        s.accept(Piece.Text(" think."))
        assertEquals(ChatMessage.Phase.THINKING, s.answer.phase)
        assertEquals("Let me think.", s.answer.thinking)
        assertFalse(s.answer.hasContent)

        now += 2_500
        s.accept(Piece.ThinkClose)
        assertEquals(ChatMessage.Phase.ANSWERING, s.answer.phase)
        assertEquals(2_500L, s.answer.thinkingMs)

        s.accept(Piece.Text("\n\n  Hello"))
        s.accept(Piece.Text(" world"))
        s.finish(error = null, cancelled = false)
        assertEquals(ChatMessage.Phase.DONE, s.answer.phase)
        assertEquals("Hello world", s.answer.content)
        assertEquals("Let me think.", s.answer.thinking)
    }

    @Test
    fun theOpenTagMovesAnAnswerIntoTheThinking() {
        val s = stream(thinkingMode = false)
        s.accept(Piece.ThinkOpen)
        assertEquals(ChatMessage.Phase.THINKING, s.answer.phase)
        s.accept(Piece.Text("why"))
        now += 1_000
        s.accept(Piece.ThinkClose)
        s.accept(Piece.Text("because"))
        s.finish(error = null, cancelled = false)
        assertEquals("why", s.answer.thinking)
        assertEquals("because", s.answer.content)
        assertEquals(1_000L, s.answer.thinkingMs)
        assertEquals(ChatMessage.Phase.DONE, s.answer.phase)
    }

    @Test
    fun aStopInsideTheThinkingLeavesAnInterruptedMessage() {
        val s = stream(thinkingMode = true)
        s.markStart()
        s.accept(Piece.Text("half a thought"))
        now += 700
        s.finish(error = null, cancelled = true)
        assertEquals(ChatMessage.Phase.INTERRUPTED, s.answer.phase)
        assertEquals(700L, s.answer.thinkingMs)
        assertEquals("half a thought", s.answer.thinking)
        assertFalse(s.answer.hasContent)
    }

    @Test
    fun anErrorInsideTheThinkingShowsTheError() {
        // The message of a failure reaches the user whatever phase the answer
        // stopped in: the errors that matter most, the full context and the
        // long conversation, both arrive while the thinking is open.
        val s = stream(thinkingMode = true)
        s.accept(Piece.Text("thought"))
        s.finish(error = "The context is full (8192 tokens). Start a new chat.", cancelled = false)
        assertEquals(ChatMessage.Phase.FAILED, s.answer.phase)
        assertEquals("The context is full (8192 tokens). Start a new chat.", s.answer.content)
        assertEquals("thought", s.answer.thinking)
    }

    @Test
    fun aThinkTagAfterTheAnswerTextIsText() {
        // One stray tag must not move a finished answer back into the thinking:
        // that phase would end the turn as interrupted and drop its text.
        val s = stream(thinkingMode = false)
        s.accept(Piece.Text("The answer is 4."))
        s.accept(Piece.ThinkOpen)
        s.accept(Piece.Text(" And more."))
        s.finish(error = null, cancelled = false)
        assertEquals(ChatMessage.Phase.DONE, s.answer.phase)
        assertEquals("The answer is 4.<think> And more.", s.answer.content)
    }

    @Test
    fun aStopAfterSomeAnswerTextKeepsTheText() {
        val s = stream(thinkingMode = true)
        s.markStart()
        s.accept(Piece.Text("thought"))
        s.accept(Piece.ThinkClose)
        s.accept(Piece.Text("the answer"))
        s.finish(error = null, cancelled = true)
        assertEquals(ChatMessage.Phase.DONE, s.answer.phase)
        assertEquals("the answer", s.answer.content)
    }

    @Test
    fun anErrorBeforeAnyTextIsAFailedMessageWithTheErrorAsText() {
        val s = stream(thinkingMode = false)
        s.finish(error = "Select a model in Settings.", cancelled = false)
        assertEquals(ChatMessage.Phase.FAILED, s.answer.phase)
        assertEquals("Select a model in Settings.", s.answer.content)
    }

    @Test
    fun anErrorAfterTextKeepsThePartialAnswer() {
        val s = stream(thinkingMode = false)
        s.accept(Piece.Text("partial"))
        s.finish(error = "boom", cancelled = false)
        assertEquals(ChatMessage.Phase.DONE, s.answer.phase)
        assertEquals("partial", s.answer.content)
    }

    @Test
    fun aStopBeforeAnyTextIsAnInterruptedMessage() {
        val s = stream(thinkingMode = false)
        s.finish(error = null, cancelled = true)
        assertEquals(ChatMessage.Phase.INTERRUPTED, s.answer.phase)
        assertFalse(s.answer.hasContent)
    }

    @Test
    fun anEmptyAnswerFromTheModelIsDone() {
        val s = stream(thinkingMode = false)
        s.finish(error = null, cancelled = false)
        assertEquals(ChatMessage.Phase.DONE, s.answer.phase)
        assertEquals("", s.answer.content)
    }

    @Test
    fun theLeadingBlankPiecesOfTheAnswerAreDropped() {
        val s = stream(thinkingMode = false)
        s.accept(Piece.Text("\n"))
        s.accept(Piece.Text("  \n"))
        s.accept(Piece.Text(" text"))
        s.accept(Piece.Text(" \n more"))
        assertEquals("text \n more", s.answer.content)
    }

    @Test
    fun thePhaseChangesResetTheChoiceOfTheUserOnTheThinkingField() {
        val s = stream(thinkingMode = true)
        s.answer.thinkingExpanded = true
        s.accept(Piece.ThinkClose)
        assertNull(s.answer.thinkingExpanded)
        s.answer.thinkingExpanded = false
        s.accept(Piece.ThinkOpen)
        assertNull(s.answer.thinkingExpanded)
        s.answer.thinkingExpanded = true
        s.finish(error = null, cancelled = true)
        assertNull(s.answer.thinkingExpanded)
    }

    @Test
    fun theHeaderFollowsThePhase() {
        val m = ChatMessage("assistant", "")
        m.phase = ChatMessage.Phase.THINKING
        assertEquals(Thinking.Label.IN_PROGRESS, Thinking.labelOf(m))
        assertTrue(Thinking.isExpanded(m))

        m.phase = ChatMessage.Phase.INTERRUPTED
        assertEquals(Thinking.Label.INTERRUPTED, Thinking.labelOf(m))
        assertFalse(Thinking.isExpanded(m))

        m.phase = ChatMessage.Phase.DONE
        assertEquals(Thinking.Label.DONE, Thinking.labelOf(m))
        m.thinkingMs = 400
        assertEquals(Thinking.Label.DONE_WITH_SECONDS, Thinking.labelOf(m))
        assertEquals(1L, Thinking.seconds(m))
        m.thinkingMs = 12_900
        assertEquals(12L, Thinking.seconds(m))

        m.thinkingExpanded = true
        assertTrue(Thinking.isExpanded(m))
    }

    @Test
    fun theTextGrowsByPiecesAndReadsBackAsOneString() {
        val m = ChatMessage("assistant", "start")
        m.appendContent(" middle")
        assertEquals("start middle", m.content)
        m.appendContent(" end")
        assertEquals("start middle end", m.content)
        m.content = "new"
        assertEquals("new", m.content)
        assertTrue(m.hasContent)
        m.appendThinking("t")
        assertTrue(m.hasThinking)
        assertEquals("t", m.thinking)
    }
}
