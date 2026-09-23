package ai.airi.qwenmobile

import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.take
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/** The wake hold of the engine and the answer flow, with a fake system lock and a clock that the test moves. */
class WakeHoldTest {
    /** A system lock that records each call, and the state of a lock that is not reference counted. */
    private class FakeLock : WakeHold.SystemLock {
        val calls = mutableListOf<String>()
        var held = false

        override fun acquire(timeoutMs: Long) {
            calls += "acquire $timeoutMs"
            held = true
        }

        override fun release() {
            calls += "release"
            held = false
        }
    }

    private var now = 0L
    private val lock = FakeLock()
    private val hold = WakeHold(lock, TIMEOUT_MS) { now }

    /** The bytes of one token: the tag byte, then the text. */
    private fun token(text: String, tag: Int = 0): ByteArray = byteArrayOf(tag.toByte()) + text.toByteArray()

    /** A token source of [tokens], then the end token. It checks that the hold is held at each token. */
    private fun source(vararg tokens: ByteArray): (Long) -> ByteArray? {
        var i = 0
        return { h ->
            assertEquals(HANDLE, h)
            assertTrue("the lock is not held during a token", lock.held)
            assertEquals(1, hold.count)
            if (i < tokens.size) tokens[i++] else null
        }
    }

    private fun startOk(): Long {
        assertTrue("the lock is not held during the prompt", lock.held)
        return HANDLE
    }

    @Test
    fun theEndTokenReleasesTheHold() = runBlocking {
        val pieces = answerPieces(hold, 100, ::startOk, source(token("Hel"), token("lo"))).toList()
        assertEquals(listOf("Hel", "lo"), pieces.map { (it as Piece.Text).text })
        assertEquals(listOf("acquire $TIMEOUT_MS", "release"), lock.calls)
        assertFalse(lock.held)
        assertEquals(0, hold.count)
    }

    @Test
    fun theTokenLimitReleasesTheHold() = runBlocking {
        val endless: (Long) -> ByteArray? = { token("x") }
        assertEquals(5, answerPieces(hold, 5, ::startOk, endless).toList().size)
        assertFalse(lock.held)
        assertEquals(0, hold.count)
    }

    @Test
    fun anErrorOfATokenReleasesTheHold() = runBlocking {
        var calls = 0
        val failing: (Long) -> ByteArray? = {
            calls += 1
            if (calls == 2) throw RuntimeException("decode failed")
            token("a")
        }
        try {
            answerPieces(hold, 100, ::startOk, failing).collect()
            fail("the error of the token did not reach the collector")
        } catch (e: RuntimeException) {
            assertEquals("decode failed", e.message)
        }
        assertFalse(lock.held)
        assertEquals(0, hold.count)
    }

    @Test
    fun anErrorOfThePromptReleasesTheHold() = runBlocking {
        val noModel: () -> Long = { throw IllegalStateException("No model is loaded") }
        try {
            answerPieces(hold, 100, noModel, source()).collect()
            fail("the error of the prompt did not reach the collector")
        } catch (e: IllegalStateException) {
            assertEquals("No model is loaded", e.message)
        }
        assertEquals(listOf("acquire $TIMEOUT_MS", "release"), lock.calls)
        assertEquals(0, hold.count)
    }

    @Test
    fun aCancelReleasesTheHold() = runBlocking {
        val endless: (Long) -> ByteArray? = { token("x") }
        var got = 0
        val job = launch {
            answerPieces(hold, 1000, ::startOk, endless).collect {
                got += 1
                if (got == 3) {
                    cancel()
                }
            }
        }
        job.join()
        assertTrue(job.isCancelled)
        assertEquals(3, got)
        assertFalse(lock.held)
        assertEquals(0, hold.count)
    }

    @Test
    fun aCollectorThatStopsEarlyReleasesTheHold() = runBlocking {
        val endless: (Long) -> ByteArray? = { token("x") }
        assertEquals(2, answerPieces(hold, 1000, ::startOk, endless).take(2).toList().size)
        assertFalse(lock.held)
        assertEquals(0, hold.count)
    }

    @Test
    fun theTagsAndTheTextOfATokenGiveTheirPieces() = runBlocking {
        val pieces = answerPieces(hold, 100, ::startOk, source(token("", 1), token("a", 2), token("b"))).toList()
        assertEquals(Piece.ThinkOpen, pieces[0])
        assertEquals("a", (pieces[1] as Piece.Text).text)
        assertEquals(Piece.ThinkClose, pieces[2])
        assertEquals("b", (pieces[3] as Piece.Text).text)
        assertEquals(4, pieces.size)
    }

    @Test
    fun aLongAnswerStartsTheTimeoutAgainAtHalfOfIt() = runBlocking {
        // Each token takes one minute of the fake clock: the hold arms again at 5 of the 10 minutes.
        val slow: (Long) -> ByteArray? = {
            now += 60_000
            token("t")
        }
        answerPieces(hold, 12, ::startOk, slow).collect()
        assertEquals(3, lock.calls.count { it.startsWith("acquire") })
        assertEquals("release", lock.calls.last())
        assertFalse(lock.held)
    }

    @Test
    fun theLastOfTwoHoldersReleasesTheLock() {
        hold.enter()
        hold.enter()
        assertEquals(2, hold.count)
        hold.exit()
        assertTrue("the first exit released the lock of the second holder", lock.held)
        hold.exit()
        assertFalse(lock.held)
        assertEquals(listOf("acquire $TIMEOUT_MS", "acquire $TIMEOUT_MS", "release"), lock.calls)
    }

    @Test
    fun aRenewWithoutAHolderDoesNothing() {
        now = TIMEOUT_MS * 3
        hold.renew()
        assertTrue(lock.calls.isEmpty())
    }

    @Test(expected = IllegalStateException::class)
    fun anExitWithoutAHolderIsAnError() {
        hold.exit()
    }

    @Test
    fun aroundReleasesAfterAnException() {
        try {
            hold.around { throw IllegalArgumentException("load failed") }
        } catch (e: IllegalArgumentException) {
            assertEquals("load failed", e.message)
        }
        assertFalse(lock.held)
        assertEquals(0, hold.count)
    }

    @Test
    fun aNullHoldRunsTheBlockAndTheFlow() = runBlocking {
        val none: WakeHold? = null
        assertEquals(7, none.around { 7 })
        assertEquals(2, answerPieces(none, 100, { HANDLE }, source2()).toList().size)
    }

    /** Two tokens without the checks of the hold, for a flow without a hold. */
    private fun source2(): (Long) -> ByteArray? {
        var i = 0
        return { if (i++ < 2) token("z") else null }
    }

    private companion object {
        const val TIMEOUT_MS = 600_000L
        const val HANDLE = 42L
    }
}
