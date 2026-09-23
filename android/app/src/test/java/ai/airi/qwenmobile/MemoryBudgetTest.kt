package ai.airi.qwenmobile

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

/** The arithmetic of the memory check, with files of a known size. */
class MemoryBudgetTest {
    @get:Rule
    val folder = TemporaryFolder()

    private val gigabyte = 1_000_000_000L

    @Test
    fun theLoadFitsWhenTheWeightsAndTheMarginAreAvailable() {
        val weights = 2 * gigabyte
        val exact = weights + MemoryBudget.RUNTIME_BYTES
        assertTrue(MemoryBudget.fits(weights, exact, 0L))
        assertFalse(MemoryBudget.fits(weights, exact - 1, 0L))
    }

    @Test
    fun theLoadedModelCountsAsFree() {
        val weights = 2 * gigabyte
        val available = MemoryBudget.RUNTIME_BYTES + gigabyte
        assertFalse(MemoryBudget.fits(weights, available, 0L))
        assertTrue(MemoryBudget.fits(weights, available, gigabyte))
    }

    @Test
    fun theHybridBackendHoldsTheWeightsTwice() {
        val model = folder.newFile("model.gguf").also { it.writeBytes(ByteArray(1_000)) }
        val projector = folder.newFile("model.mmproj.gguf").also { it.writeBytes(ByteArray(100)) }
        val cpu = EngineConfig(model.absolutePath, Backend.CPU, 4, 8192, projector.absolutePath)
        val gpu = EngineConfig(model.absolutePath, Backend.GPU, 4, 8192, null)
        val hybrid = EngineConfig(model.absolutePath, Backend.HYBRID, 4, 8192, projector.absolutePath)
        assertEquals(1_100L, MemoryBudget.weightBytes(cpu))
        assertEquals(1_000L, MemoryBudget.weightBytes(gpu))
        assertEquals(2_100L, MemoryBudget.weightBytes(hybrid))
    }

    @Test
    fun speculativeDecodingCountsTheDraftContextAndTheStateSnapshots() {
        val model = folder.newFile("mtp.gguf").also { it.writeBytes(ByteArray(1_000)) }
        val plain = EngineConfig(model.absolutePath, Backend.GPU, 4, 8192)
        val draft = plain.copy(speculative = true)
        assertEquals(0L, MemoryBudget.speculativeBytes(plain))
        assertEquals(1_000L, MemoryBudget.loadBytes(plain))
        // 8192 positions of 4 KB, the compute buffer of the draft context, and the four recurrent state snapshots.
        val expected = 8_192L * MemoryBudget.DRAFT_KV_BYTES_PER_POSITION + MemoryBudget.DRAFT_COMPUTE_BYTES +
            MemoryBudget.DRAFT_STATE_BYTES
        assertEquals(expected, MemoryBudget.speculativeBytes(draft))
        assertEquals(1_000L + expected, MemoryBudget.loadBytes(draft))
        // A longer context needs a longer draft cache.
        assertEquals(
            expected + 8_192L * MemoryBudget.DRAFT_KV_BYTES_PER_POSITION,
            MemoryBudget.speculativeBytes(draft.copy(nCtx = 16384)),
        )
    }

    @Test
    fun theHybridBackendDraftsNothing() {
        val model = folder.newFile("hybrid.gguf").also { it.writeBytes(ByteArray(1_000)) }
        // The prompt of the hybrid backend runs on a second model, thus the draft block cannot follow it.
        val hybrid = EngineConfig(model.absolutePath, Backend.HYBRID, 4, 8192, speculative = true)
        assertFalse(hybrid.speculativeReady)
        assertEquals(0L, MemoryBudget.speculativeBytes(hybrid))
        assertEquals(2_000L + MemoryBudget.HYBRID_CONTEXT_BYTES, MemoryBudget.loadBytes(hybrid))
    }

    @Test
    fun onlyTheHybridBackendCountsASecondContext() {
        val model = folder.newFile("second.gguf").also { it.writeBytes(ByteArray(1_000)) }
        val gpu = EngineConfig(model.absolutePath, Backend.GPU, 4, 8192)
        val hybrid = EngineConfig(model.absolutePath, Backend.HYBRID, 4, 8192)
        assertEquals(0L, MemoryBudget.hybridContextBytes(gpu))
        assertEquals(MemoryBudget.HYBRID_CONTEXT_BYTES, MemoryBudget.hybridContextBytes(hybrid))
        assertEquals(1_000L, MemoryBudget.loadBytes(gpu))
    }

    @Test
    fun aMissingFileWeighsNothing() {
        val config = EngineConfig("/no/such/model.gguf", Backend.GPU, 4, 8192, "/no/such/projector.gguf")
        assertEquals(0L, MemoryBudget.weightBytes(config))
    }

    @Test
    fun memAvailableIsReadInKilobytes() {
        val lines = sequenceOf("MemTotal:       11895400 kB", "MemFree:          213000 kB", "MemAvailable:    4321000 kB")
        assertEquals(4_321_000L * 1024, MemoryBudget.parseMemAvailable(lines))
        assertNull(MemoryBudget.parseMemAvailable(sequenceOf("MemTotal:       11895400 kB")))
        assertNull(MemoryBudget.parseMemAvailable(sequenceOf("MemAvailable: none")))
    }

    @Test
    fun theFormatIsOneDecimalInGigabytes() {
        assertEquals("7.3 GB", MemoryBudget.format(7_300_000_000L))
        assertEquals("0.0 GB", MemoryBudget.format(0L))
        assertEquals("1.2 GB", MemoryBudget.format(1_249_999_999L))
    }
}
