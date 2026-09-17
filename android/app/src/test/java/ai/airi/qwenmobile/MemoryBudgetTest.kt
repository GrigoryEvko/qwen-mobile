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
