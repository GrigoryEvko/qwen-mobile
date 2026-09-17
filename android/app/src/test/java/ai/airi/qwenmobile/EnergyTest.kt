package ai.airi.qwenmobile

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** The arithmetic of the energy measurement, with samples of a known value. */
class EnergyTest {
    /** A sample at [ms] with a current of [mA] milliamperes and 3800 mV. */
    private fun sample(ms: Long, mA: Long, mv: Int = 3800): PowerSample =
        PowerSample(ms, -mA * 1000, mv)

    @Test
    fun theSignOfTheCurrentDoesNotChangeThePower() {
        // 2 A at 3.8 V is 7.6 W, with each sign convention.
        assertEquals(7.6, PowerSample(0, -2_000_000, 3800).watts, 1e-9)
        assertEquals(7.6, PowerSample(0, 2_000_000, 3800).watts, 1e-9)
    }

    @Test
    fun aSampleWithoutValuesIsNotUsable() {
        assertFalse(PowerSample(0, PowerSample.INVALID, 3800).valid)
        assertFalse(PowerSample(0, -2_000_000, 0).valid)
        assertFalse(PowerSample(0, 0, 3800).valid)
        assertTrue(PowerSample(0, -2_000_000, 3800).valid)
        // The trace keeps only the usable samples.
        val trace = PowerTrace()
        trace.add(PowerSample(0, PowerSample.INVALID, 3800))
        trace.add(sample(0, 1000))
        assertEquals(1, trace.size)
    }

    @Test
    fun aConstantCurrentGivesTheEnergyOfTheWindow() {
        val trace = PowerTrace()
        // 4 samples at 4 Hz: 1 A at 3.8 V over 750 ms is 2.85 J.
        for (i in 0..3) {
            trace.add(sample(i * 250L, 1000))
        }
        assertEquals(750L, trace.durationMs())
        assertEquals(3.8, trace.meanPowerW(), 1e-9)
        assertEquals(2.85, trace.energyJoules(), 1e-9)
        assertEquals(3800.0, trace.meanVoltageMv(), 1e-9)
        assertEquals(-1_000_000.0, trace.meanCurrentUa(), 1e-9)
    }

    @Test
    fun theTrapezoidRuleFollowsAStepOfTheCurrent() {
        val trace = PowerTrace()
        // 1 s at 1 A, then 1 s at 3 A, at 4 V: 4 W and 12 W give 4 + 8 = 12 J.
        trace.add(sample(0, 1000, 4000))
        trace.add(sample(1000, 1000, 4000))
        trace.add(sample(2000, 3000, 4000))
        assertEquals(2000L, trace.durationMs())
        assertEquals(12.0, trace.energyJoules(), 1e-9)
        assertEquals(6.0, trace.meanPowerW(), 1e-9)
    }

    @Test
    fun theEnergyOfOneTokenSubtractsTheIdlePower() {
        val trace = PowerTrace()
        // 10 s at 5 W is 50 J. With 128 tokens that is 390.6 mJ for each token.
        trace.add(PowerSample(0, -1_250_000, 4000))
        trace.add(PowerSample(10_000, -1_250_000, 4000))
        assertEquals(50.0, trace.energyJoules(), 1e-9)
        assertEquals(390.625, trace.energyPerTokenMj(128), 1e-6)
        // The phone draws 1 W without an answer, thus the answer costs 4 W.
        assertEquals(312.5, trace.netEnergyPerTokenMj(1.0, 128), 1e-6)
        assertEquals(0.0, trace.energyPerTokenMj(0), 1e-9)
        assertEquals(0.0, trace.netEnergyPerTokenMj(1.0, 0), 1e-9)
    }

    @Test
    fun anEmptyWindowGivesNoEnergy() {
        val trace = PowerTrace()
        assertTrue(trace.empty)
        assertEquals(0L, trace.durationMs())
        assertEquals(0.0, trace.energyJoules(), 1e-9)
        assertEquals(0.0, trace.meanPowerW(), 1e-9)
        assertEquals(0.0, trace.meanVoltageMv(), 1e-9)
        // One sample has no duration, thus the mean power is the power of that sample.
        trace.add(sample(0, 2000, 4000))
        assertEquals(8.0, trace.meanPowerW(), 1e-9)
        assertEquals(0.0, trace.energyJoules(), 1e-9)
    }

    @Test
    fun theChargeCounterGivesTheSameEnergyAsAConstantCurrent() {
        // 2 A for 30 s is 16.67 mAh. At 3.75 V that is 225 J.
        val uah = (2_000_000.0 * 30.0 / 3600.0).toLong()
        assertEquals(225.0, Energy.chargeEnergyJoules(uah, 3750.0), 0.1)
        // The counter falls while the battery supplies the load, thus the sign does not count.
        assertEquals(225.0, Energy.chargeEnergyJoules(-uah, 3750.0), 0.1)
        assertEquals(1757.8, Energy.perTokenMj(225.0, 128), 0.1)
        assertEquals(0.0, Energy.perTokenMj(225.0, 0), 1e-9)
    }

    @Test
    fun theReportFormatHasOneDecimal() {
        assertEquals("5.9 W", Energy.format(5.94, "W"))
        assertEquals("467.0 mJ", Energy.format(467.0, "mJ"))
    }
}
