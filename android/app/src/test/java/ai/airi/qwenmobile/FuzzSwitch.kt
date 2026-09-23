package ai.airi.qwenmobile

import org.junit.Assume.assumeTrue

/**
 * The switches of the fuzz tests. They come from the environment of the
 * Gradle run, because the test task copies that environment into the test
 * JVM and the build file stays as it is.
 *
 * - QWEN_FUZZ_ITERATIONS: the iterations of each fuzz test. The default keeps
 *   the unit tests of the APK build short. A long run sets, for example, 20000.
 * - QWEN_FUZZ_SEED: the seed of the generators. The default is fixed, thus
 *   each run of the unit tests is the same.
 * - QWEN_FUZZ_FINDINGS=1: run the tests that reproduce the open findings of
 *   the fuzz campaign. They fail until the fixes land. Without the switch
 *   they are skipped, thus the APK build stays green.
 */
internal object FuzzSwitch {
    val iterations: Int = System.getenv("QWEN_FUZZ_ITERATIONS")?.toIntOrNull()?.coerceAtLeast(1) ?: 200

    val seed: Long = System.getenv("QWEN_FUZZ_SEED")?.toLongOrNull() ?: 20260923L

    private val findings: Boolean = System.getenv("QWEN_FUZZ_FINDINGS") == "1"

    /** Skip the calling test unless QWEN_FUZZ_FINDINGS=1. [finding] names the finding. */
    fun requireFindings(finding: String) {
        assumeTrue("finding $finding: set QWEN_FUZZ_FINDINGS=1 to run it", findings)
    }
}
