package ai.airi.qwenmobile

/**
 * The switches of the fuzz tests. They come from the environment of the
 * Gradle run, because the test task copies that environment into the test
 * JVM and the build file stays as it is.
 *
 * - QWEN_FUZZ_ITERATIONS: the iterations of each fuzz test. The default keeps
 *   the unit tests of the APK build short. A long run sets, for example, 20000.
 * - QWEN_FUZZ_SEED: the seed of the generators. The default is fixed, thus
 *   each run of the unit tests is the same.
 */
internal object FuzzSwitch {
    val iterations: Int = System.getenv("QWEN_FUZZ_ITERATIONS")?.toIntOrNull()?.coerceAtLeast(1) ?: 200

    val seed: Long = System.getenv("QWEN_FUZZ_SEED")?.toLongOrNull() ?: 20260923L
}
