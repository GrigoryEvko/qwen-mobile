package ai.airi.qwenmobile

import android.content.SharedPreferences
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.lang.reflect.InvocationTargetException
import kotlin.random.Random

/**
 * The fuzz tests of the load path of [SettingsStore]: the read of the
 * preferences file with random values, values of a wrong type, and values
 * out of range. The file comes from the app, and also from a hand edit of
 * shared_prefs/settings.xml through run-as, which the measurement procedure
 * of the team uses.
 *
 * The store starts a coroutine on the main dispatcher in its constructor,
 * and the JVM has no main looper. Thus the test makes the object without its
 * constructor and calls its private read() through reflection.
 */
class SettingsFuzzTest {
    /**
     * SharedPreferences with the casts of the Android implementation
     * (SharedPreferencesImpl): a value of a wrong type throws ClassCastException.
     */
    private class FakePrefs(private val map: Map<String, Any?>) : SharedPreferences {
        override fun getAll(): MutableMap<String, *> = map.toMutableMap()
        override fun getString(key: String, defValue: String?): String? = (map[key] as String?) ?: defValue
        override fun getStringSet(key: String, defValues: MutableSet<String>?): MutableSet<String>? {
            @Suppress("UNCHECKED_CAST")
            return (map[key] as MutableSet<String>?) ?: defValues
        }
        override fun getInt(key: String, defValue: Int): Int = (map[key] as Int?) ?: defValue
        override fun getLong(key: String, defValue: Long): Long = (map[key] as Long?) ?: defValue
        override fun getFloat(key: String, defValue: Float): Float = (map[key] as Float?) ?: defValue
        override fun getBoolean(key: String, defValue: Boolean): Boolean = (map[key] as Boolean?) ?: defValue
        override fun contains(key: String): Boolean = map.containsKey(key)
        override fun edit(): SharedPreferences.Editor = throw UnsupportedOperationException("the read path does not write")
        override fun registerOnSharedPreferenceChangeListener(l: SharedPreferences.OnSharedPreferenceChangeListener) = Unit
        override fun unregisterOnSharedPreferenceChangeListener(l: SharedPreferences.OnSharedPreferenceChangeListener) = Unit
    }

    /** The type of the value that SettingsStore writes for each key constant. */
    private enum class Type { STRING, INT, FLOAT, BOOLEAN }

    private val keyTypes = mapOf(
        "KEY_MODEL" to Type.STRING, "KEY_BACKEND" to Type.STRING, "KEY_THREADS" to Type.INT, "KEY_N_CTX" to Type.INT,
        "KEY_VISION_GPU" to Type.BOOLEAN, "KEY_IMAGE_DETAIL" to Type.STRING, "KEY_THINKING" to Type.BOOLEAN,
        "KEY_TEMPERATURE" to Type.FLOAT, "KEY_TOP_P" to Type.FLOAT, "KEY_SPECULATIVE" to Type.BOOLEAN,
        "KEY_SYSTEM_PROMPT" to Type.STRING,
    )

    /** The preference key of a key constant, read from the class, thus a renamed key fails here and not silently. */
    private fun key(constant: String): String =
        SettingsStore::class.java.getDeclaredField(constant).apply { isAccessible = true }.get(null) as String

    /** A store on the preferences, made without its constructor. */
    private fun storeOn(prefs: SharedPreferences): SettingsStore {
        val unsafeField = Class.forName("sun.misc.Unsafe").getDeclaredField("theUnsafe").apply { isAccessible = true }
        val unsafe = unsafeField.get(null)
        val store = unsafe.javaClass.getMethod("allocateInstance", Class::class.java)
            .invoke(unsafe, SettingsStore::class.java) as SettingsStore
        SettingsStore::class.java.getDeclaredField("prefs").apply { isAccessible = true }.set(store, prefs)
        return store
    }

    /** The private read() of the store, with the exception of the call itself. */
    private fun read(values: Map<String, Any?>): AppSettings {
        val method = SettingsStore::class.java.getDeclaredMethod("read").apply { isAccessible = true }
        try {
            return method.invoke(storeOn(FakePrefs(values))) as AppSettings
        } catch (e: InvocationTargetException) {
            throw e.targetException
        }
    }

    private fun randomString(rnd: Random, max: Int): String {
        val pieces = listOf("a", " ", "\n", "Привет", "😀", "CPU", "GPU", "NPU", "HYBRID", "FAST", "STANDARD",
            "DETAILED", "/sdcard/qwen/models/x.gguf", "\u0000", "\t")
        val sb = StringBuilder()
        val n = rnd.nextInt(max + 1)
        while (sb.length < n) sb.append(pieces[rnd.nextInt(pieces.size)])
        return sb.toString()
    }

    private fun randomFloat(rnd: Random): Float = when (rnd.nextInt(9)) {
        0 -> 0f
        1 -> -rnd.nextFloat() * 1e30f
        2 -> rnd.nextFloat() * 1e30f
        3 -> Float.MAX_VALUE
        4 -> Float.NaN
        5 -> Float.POSITIVE_INFINITY
        6 -> Float.NEGATIVE_INFINITY
        else -> rnd.nextFloat() * 3f - 1f
    }

    /** A value of the type. A float can be NaN or infinite. */
    private fun randomValue(rnd: Random, type: Type): Any = when (type) {
        Type.STRING -> randomString(rnd, if (rnd.nextInt(8) == 0) 5000 else 40)
        Type.INT -> if (rnd.nextBoolean()) rnd.nextInt() else listOf(2048, 4096, 8192, 16384, 1, 8, 0, -1)[rnd.nextInt(8)]
        Type.FLOAT -> randomFloat(rnd)
        Type.BOOLEAN -> rnd.nextBoolean()
    }

    /** A value of a type that is not the type of the key. */
    private fun wrongValue(rnd: Random, type: Type): Any {
        val others = Type.values().filter { it != type }
        return when (others[rnd.nextInt(others.size)]) {
            Type.STRING -> "4"
            Type.INT -> 4
            Type.FLOAT -> 0.5f
            Type.BOOLEAN -> true
        }
    }

    private fun assertInRange(s: AppSettings, where: String) {
        assertTrue("$where: threads ${s.threads}", s.threads in SettingsStore.MIN_THREADS..SettingsStore.MAX_THREADS)
        assertTrue("$where: nCtx ${s.nCtx}", s.nCtx in SettingsStore.CONTEXT_LENGTHS)
        assertTrue("$where: temperature ${s.temperature}", s.temperature >= 0f && s.temperature <= SettingsStore.MAX_TEMPERATURE)
        assertTrue("$where: topP ${s.topP}", s.topP >= SettingsStore.MIN_TOP_P && s.topP <= 1f)
        assertTrue("$where: system prompt of ${s.systemPrompt.length}", s.systemPrompt.length <= SettingsStore.MAX_SYSTEM_PROMPT)
        assertTrue("$where: the system prompt starts without a blank", s.systemPrompt.isEmpty() || !s.systemPrompt[0].isWhitespace())
    }

    @Test
    fun randomValuesOfTheRightTypeReadAsSettingsInRange() {
        val rnd = Random(FuzzSwitch.seed + 10)
        repeat(FuzzSwitch.iterations) { i ->
            val values = HashMap<String, Any?>()
            for ((constant, type) in keyTypes) {
                if (rnd.nextInt(4) != 0) values[key(constant)] = randomValue(rnd, type)
            }
            val s = try {
                read(values)
            } catch (t: Throwable) {
                fail("iteration $i: read() threw $t for $values")
                return
            }
            assertInRange(s, "iteration $i")
        }
    }

    @Test
    fun aRandomSettingsObjectSanitizesIntoRange() {
        val rnd = Random(FuzzSwitch.seed + 11)
        repeat(FuzzSwitch.iterations) { i ->
            val raw = AppSettings(
                modelPath = if (rnd.nextBoolean()) null else randomString(rnd, 30),
                backend = Backend.values()[rnd.nextInt(Backend.values().size)],
                threads = rnd.nextInt(),
                nCtx = rnd.nextInt(),
                imageDetail = ImageDetail.values()[rnd.nextInt(ImageDetail.values().size)],
                temperature = randomFloat(rnd),
                topP = randomFloat(rnd),
                systemPrompt = randomString(rnd, 3000),
            )
            val s = SettingsStore.sanitize(raw)
            assertInRange(s, "iteration $i")
            assertEquals("iteration $i: the backend stays while the devices are not known", raw.backend, s.backend)
        }
    }

    /**
     * A value of a wrong type in the preferences file gives the default of
     * its key (finding settings-wrong-type, task #94). read() runs in the
     * constructor of the store, which the first screen makes: a
     * ClassCastException there stops the app at each start.
     */
    @Test
    fun aValueOfAWrongTypeDoesNotStopTheRead() {
        val rnd = Random(FuzzSwitch.seed + 12)
        val failures = ArrayList<String>()
        for ((constant, type) in keyTypes) {
            val k = key(constant)
            try {
                assertInRange(read(mapOf(k to wrongValue(rnd, type))), k)
            } catch (t: Throwable) {
                failures += "$k: $t"
            }
        }
        assertTrue("read() threw for these keys: $failures", failures.isEmpty())
    }

    /**
     * A NaN temperature or top-p in the file takes the default (finding
     * settings-nan, task #164). coerceIn keeps NaN, and NaN stops the sampler
     * of the engine.
     */
    @Test
    fun aNanTemperatureAndTopPAreSanitized() {
        val s = SettingsStore.sanitize(AppSettings(modelPath = null, backend = Backend.CPU, temperature = Float.NaN, topP = Float.NaN))
        assertEquals(SettingsStore.DEFAULT_TEMPERATURE, s.temperature)
        assertEquals(SettingsStore.DEFAULT_TOP_P, s.topP)
    }

    /**
     * Finding settings-surrogate: take(MAX_SYSTEM_PROMPT) counts UTF-16
     * units, thus it can cut an emoji in two and leave a lone high surrogate
     * at the end of the system prompt. GetStringUTFChars gives that unit to
     * the engine as three bytes that are not UTF-8.
     */
    @Test
    fun theSystemPromptLimitKeepsSurrogatePairsWhole() {
        FuzzSwitch.requireFindings("settings-surrogate")
        val prompt = "a".repeat(SettingsStore.MAX_SYSTEM_PROMPT - 1) + "😀"
        val s = SettingsStore.sanitize(AppSettings(modelPath = null, backend = Backend.CPU, systemPrompt = prompt))
        assertFalse("the prompt ends with a lone high surrogate", s.systemPrompt.last().isHighSurrogate())
    }
}
