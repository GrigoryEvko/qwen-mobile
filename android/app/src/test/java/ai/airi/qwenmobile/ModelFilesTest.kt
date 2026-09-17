package ai.airi.qwenmobile

import org.junit.Assert.assertEquals
import org.junit.Test
import java.io.File

/** The display name of a model file. */
class ModelFilesTest {
    private fun name(file: String): String = ModelFiles.displayName(File("/sdcard/qwen/models/$file"))

    @Test
    fun theRecipeTagsReadInLongForm() {
        assertEquals(
            "Qwen3.5-2B · Q4_0 · Hadamard · col-scales · GPTQ · block-opt",
            name("Qwen3.5-2B-Q4_0-HAD-CS-GPTQ-BO.gguf"),
        )
    }

    @Test
    fun anUnknownTagStaysAsItIs() {
        assertEquals("Qwen3.5-2B · Q8_0 · XYZ", name("Qwen3.5-2B-Q8_0-XYZ.gguf"))
    }

    @Test
    fun aFamilyOnlyNameIsTheFamily() {
        assertEquals("Qwen3.5-2B", name("Qwen3.5-2B.gguf"))
        assertEquals("model", name("model.gguf"))
    }

    @Test
    fun aProjectorNameDropsTheProjectorSuffix() {
        assertEquals("Qwen3.5-2B", name("Qwen3.5-2B.mmproj.gguf"))
        assertEquals("Qwen3.5-2B · F16", name("Qwen3.5-2B-F16.mmproj.gguf"))
    }
}
