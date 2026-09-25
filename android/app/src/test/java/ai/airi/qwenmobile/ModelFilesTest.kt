package ai.airi.qwenmobile

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder
import java.io.File

/** The display name of a model file, and the file with the reduced draft head. */
class ModelFilesTest {
    @get:Rule
    val folder = TemporaryFolder()

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

    @Test
    fun theDraftHeadFileIsTheTwinOfTheModel() {
        val model = folder.newFile("Qwen3.5-4B-Q8_0.gguf")
        assertNull(ModelFiles.draftHeadFile(model))
        val twin = folder.newFile("Qwen3.5-4B-Q8_0-draft32k.gguf")
        assertEquals(twin, ModelFiles.draftHeadFile(model))
        // The twin has no twin of its own, and the list shows the model only.
        assertNull(ModelFiles.draftHeadFile(twin))
        assertTrue(ModelFiles.isDraftHeadTwin(twin))
        assertFalse(ModelFiles.isDraftHeadTwin(model))
    }

    @Test
    fun aDraftHeadFileWithoutItsModelIsAModel() {
        val alone = folder.newFile("Qwen3.5-2B-Q8_0-draft32k.gguf")
        assertFalse(ModelFiles.isDraftHeadTwin(alone))
    }
}
