package ai.airi.qwenmobile

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The geometry of the image path against the mtmd preprocessor. The table
 * comes from a host program that calls `img_tool::calc_size_preserved_ratio`
 * and the letterbox of `img_tool::resize` of tools/mtmd/mtmd-image.cpp at
 * llama.cpp c6824a9: one row is `width height cap -> target tokens | content offset`.
 */
class ImageGeometryTest {
    private val table = """
        681 1024 256 -> 416 608 247 | 405 608 5 0
        681 1024 576 -> 608 928 551 | 608 915 0 6
        681 1024 768 -> 672 1024 672 | 672 1011 0 6
        681 1024 1024 -> 672 1024 672 | 672 1011 0 6
        1024 681 256 -> 608 416 247 | 608 405 0 5
        1024 681 576 -> 928 608 551 | 915 608 6 0
        1024 681 768 -> 1024 672 672 | 1011 672 6 0
        768 1024 256 -> 416 576 234 | 416 555 0 10
        768 1024 576 -> 640 864 540 | 640 854 0 5
        768 1024 768 -> 768 1024 768 | 768 1024 0 0
        1024 768 256 -> 576 416 234 | 555 416 10 0
        1024 768 576 -> 864 640 540 | 854 640 5 0
        1024 768 768 -> 1024 768 768 | 1024 768 0 0
        1024 1024 256 -> 512 512 256 | 512 512 0 0
        1024 1024 576 -> 768 768 576 | 768 768 0 0
        1024 1024 768 -> 864 864 729 | 864 864 0 0
        1024 1024 1024 -> 1024 1024 1024 | 1024 1024 0 0
        1024 576 256 -> 672 384 252 | 672 378 0 3
        1024 576 576 -> 1024 576 576 | 1024 576 0 0
        1024 576 768 -> 1024 576 576 | 1024 576 0 0
        576 1024 256 -> 384 672 252 | 378 672 3 0
        576 1024 576 -> 576 1024 576 | 576 1024 0 0
        3000 4000 256 -> 416 576 234 | 416 555 0 10
        3000 4000 576 -> 640 864 540 | 640 854 0 5
        3000 4000 768 -> 768 1024 768 | 768 1024 0 0
        4000 3000 576 -> 864 640 540 | 854 640 5 0
        4032 3024 256 -> 576 416 234 | 555 416 10 0
        4032 3024 576 -> 864 640 540 | 854 640 5 0
        4032 3024 768 -> 1024 768 768 | 1024 768 0 0
        1224 1632 576 -> 640 864 540 | 640 854 0 5
        1920 1080 256 -> 672 384 252 | 672 378 0 3
        1920 1080 576 -> 1024 576 576 | 1024 576 0 0
        1920 1080 768 -> 1152 640 720 | 1138 640 7 0
        4000 1000 256 -> 1024 256 256 | 1024 256 0 0
        4000 1000 576 -> 1536 384 576 | 1536 384 0 0
        4000 1000 768 -> 1760 416 715 | 1664 416 48 0
        100 100 576 -> 96 96 9 | 96 96 0 0
        64 64 576 -> 96 96 9 | 96 96 0 0
        1 1 256 -> 96 96 9 | 96 96 0 0
        33 3000 576 -> 32 3008 94 | 32 2910 0 49
        2409 3212 768 -> 768 1024 768 | 768 1024 0 0
        1280 720 768 -> 1152 640 720 | 1138 640 7 0
        992 744 768 -> 992 736 713 | 982 736 5 0
        608 928 256 -> 384 608 228 | 384 587 0 10
        608 928 576 -> 608 928 551 | 608 928 0 0
        416 608 256 -> 416 608 247 | 416 608 0 0
        800 600 256 -> 576 416 234 | 555 416 10 0
        800 600 576 -> 800 608 475 | 800 600 0 4
        960 1280 768 -> 768 1024 768 | 768 1024 0 0
        31 31 768 -> 96 96 9 | 96 96 0 0
        33 33 256 -> 96 96 9 | 96 96 0 0
        48 48 768 -> 96 96 9 | 96 96 0 0
    """.trimIndent().lines()

    private class Row(val width: Int, val height: Int, val cap: Int, val target: ImageGeometry.Size, val tokens: Int,
                      val placement: ImageGeometry.Placement)

    private fun rows(): List<Row> = table.map { line ->
        val n = line.replace("->", " ").replace("|", " ").split(Regex("\\s+")).filter { it.isNotEmpty() }.map { it.toInt() }
        Row(n[0], n[1], n[2], ImageGeometry.Size(n[3], n[4]), n[5], ImageGeometry.Placement(n[6], n[7], n[8], n[9]))
    }

    @Test
    fun theTargetIsTheSizeOfTheMtmdPreprocessor() {
        for (row in rows()) {
            val target = ImageGeometry.targetSize(row.width, row.height, row.cap)
            assertEquals("${row.width} x ${row.height} at ${row.cap}", row.target, target)
            assertEquals("${row.width} x ${row.height} at ${row.cap}", row.tokens, target.tokens)
        }
    }

    @Test
    fun thePlacementIsTheLetterboxOfMtmd() {
        for (row in rows()) {
            val place = ImageGeometry.placement(row.width, row.height, row.target)
            assertEquals("${row.width} x ${row.height} in ${row.target}", row.placement, place)
        }
    }

    @Test
    fun theTargetOfATargetIsItself() {
        // The placeholder of a known image has the target size, and the preprocessor must keep it.
        for (row in rows()) {
            val again = ImageGeometry.targetSize(row.target.width, row.target.height, row.cap)
            assertEquals("${row.target} at ${row.cap}", row.target, again)
            val place = ImageGeometry.placement(row.target.width, row.target.height, row.target)
            assertEquals(ImageGeometry.Placement(row.target.width, row.target.height, 0, 0), place)
        }
    }

    @Test
    fun theTargetStaysInsideTheBudgetForEverySize() {
        for (cap in listOf(256, 576, 768)) {
            for (width in 1..4200 step 37) {
                for (height in 1..4200 step 41) {
                    val target = ImageGeometry.targetSize(width, height, cap)
                    val tokens = target.tokens
                    assertTrue("$width x $height at $cap gives $tokens tokens", tokens in ImageGeometry.MIN_TOKENS..cap)
                    assertEquals(0, target.width % ImageGeometry.ALIGN)
                    assertEquals(0, target.height % ImageGeometry.ALIGN)
                    assertEquals(target, ImageGeometry.targetSize(target.width, target.height, cap))
                    val place = ImageGeometry.placement(width, height, target)
                    assertTrue(place.left >= 0 && place.top >= 0)
                    assertTrue(place.left + place.width <= target.width)
                    assertTrue(place.top + place.height <= target.height)
                    assertTrue(place.width == target.width || place.height == target.height)
                }
            }
        }
    }

    @Test
    fun theSampleSizeKeepsTheDecodeAtLeastAsLargeAsTheTarget() {
        val cases = listOf(
            // width, height, target, sample
            listOf(3000, 4000, 640, 864, 4),
            listOf(3000, 4000, 416, 576, 4),
            listOf(3000, 4000, 768, 1024, 2),
            listOf(4032, 3024, 576, 416, 4),
            listOf(1224, 1632, 640, 864, 1),
            listOf(1224, 1632, 416, 576, 2),
            listOf(768, 1024, 768, 1024, 1),
            listOf(1280, 960, 576, 416, 2),
            listOf(64, 64, 96, 96, 1),
            listOf(10000, 10000, 864, 864, 8),
            listOf(100000, 100000, 864, 864, 8),
        )
        for (c in cases) {
            val target = ImageGeometry.Size(c[2], c[3])
            val sample = ImageGeometry.sampleSize(c[0], c[1], target)
            assertEquals("${c[0]} x ${c[1]} to $target", c[4], sample)
            // A sample above 1 keeps the decode at least as large as the target. Sample 1 also covers an upscale.
            if (sample > 1) {
                assertTrue(c[0] / sample >= target.width && c[1] / sample >= target.height)
            }
            // The next power of two would make the decode smaller than the target, unless the ceiling stops it.
            if (sample < ImageGeometry.MAX_SAMPLE_SIZE) {
                assertTrue(c[0] / (sample * 2) < target.width || c[1] / (sample * 2) < target.height)
            }
        }
    }

    @Test
    fun theBoundedSizeKeepsTheAspectRatio() {
        assertEquals(ImageGeometry.Size(960, 1280), ImageGeometry.boundedSize(3000, 4000, 1280))
        assertEquals(ImageGeometry.Size(1280, 320), ImageGeometry.boundedSize(4000, 1000, 1280))
        assertEquals(ImageGeometry.Size(681, 1024), ImageGeometry.boundedSize(681, 1024, 1280))
        assertEquals(ImageGeometry.Size(1280, 1280), ImageGeometry.boundedSize(1280, 1280, 1280))
        assertEquals(ImageGeometry.Size(1280, 1), ImageGeometry.boundedSize(100000, 1, 1280))
    }

    @Test
    fun theStoredImageIsNeverSmallerThanTheTargetOfTheDetailedLevel() {
        // A 16:9, a 4:3, a 3:2 and a square photo of 12 MP, bounded to MAX_SIDE, then the 768 target.
        for ((w, h) in listOf(4000 to 2250, 4000 to 3000, 4000 to 2667, 3464 to 3464, 2250 to 4000)) {
            val stored = ImageGeometry.boundedSize(w, h, ImageBytes.MAX_SIDE)
            val target = ImageGeometry.targetSize(stored.width, stored.height, ImageDetail.DETAILED.tokens)
            assertTrue("$w x $h stored as $stored, target $target", stored.width >= target.width && stored.height >= target.height)
        }
    }
}
