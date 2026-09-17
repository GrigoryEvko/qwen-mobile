package ai.airi.qwenmobile

import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * The geometry of the image path: the same arithmetic as the Qwen3-VL
 * preprocessor of mtmd, in single precision.
 *
 * [targetSize] is `calc_size_preserved_ratio` of tools/mtmd/mtmd-image.cpp
 * (the smart resize of the Qwen image processor). [placement] is the
 * letterbox of `img_tool::resize` with `PAD_CEIL`. The app hands mtmd a
 * bitmap of the target size with the content at the placement, thus mtmd
 * copies it without a resize, and a known image gives the same target
 * from its placeholder. No Android dependency: the unit tests run on the JVM.
 */
object ImageGeometry {
    /** The alignment of a side, in pixels: the patch of 16 px times the merge of 2. */
    const val ALIGN = 32

    /** The pixels of one vision token: one aligned block. */
    const val PIXELS_PER_TOKEN = ALIGN * ALIGN

    /** The minimum number of tokens of one image, from the projector metadata of Qwen3-VL. */
    const val MIN_TOKENS = 8

    /** The largest decode sample size. The JPEG decoder scales by 1/2, 1/4 and 1/8 in the DCT domain. */
    const val MAX_SAMPLE_SIZE = 8

    /** A size in pixels. */
    data class Size(val width: Int, val height: Int) {
        /** The number of vision tokens of an image of this size, when the sides are aligned. */
        val tokens: Int get() = (width / ALIGN) * (height / ALIGN)
    }

    /** The content inside the target bitmap: its size and the offset of its top left corner. The rest is black. */
    data class Placement(val width: Int, val height: Int, val left: Int, val top: Int)

    /**
     * The size that the preprocessor selects for an image: the sides
     * rounded to [ALIGN], then floored to the pixel budget of [maxTokens]
     * tokens, or raised to the budget of [MIN_TOKENS]. The aspect ratio
     * stays within the rounding. The result is a fixed point: the target
     * of a target is itself.
     */
    fun targetSize(width: Int, height: Int, maxTokens: Int): Size {
        require(width > 0 && height > 0) { "The image size must be positive, not $width x $height" }
        require(maxTokens >= MIN_TOKENS) { "The token limit must be at least $MIN_TOKENS, not $maxTokens" }
        val maxPixels = maxTokens * PIXELS_PER_TOKEN
        val minPixels = MIN_TOKENS * PIXELS_PER_TOKEN
        var w = maxOf(ALIGN, roundToAlign(width.toFloat()))
        var h = maxOf(ALIGN, roundToAlign(height.toFloat()))
        if (h * w > maxPixels) {
            val beta = sqrt(height.toFloat() * width / maxPixels)
            h = maxOf(ALIGN, floorToAlign(height / beta))
            w = maxOf(ALIGN, floorToAlign(width / beta))
        } else if (h * w < minPixels) {
            val beta = sqrt(minPixels.toFloat() / (height.toFloat() * width))
            h = ceilToAlign(height * beta)
            w = ceilToAlign(width * beta)
        }
        return Size(w, h)
    }

    /**
     * Where the image goes inside the target: scaled by the smaller of the
     * two factors, the sides rounded up, centered with the odd pixel on the
     * right and at the bottom. An image of the target size fills it.
     */
    fun placement(width: Int, height: Int, target: Size): Placement {
        require(width > 0 && height > 0) { "The image size must be positive, not $width x $height" }
        val scale = min(target.width.toFloat() / width, target.height.toFloat() / height)
        val w = min(ceil(width * scale).toInt(), target.width)
        val h = min(ceil(height * scale).toInt(), target.height)
        return Placement(w, h, (target.width - w) / 2, (target.height - h) / 2)
    }

    /** The size with the long side at most [maxSide], the aspect ratio kept, each side at least 1. */
    fun boundedSize(width: Int, height: Int, maxSide: Int): Size {
        require(width > 0 && height > 0) { "The image size must be positive, not $width x $height" }
        val scale = maxSide.toFloat() / maxOf(width, height)
        if (scale >= 1f) {
            return Size(width, height)
        }
        return Size(maxOf(1, (width * scale).toInt()), maxOf(1, (height * scale).toInt()))
    }

    /**
     * The decode sample size for an image that goes to [target]: the
     * largest power of two, at most [MAX_SAMPLE_SIZE], that keeps the
     * decoded image at least as large as the target on the two sides.
     * The decoder floors the sampled sides, thus the check floors too.
     * The remaining factor to the target is between 1 and 2, which a
     * bilinear scale does without aliasing.
     */
    fun sampleSize(width: Int, height: Int, target: Size): Int {
        var sample = 1
        while (sample < MAX_SAMPLE_SIZE && width / (sample * 2) >= target.width && height / (sample * 2) >= target.height) {
            sample *= 2
        }
        return sample
    }

    private fun roundToAlign(x: Float): Int = (x / ALIGN).roundToInt() * ALIGN

    private fun ceilToAlign(x: Float): Int = ceil(x / ALIGN).toInt() * ALIGN

    private fun floorToAlign(x: Float): Int = floor(x / ALIGN).toInt() * ALIGN
}
