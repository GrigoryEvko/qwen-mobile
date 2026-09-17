package ai.airi.qwenmobile

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ImageDecoder
import android.graphics.Paint
import android.graphics.Rect
import android.net.Uri
import android.util.Log
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.nio.ByteBuffer

/**
 * The image files of the conversation. [load] reads an image behind a
 * content URI as a JPEG of a bounded dimension: the message holds it, the
 * conversation file keeps it, and its SHA-256 is the key of the encoder
 * cache. [decodeForModel] decodes that JPEG for the vision encoder, at the
 * size that the preprocessor of mtmd selects, thus mtmd copies the pixels
 * without a resize.
 *
 * The two decodes go through ImageDecoder: the JPEG decoder of the
 * platform samples the file in the DCT domain at 1/2, 1/4 or 1/8, applies
 * the EXIF orientation, and reports the oriented size. One bilinear draw
 * gives the remaining factor, which stays between 1 and 2.
 */
object ImageBytes {
    /**
     * The maximum length of the long side of the stored image. 1280 px
     * keeps the 786432 pixels of 768 tokens for a 16:9 photo, thus the
     * stored image is never smaller than the target of the encoder.
     */
    const val MAX_SIDE = 1280

    /** The JPEG quality of the stored image. */
    private const val JPEG_QUALITY = 90

    private const val TAG = "ImageBytes"

    /**
     * Decode the image at the URI, apply its EXIF orientation, bound the
     * long side to [MAX_SIDE], and encode it as a JPEG.
     *
     * @throws java.io.IOException If the URI does not open or is not an image
     */
    fun load(resolver: ContentResolver, uri: Uri): ByteArray {
        var bounded = ImageGeometry.Size(0, 0)
        val decoded = ImageDecoder.decodeBitmap(ImageDecoder.createSource(resolver, uri)) { decoder, info, _ ->
            bounded = ImageGeometry.boundedSize(info.size.width, info.size.height, MAX_SIDE)
            decoder.setTargetSampleSize(ImageGeometry.sampleSize(info.size.width, info.size.height, bounded))
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
        }
        val scaled = if (decoded.width == bounded.width && decoded.height == bounded.height) {
            decoded
        } else {
            Bitmap.createScaledBitmap(decoded, bounded.width, bounded.height, true)
        }
        return ByteArrayOutputStream().use { out ->
            scaled.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out)
            out.toByteArray()
        }
    }

    /**
     * Decode a stored image for the model, at the size that the
     * preprocessor selects for [maxTokens] tokens ([ImageGeometry.targetSize])
     * with the content at the letterbox placement of mtmd
     * ([ImageGeometry.placement]) on black. The engine calls this on its
     * thread for an image that its cache does not know.
     *
     * @param dims Receives the width, the height, and the row stride in bytes of the pixels
     * @return The RGBA pixels in a direct buffer, row by row with the stride, or null when
     *   the platform does not decode the bytes. Then the engine uses its own decoder.
     */
    fun decodeForModel(bytes: ByteArray, maxTokens: Int, dims: IntArray): ByteBuffer? {
        var original = ImageGeometry.Size(0, 0)
        var target = ImageGeometry.Size(0, 0)
        val decoded = try {
            ImageDecoder.decodeBitmap(ImageDecoder.createSource(ByteBuffer.wrap(bytes))) { decoder, info, _ ->
                original = ImageGeometry.Size(info.size.width, info.size.height)
                target = ImageGeometry.targetSize(original.width, original.height, maxTokens)
                decoder.setTargetSampleSize(ImageGeometry.sampleSize(original.width, original.height, target))
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            }
        } catch (e: IOException) {
            Log.w(TAG, "The platform did not decode the image, the engine uses its own decoder", e)
            return null
        }
        // The placement comes from the original size, as in mtmd, not from the sampled decode.
        val place = ImageGeometry.placement(original.width, original.height, target)
        val canvas = Bitmap.createBitmap(target.width, target.height, Bitmap.Config.ARGB_8888)
        val dst = Rect(place.left, place.top, place.left + place.width, place.top + place.height)
        Canvas(canvas).drawBitmap(decoded, null, dst, Paint(Paint.FILTER_BITMAP_FLAG))
        decoded.recycle()
        val pixels = ByteBuffer.allocateDirect(canvas.byteCount)
        canvas.copyPixelsToBuffer(pixels)
        dims[0] = canvas.width
        dims[1] = canvas.height
        dims[2] = canvas.rowBytes
        canvas.recycle()
        return pixels
    }
}
