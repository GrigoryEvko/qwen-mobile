package ai.airi.qwenmobile

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import java.io.ByteArrayOutputStream

/** Reads an image behind a content URI as a JPEG of a bounded dimension. */
object ImageBytes {
    /** The maximum length of the long side after the decode, in pixels. 1024 px is 1024 vision tokens. */
    const val MAX_SIDE = 1024

    /** The JPEG quality of the encoded image. */
    private const val JPEG_QUALITY = 90

    /**
     * Decode the image at the URI, apply its EXIF orientation, bound the
     * long side to [MAX_SIDE], and encode it as a JPEG.
     *
     * @throws java.io.IOException If the URI does not open or is not an image
     */
    fun load(resolver: ContentResolver, uri: Uri): ByteArray {
        val source = ImageDecoder.createSource(resolver, uri)
        val decoded = ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
            val longSide = maxOf(info.size.width, info.size.height)
            decoder.setTargetSampleSize(maxOf(1, longSide / MAX_SIDE))
            decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
        }
        val scale = MAX_SIDE.toFloat() / maxOf(decoded.width, decoded.height)
        val bounded = if (scale < 1f) {
            Bitmap.createScaledBitmap(decoded, (decoded.width * scale).toInt(), (decoded.height * scale).toInt(), true)
        } else {
            decoded
        }
        return ByteArrayOutputStream().use { out ->
            bounded.compress(Bitmap.CompressFormat.JPEG, JPEG_QUALITY, out)
            out.toByteArray()
        }
    }
}
