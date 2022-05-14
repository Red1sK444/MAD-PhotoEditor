package com.example.android.wonderfulapp.image.maskering

import android.graphics.Bitmap
import android.graphics.Color
import androidx.core.graphics.createBitmap
import com.example.android.wonderfulapp.image.rotating.ImageParameters
import kotlinx.coroutines.*
import kotlin.math.abs

class PixelRGBA(R: Double = 0.0, G: Double = 0.0, B: Double = 0.0, A: Double = 0.0) {
    var r = R
    var g = G
    var b = B
    var a = A

    companion object {

        fun getPixel(n: Int): PixelRGBA {
            val pixel = PixelRGBA()
            pixel.r = Color.red(n).toDouble()
            pixel.g = Color.green(n).toDouble()
            pixel.b = Color.blue(n).toDouble()
            pixel.a = Color.alpha(n).toDouble()
            return pixel
        }

        fun preparePixel(pixel: PixelRGBA) {
            if (pixel.r > 255.0) {
                pixel.r = 255.0
            }
            if (pixel.g > 255.0) {
                pixel.g = 255.0
            }
            if (pixel.b > 255.0) {
                pixel.b = 255.0
            }

            if (pixel.r < 0.0) {
                pixel.r = 0.0
            }
            if (pixel.g < 0.0) {
                pixel.g = 0.0
            }
            if (pixel.b < 0.0) {
                pixel.b = 0.0
            }
        }

        fun minus(lPixel: PixelRGBA, rPixel: PixelRGBA): PixelRGBA {
            var dif = PixelRGBA()

            dif.r = lPixel.r - rPixel.r
            dif.g = lPixel.g - rPixel.g
            dif.b = lPixel.b - rPixel.b
            dif.a = lPixel.a

            return dif
        }

        fun plus(lPixel: PixelRGBA, rPixel: PixelRGBA): PixelRGBA {
            var dif = PixelRGBA()

            dif.r = lPixel.r + rPixel.r
            dif.g = lPixel.g + rPixel.g
            dif.b = lPixel.b + rPixel.b
            dif.a = lPixel.a

            return dif
        }
    }

    fun toInt(): Int {
        val alpha = a.toInt()
        val red = r.toInt()
        val green = g.toInt()
        val blue = b.toInt()
        return Color.argb(alpha, red, green, blue)
    }

    fun mult(n: Double) {
        r *= n
        g *= n
        b *= n
    }

    fun add(pixel: PixelRGBA) {
        r += pixel.r
        g += pixel.g
        b += pixel.b
    }

    fun div(n: Double) {
        r /= n
        g /= n
        b /= n
    }

    fun minus(pixel: PixelRGBA) {
        r -= pixel.r
        g -= pixel.g
        b -= pixel.b
    }
}

class RGB {

    private fun thresholdCheck(delta: Int, threshold: Int): Int {
        if (abs(delta) < threshold) {
            return 0
        } else {
            return delta
        }
    }

    private fun pixelMaskering(original: Int, blured: Int, threshold: Int): Int {
        val pixel = PixelRGBA(A = Color.alpha(original).toDouble())

        pixel.r = Color.red(original) + thresholdCheck(
            Color.red(original) - Color.red(blured),
            threshold
        ) * 1.0
        pixel.g = Color.green(original) + thresholdCheck(
            Color.green(original) - Color.green(blured),
            threshold
        ) * 1.0
        pixel.b = Color.blue(original) + thresholdCheck(
            Color.blue(original) - Color.blue(blured),
            threshold
        ) * 1.0

        PixelRGBA.preparePixel(pixel)

        val alpha = pixel.a.toInt()
        val red = pixel.r.toInt()
        val green = pixel.g.toInt()
        val blue = pixel.b.toInt()

        return Color.argb(alpha, red, green, blue)
    }

    fun copyRGB(pixels: IntArray): IntArray {
        return pixels.copyOf()
    }

    fun minus(
        lPixels: Array<Array<PixelRGBA>>,
        rPixels: Array<Array<PixelRGBA>>
    ): Array<Array<PixelRGBA>> {
        var dif: Array<Array<PixelRGBA>> = arrayOf<Array<PixelRGBA>>()

        for (i in lPixels.indices) {
            var difPixelsLine = arrayOf<PixelRGBA>()
            for (j in rPixels[0].indices) {
                difPixelsLine += PixelRGBA.minus(lPixels[i][j], rPixels[i][j])
            }
            dif += difPixelsLine
        }
        return dif
    }

    fun plus(
        lPixels: Array<Array<PixelRGBA>>,
        rPixels: Array<Array<PixelRGBA>>
    ): Array<Array<PixelRGBA>> {
        var sum: Array<Array<PixelRGBA>> = arrayOf<Array<PixelRGBA>>()

        for (i in lPixels.indices) {
            var sumPixelsLine = arrayOf<PixelRGBA>()
            for (j in rPixels[0].indices) {
                sumPixelsLine += PixelRGBA.plus(lPixels[i][j], rPixels[i][j])
            }
            sum += sumPixelsLine
        }
        return sum
    }

    fun prepareRGB(pixels: Array<Array<PixelRGBA>>) {
        for (i in pixels.indices) {
            for (j in pixels[0].indices) {
                PixelRGBA.preparePixel(pixels[i][j])
            }
        }
    }

    fun getPixelArray(bitmap: Bitmap): IntArray {//Длина и шир
        val pixArr = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixArr, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        return pixArr
    }

    fun getBitmap(pixels: IntArray, imParam: ImageParameters): Bitmap {
        var bitmap = createBitmap(imParam.width, imParam.height)
        bitmap.setPixels(pixels, 0, imParam.width, 0, 0, imParam.width, imParam.height)
        return bitmap
    }

    private fun maskerRow(
        i: Int,
        original: IntArray,
        blured: IntArray,
        imParam: ImageParameters,
        threshold: Int
    ) {
        for (j in 0 until imParam.width) {
            original[i * imParam.width + j] =
                pixelMaskering(
                    original[i * imParam.width + j],
                    blured[i * imParam.width + j],
                    threshold
                )
        }
    }

    suspend fun maskering(
        original: IntArray,
        blured: IntArray,
        imParam: ImageParameters,
        threshold: Int
    ) =
        coroutineScope {
            var deferreds = listOf(
                launch { maskerRow(0, original, blured, imParam, threshold) }
            )
            for (i in 1 until imParam.height) {
                deferreds += launch { maskerRow(i, original, blured, imParam, threshold) }
            }
            deferreds.joinAll()
        }


    fun mult(pixel: PixelRGBA, n: Double): PixelRGBA {
        var resPixel = PixelRGBA(pixel.r, pixel.g, pixel.b, pixel.a)
        resPixel.mult(n)
        return resPixel
    }
}