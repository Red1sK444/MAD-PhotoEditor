package com.example.android.wonderfulapp.image.colorCorrection

import android.graphics.Bitmap
import android.graphics.Color
import kotlinx.coroutines.*

class SepiaFilter: Filter() {
    override suspend fun apply(curBitmap: Bitmap): Bitmap {
        return withContext(Dispatchers.IO) {
            val width = curBitmap.width
            val height = curBitmap.height
            val pixels = IntArray(width * height)
            val newPixels = IntArray(width * height)
            curBitmap.getPixels(pixels, 0, width, 0, 0, width, height)
            filter(pixels, newPixels, width, height)

            Bitmap.createBitmap(newPixels, width, height, Bitmap.Config.ARGB_8888)
        }
    }

    private fun filter(
        pixels: IntArray,
        newPixels: IntArray,
        width: Int,
        height: Int
    ) = runBlocking {
        var deffereds = listOf(
            launch { miniFilter(width, height, pixels, newPixels, 0) }
        )
        for (i in 1 until width) {
            deffereds += launch { miniFilter(width, height, pixels, newPixels, i) }
        }
        deffereds.joinAll()
    }

    private fun miniFilter(
        width: Int,
        height: Int,
        pixels: IntArray,
        newPixels: IntArray,
        i: Int
    ) {
        for (j in 0 until height) {
            val pixel = pixels[j * width + i]

            val oldAlpha = Color.alpha(pixel)
            val oldRed = Color.red(pixel)
            val oldGreen = Color.green(pixel)
            val oldBlue = Color.blue(pixel)

            var red = (0.393 * oldRed.toDouble() + 0.769 * oldGreen.toDouble() + 0.189 * oldBlue.toDouble()).toInt()
            var green = (0.349 * oldRed.toDouble() + 0.686 * oldGreen.toDouble() + 0.168 * oldBlue.toDouble()).toInt()
            var blue = (0.272 * oldRed.toDouble() + 0.534 * oldGreen.toDouble() + 0.131 * oldBlue.toDouble()).toInt()

            red = red.coerceAtMost(255)
            green = green.coerceAtMost(255)
            blue = blue.coerceAtMost(255)

            val newPixel = Color.argb(oldAlpha, red, green, blue)

            newPixels[j * width + i] = newPixel
        }
    }
}