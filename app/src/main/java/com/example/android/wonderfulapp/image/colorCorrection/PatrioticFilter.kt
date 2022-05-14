package com.example.android.wonderfulapp.image.colorCorrection

import android.graphics.Bitmap
import android.graphics.Color
import kotlinx.coroutines.*

class PatrioticFilter : Filter() {
    override suspend fun apply(curBitmap: Bitmap): Bitmap {
        return withContext(Dispatchers.IO) {
            val width = curBitmap.width
            val height = curBitmap.height
            val pixels = IntArray(width * height)
            val newPixels = IntArray(width * height)
            curBitmap.getPixels(pixels, 0, width, 0, 0, width, height)

            for (j in 1 until height) {
                pixels[j * width] = Color.BLACK
                pixels[j * width + width - 1] = Color.BLACK
            }

            filter(pixels, newPixels, width, height)

            for (i in 0 until width) {
                newPixels[i] = Color.BLACK
                newPixels[width * height - width + i] = Color.BLACK
            }

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

            val newPixel = when {
                j < height / 3 -> {
                    var average = 2 * (oldRed + oldGreen + oldBlue) / 3
                    average = average.coerceAtMost(255)
                    Color.argb(oldAlpha, average, average,average)
                }
                j < 2 * height / 3 -> {
                    Color.argb(oldAlpha, oldRed, oldGreen, 255)
                }
                else -> {
                    Color.argb(oldAlpha, 255, oldGreen, oldBlue)
                }
            }
            newPixels[j * width + i] = newPixel
        }
    }

}