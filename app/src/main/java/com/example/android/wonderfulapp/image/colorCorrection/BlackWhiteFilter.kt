package com.example.android.wonderfulapp.image.colorCorrection

import android.graphics.Bitmap
import android.graphics.Color
import kotlinx.coroutines.*

class BlackWhiteFilter : Filter() {
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
            val average = (Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / 3
            val newPixel = Color.argb(Color.alpha(pixel), average, average, average)

            newPixels[j * width + i] = newPixel
        }
    }
}