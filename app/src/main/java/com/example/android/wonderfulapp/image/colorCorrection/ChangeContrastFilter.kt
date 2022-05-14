package com.example.android.wonderfulapp.image.colorCorrection

import android.graphics.Bitmap
import android.graphics.Color
import kotlinx.coroutines.*
import kotlin.math.pow

class ChangeContrastFilter : Filter() {
    override suspend fun apply(
        curBitmap: Bitmap
    ): Bitmap {
        return withContext(Dispatchers.IO) {
            val contrastValue = 80f
            val contrast = ((100f + contrastValue) / 100).pow(2f)
            val width = curBitmap.width
            val height = curBitmap.height
            val pixels = IntArray(width * height)
            val newPixels = IntArray(width * height)
            curBitmap.getPixels(pixels, 0, width, 0, 0, width, height)
            filter(pixels, newPixels, width, height, contrast)

            Bitmap.createBitmap(newPixels, width, height, Bitmap.Config.ARGB_8888)
        }
    }

    private fun filter(
        pixels: IntArray,
        newPixels: IntArray,
        width: Int,
        height: Int,
        contrast: Float
    ) = runBlocking {
        for (i in 0 until width) {
            var deffereds = listOf(
                launch { miniFilter(width, height, pixels, newPixels, 0, contrast) }
            )
            for (i in 1 until width) {
                deffereds += launch { miniFilter(width, height, pixels, newPixels, i, contrast) }
            }
            deffereds.joinAll()
        }
    }

    private fun miniFilter(
        width: Int,
        height: Int,
        pixels: IntArray,
        newPixels: IntArray,
        i: Int,
        contrast: Float
    ) {
        for (j in 0 until height) {
            val pixel = pixels[j * width + i]

            val alpha = Color.alpha(pixel)
            var red = Color.red(pixel)
            var green = Color.green(pixel)
            var blue = Color.blue(pixel)

            red = ((((red / 255f - 0.5f) * contrast) + 0.5f) * 255f).toInt()
            if (red < 0) red = 0
            else if (red > 255) red = 255

            green = ((((green / 255f - 0.5f) * contrast) + 0.5f) * 255f).toInt()
            if (green < 0) green = 0
            else if (green > 255) green = 255

            blue = ((((blue / 255f - 0.5f) * contrast) + 0.5f) * 255f).toInt()
            if (blue < 0) blue = 0
            else if (blue > 255) blue = 255

            newPixels[j * width + i] = Color.argb(alpha, red, green, blue)
        }
    }
}