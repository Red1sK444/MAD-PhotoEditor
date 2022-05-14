package com.example.android.wonderfulapp.image.maskering

import android.graphics.Bitmap
import com.example.android.wonderfulapp.image.helpFunc.GaussianBlurAlg
import com.example.android.wonderfulapp.image.rotating.ImageParameters
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext

class MaskeringAlgorithm(private val image: Bitmap) {

    private val gb: GaussianBlurAlg = GaussianBlurAlg()
    private val RGB: RGB = RGB()

    suspend fun maskering(radius: Double, amount: Double, threshold: Int): Bitmap {
        return withContext(Dispatchers.IO) {
            val imageParams = ImageParameters(image.width, image.height)
            val pixels = RGB.getPixelArray(image)
            val bluredPixels = RGB.copyRGB(pixels)

            gb.gaussianBlur(bluredPixels, amount, radius, imageParams)

            runBlocking {
                val l1 = launch {
                    RGB.maskering(pixels, bluredPixels, imageParams, threshold)
                }
                l1.join()
            }

            RGB.getBitmap(pixels, imageParams)
        }
    }
}