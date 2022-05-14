package com.example.android.wonderfulapp.image.retouch

import android.graphics.Bitmap
import com.example.android.wonderfulapp.image.helpFunc.Coordinates
import com.example.android.wonderfulapp.image.helpFunc.GaussianBlurAlg
import com.example.android.wonderfulapp.image.maskering.RGB
import com.example.android.wonderfulapp.image.rotating.ImageParameters
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

data class RetouchPointData(val coordinates: Coordinates, val radius: Double, val sigma: Double)

class RetouchingAlgorithm {
    val retouchFinishPoints = mutableListOf<RetouchPointData>()

    suspend fun retouching(x:Int, y:Int, imageBitmap: Bitmap, radius: Double, sigma:Double): Bitmap {

        return withContext(Dispatchers.IO) {
            val rgb = RGB()
            val gb = GaussianBlurAlg()
            val imageParams = ImageParameters(imageBitmap.width, imageBitmap.height)
            val pixels = rgb.getPixelArray(imageBitmap)

            gb.gaussianRoundBlur(x, y, pixels, radius, sigma, imageParams)
            rgb.getBitmap(pixels, imageParams)
        }
    }
}