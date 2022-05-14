package com.example.android.wonderfulapp.image.rotating

import android.graphics.Bitmap
import com.example.android.wonderfulapp.image.helpFunc.ValueCheck
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.*

data class ImageParameters(var width: Int, var height: Int)

class RotatingAlgorithm(private val image: Bitmap) {

    private fun rotate90(
        pixels: IntArray,
        theta: Int,
        initImageParams: ImageParameters,
        resImageParams: ImageParameters
    ): IntArray {
        val height = initImageParams.height
        val width = initImageParams.width
        val resultPixels = IntArray(height * width)

        when (theta) {
            90 -> {
                for (i: Int in 0 until height) {
                    for (j: Int in 0 until width) {
                        resultPixels[(width - 1 - j) * height + i] = pixels[i * width + j]
                    }
                }
                resImageParams.width = height
                resImageParams.height = width
            }
            -90 -> {
                for (i: Int in 0 until height) {
                    for (j: Int in 0 until width) {
                        resultPixels[j * height + (height - 1 - i)] = pixels[i * width + j]
                    }
                }
                resImageParams.width = height
                resImageParams.height = width
            }
            180 -> {
                for (i: Int in 0 until height) {
                    for (j: Int in 0 until width) {
                        resultPixels[(height - 1 - i) * width + (width - 1 - j)] =
                            pixels[i * width + j]
                    }
                }
            }
        }
        initImageParams.width = resImageParams.width
        initImageParams.height = resImageParams.height
        return resultPixels
    }

    private fun getInRadians(angle: Int): Double {
        return if (abs(angle) == 30) {
            (-(angle + 0.01) * PI) / 180
        } else {
            (-angle * PI) / 180
        }
    }


    private fun shear(y: Int, x: Int, tan: Double, sin: Double): Array<Int> {
        var newX = round(x - y * tan).toInt()
        var newY = y
        newY = (round(newX * sin + newY)).toInt()
        newX = (round(newX - newY * tan)).toInt()
        return arrayOf(newY, newX)
    }


    private fun rotateImage(
        pixels: IntArray,
        angle: Int,
        initImageParams: ImageParameters,
        resImageParams: ImageParameters
    ): IntArray {

        if (angle == 0) {
            return pixels
        }

        val checker = ValueCheck()
        val theta: Double = getInRadians(angle)
        val sin: Double = sin(theta)
        val cos: Double = cos(theta)
        val tan: Double = tan(theta / 2)

        resImageParams.height =
            (round(abs(initImageParams.height * cos) + abs(initImageParams.width * sin)) + 1).toInt()
        resImageParams.width =
            (round(abs(initImageParams.width * cos) + abs(initImageParams.height * sin)) + 1).toInt()

        val resultPixels = IntArray((resImageParams.width * resImageParams.height))

        val originalCentreHeight = round(((initImageParams.height.toDouble() + 1) / 2) - 1).toInt()
        val originalCentreWidth = round(((initImageParams.width.toDouble() + 1) / 2) - 1).toInt()

        val newCentreHeight = round(((resImageParams.height.toDouble() + 1) / 2) - 1).toInt()
        val newCentreWidth = round(((resImageParams.width.toDouble() + 1) / 2) - 1).toInt()

        for (i: Int in 0 until initImageParams.height) {
            for (j: Int in 0 until initImageParams.width) {
                val y = (initImageParams.height - 1 - i - originalCentreHeight)
                val x = (initImageParams.width - 1 - j - originalCentreWidth)

                val shearRes = shear(y, x, tan, sin)
                var newY = shearRes[0]
                var newX = shearRes[1]

                newY = newCentreHeight - newY
                newX = newCentreWidth - newX

                if (checker.inRange(newY, resImageParams.height) && checker.inRange(
                        newX,
                        resImageParams.width
                    )
                )
                    resultPixels[newY * resImageParams.width + newX] =
                        pixels[i * initImageParams.width + j]
            }
        }
        return resultPixels
    }

    suspend fun rotate(angle: Int): Bitmap {
        return withContext(Dispatchers.IO) {
            val theta = (angle.toDouble()).IEEErem(360.0)
            var pixels = IntArray(image.height * image.width)
            image.getPixels(pixels, 0, image.width, 0, 0, image.width, image.height)

            val initImageParams = ImageParameters(image.width, image.height)
            val resImageParams = ImageParameters(image.width, image.height)

            when (theta) {
                in -45.0..45.0 -> {
                    pixels = rotateImage(pixels, theta.toInt(), initImageParams, resImageParams)
                }
                in 46.0..135.0 -> {
                    pixels = rotate90(pixels, 90, initImageParams, resImageParams)
                    pixels =
                        rotateImage(pixels, theta.toInt() - 90, initImageParams, resImageParams)
                }
                in 136.0..181.0 -> {
                    pixels = rotate90(pixels, 180, initImageParams, resImageParams)
                    pixels =
                        rotateImage(pixels, theta.toInt() - 180, initImageParams, resImageParams)
                }
                in -180.0..-136.0 -> {
                    pixels = rotate90(pixels, 180, initImageParams, resImageParams)
                    pixels =
                        rotateImage(pixels, theta.toInt() + 180, initImageParams, resImageParams)
                }
                in -135.0..-46.0 -> {
                    pixels = rotate90(pixels, -90, initImageParams, resImageParams)
                    pixels =
                        rotateImage(pixels, theta.toInt() + 90, initImageParams, resImageParams)
                }
            }

            val resultBitmap: Bitmap = Bitmap.createBitmap(
                resImageParams.width,
                resImageParams.height,
                image.config
            )
            resultBitmap.setPixels(
                pixels,
                0,
                resImageParams.width,
                0,
                0,
                resImageParams.width,
                resImageParams.height
            )

            resultBitmap
        }
    }
}

