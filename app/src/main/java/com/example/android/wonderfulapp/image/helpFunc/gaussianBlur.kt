package com.example.android.wonderfulapp.image.helpFunc

import android.graphics.Color
import com.example.android.wonderfulapp.image.rotating.ImageParameters
import com.example.android.wonderfulapp.image.maskering.PixelRGBA
import kotlinx.coroutines.*
import kotlin.math.*

class GaussianBlurAlg {

    private fun fillBlurRad(blurRad: IntArray, N: Int) {
        val x = N
        val y = N
        for (i in 0..2 * N) {
            for (j in 0..2 * N) {
                blurRad[i * (2 * N + 1) + j] = max(
                    1,
                    N - abs(x - j) - abs(y - i)
                )
                blurRad[i * (2 * N + 1) + j] = N
            }
        }
    }

    private fun fillGaussianCoef(array: DoubleArray, s2: Double) {
        val N: Int = floor(array.size.toDouble() / 2).toInt()

        array[N] = 1.0
        for (i in 1..N) {
            array[N + i] = exp((-i * i) / s2)
            array[N - i] = array[N + i]
        }
    }


    //Полное размытие

    private fun processRow(
        i: Int,
        pixels: IntArray,
        window: DoubleArray,
        width: Int,
        height: Int,
        N: Int
    ) {
        val checker = ValueCheck()
        var tmp = arrayOf<PixelRGBA>()
        for (j in 0 until width) {
            var summ = 0.0
            var pixel = PixelRGBA(A = Color.alpha(pixels[i * width + j]).toDouble())
            for (k in -N..N) {
                if (checker.inRange(j + k, width)) {
                    var curPix = PixelRGBA.getPixel(pixels[i * width + j + k])
                    curPix.mult(window[N + k])
                    pixel.add(curPix)
                    summ += window[N + k]
                }
            }
            pixel.div(summ)
            tmp += pixel
        }
    }


    private suspend fun blurRows(
        pixels: IntArray,
        window: DoubleArray,
        width: Int,
        height: Int,
        N: Int
    ) = coroutineScope {
        var deferreds = listOf(
            launch { processRow(0, pixels, window, width, height, N) }
        )
        for (i in 1 until height) {
            deferreds += launch { processRow(i, pixels, window, width, height, N) }
        }
        deferreds.joinAll()
    }

    private fun processCol(
        j: Int,
        pixels: IntArray,
        window: DoubleArray,
        width: Int,
        height: Int,
        N: Int
    ) {
        val checker = ValueCheck()
        var tmp = arrayOf<PixelRGBA>()
        for (i in 0 until height) {
            var summ = 0.0
            var pixel = PixelRGBA(A = Color.alpha(pixels[i * width + j]).toDouble())
            for (k in -N..N) {
                if (checker.inRange(i + k, height)) {
                    var curPix = PixelRGBA.getPixel(pixels[(i + k) * width + j])
                    curPix.mult(window[N + k])
                    pixel.add(curPix)
                    summ += window[N + k]
                }
            }
            pixel.div(summ)
            tmp += pixel
        }
        for (i in 0 until height) {
            pixels[i * width + j] = tmp[i].toInt()
        }
    }

    private suspend fun blurCols(
        pixels: IntArray,
        window: DoubleArray,
        width: Int,
        height: Int,
        N: Int
    ) = coroutineScope {
        var deferreds = listOf(
            launch { processCol(0, pixels, window, width, height, N) }
        )
        for (j in 1 until width) {
            deferreds += launch { processCol(j, pixels, window, width, height, N) }
        }
        deferreds.joinAll()
    }

    fun gaussianBlur(pixels: IntArray, sigma: Double, radius: Double, imParam: ImageParameters) {
        val s2 = 2 * sigma * sigma
        val N = radius.toInt()
        val window = DoubleArray(2 * N + 1)

        fillGaussianCoef(window, s2)

        runBlocking {
            val l1 = launch { blurRows(pixels, window, imParam.width, imParam.height, N) }
            l1.join()
            val l2 = launch { blurCols(pixels, window, imParam.width, imParam.height, N) }
            l2.join()
        }
    }


    //Точечное размытие
    private suspend fun processRoundRow(
        x: Int,
        y: Int,
        i: Int,
        pixels: IntArray,
        width: Int,
        height: Int,
        N: Int,
        blurRad: IntArray,
        sigma: Double
    ) = coroutineScope {
        val checker = ValueCheck()
        if (checker.inRange(i, height)) {
            var tmp = arrayOf<PixelRGBA>()
            for (j in (x - N)..(x + N)) {
                if (checker.inRange(j, width) && Coordinates.euclideanDist(
                        x,
                        y,
                        j,
                        i
                    ) <= N.toDouble()
                ) {
                    var summ = 0.0
                    var pixel = PixelRGBA(A = Color.alpha(pixels[i * width + j]).toDouble())
                    val rad = blurRad[(i - y + N) * (2 * N + 1) + (j - x + N)]

                    for (k in -rad..rad) {
                        if (checker.inRange(j + k, width)) {
                            var curPix = PixelRGBA.getPixel(pixels[i * width + j + k])
                            val win = getWindowElem(sigma * getScaleCoef(x, y, i + k, j, N), k)
                            curPix.mult(win)
                            pixel.add(curPix)
                            summ += win
                        }
                    }
                    pixel.div(summ)
                    tmp += pixel
                }
            }

            var tmpIdx = 0
            for (j in (x - N)..(x + N)) {
                if (checker.inRange(j, width) && Coordinates.euclideanDist(
                        x,
                        y,
                        j,
                        i
                    ) <= N.toDouble()
                ) {
                    pixels[i * width + j] = tmp[tmpIdx++].toInt()
                }
            }
        }
    }


    private suspend fun blurRoundRows(
        x: Int,
        y: Int,
        pixels: IntArray,
        width: Int,
        height: Int,
        N: Int,
        blurRad: IntArray,
        sigma: Double
    ) = coroutineScope {
        var deferreds = listOf<Job>()

        for (i in (y - N)..(y + N)) {
            deferreds += launch {
                processRoundRow(
                    x,
                    y,
                    i,
                    pixels,
                    width,
                    height,
                    N,
                    blurRad,
                    sigma
                )
            }
        }
        deferreds.joinAll()
    }

    private fun getScaleCoef(x: Int, y: Int, i: Int, j: Int, N: Int): Double {
        return max(1 - Coordinates.euclideanDist(x, y, j, i) / N, 0.001)
    }

    private fun getWindowElem(newSigma: Double, k: Int): Double {
        return exp((-k * k) / (2 * newSigma * newSigma))
    }

    private fun processRoundCol(
        x: Int,
        y: Int,
        j: Int,
        pixels: IntArray,
        width: Int,
        height: Int,
        N: Int,
        blurRad: IntArray,
        sigma: Double
    ) {
        val checker = ValueCheck()
        if (checker.inRange(j, width)) {
            var tmp = arrayOf<PixelRGBA>()
            for (i in (y - N)..(y + N)) {
                if (checker.inRange(i, height) && Coordinates.euclideanDist(
                        x,
                        y,
                        j,
                        i
                    ) <= N.toDouble()
                ) {
                    var summ = 0.0
                    var pixel = PixelRGBA(A = Color.alpha(pixels[i * width + j]).toDouble())
                    val rad = blurRad[(i - y + N) * (2 * N + 1) + (j - x + N)]

                    for (k in -rad..rad) {//-------------------------------------------------------------------------------------  -N..N
                        if (checker.inRange(i + k, height)) {
                            var curPix = PixelRGBA.getPixel(pixels[(i + k) * width + j])
                            val win = getWindowElem(sigma * getScaleCoef(x, y, i + k, j, N), k)
                            curPix.mult(win)
                            pixel.add(curPix)
                            summ += win//window[N + k]
                        }
                    }
                    pixel.div(summ)
                    tmp += pixel
                }
            }

            var tmpIdx = 0
            for (i in (y - N)..(y + N)) {
                if (checker.inRange(i, height) && Coordinates.euclideanDist(
                        x,
                        y,
                        j,
                        i
                    ) <= N.toDouble()
                ) {
                    pixels[i * width + j] = tmp[tmpIdx++].toInt()
                }
            }
        }
    }

    private suspend fun blurRoundCols(
        x: Int,
        y: Int,
        pixels: IntArray,
        width: Int,
        height: Int,
        N: Int,
        blurRad: IntArray,
        sigma: Double
    ) = coroutineScope {
        var deferreds = listOf<Job>()

        for (j in (x - N)..(x + N)) {
            deferreds += launch {
                processRoundCol(
                    x,
                    y,
                    j,
                    pixels,
                    width,
                    height,
                    N,
                    blurRad,
                    sigma
                )
            }
        }
        deferreds.joinAll()
    }


    fun gaussianRoundBlur(
        x: Int,
        y: Int,
        pixels: IntArray,
        radius: Double,
        sigma: Double,
        imParam: ImageParameters
    ) {
        val N = radius.toInt()
        val blurRad: IntArray = IntArray((2 * N + 1) * (2 * N + 1))
        fillBlurRad(blurRad, N)

        runBlocking {
            val l1 =
                launch {
                    blurRoundRows(
                        x,
                        y,
                        pixels,
                        imParam.width,
                        imParam.height,
                        N,
                        blurRad,
                        sigma
                    )
                }
            l1.join()
            val l2 =
                launch {
                    blurRoundCols(
                        x,
                        y,
                        pixels,
                        imParam.width,
                        imParam.height,
                        N,
                        blurRad,
                        sigma
                    )
                }
            l2.join()
        }
    }
}