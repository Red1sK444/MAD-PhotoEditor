package com.example.android.wonderfulapp.image.filtering

import android.graphics.Bitmap
import com.example.android.wonderfulapp.image.scale.ImageData
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class FilteringAlgorithm(image: Bitmap) {
    private val width: Int = image.width
    private val height: Int = image.height
    private val imageData: ImageData = ImageData(IntArray(width * height), width, height)

    init {
        image.getPixels(imageData.pixels, 0, width, 0, 0, width, height)
    }

    fun getAffineMatrices(): Array<FloatArray> {
        val determinant = Points.getPoints(PointsState.START).determinant3d()
        val factorMatrix = calculateFactorMatrix(determinant)
        val translationMatrix = calculateTranslationMatrix(determinant)

        return arrayOf(factorMatrix, translationMatrix)
    }

    suspend fun getTransformedImage(): Bitmap {
        return withContext(Dispatchers.IO) {
            val determinant = Points.getPoints(PointsState.START).determinant3d()
            val factorMatrix = calculateFactorMatrix(determinant)
            val translationMatrix = calculateTranslationMatrix(determinant)

            transformImage(factorMatrix, translationMatrix)
        }
    }

    suspend fun transformImage(factorMatrix: FloatArray, translationMatrix: FloatArray): Bitmap {
        return withContext(Dispatchers.IO) {
            var maxX: Int = Int.MIN_VALUE
            var minX: Int = Int.MAX_VALUE
            var maxY: Int = Int.MIN_VALUE
            var minY: Int = Int.MAX_VALUE

            for (x in 0 until width - 1) {
                for (y in 0 until height - 1) {
                    val newCoordinates = getTransformedCoordinates(
                        x.toFloat(),
                        y.toFloat(),
                        factorMatrix,
                        translationMatrix
                    )
                    if (newCoordinates[0] > maxX) maxX = newCoordinates[0]
                    if (newCoordinates[0] < minX) minX = newCoordinates[0]
                    if (newCoordinates[1] > maxY) maxY = newCoordinates[1]
                    if (newCoordinates[1] < minY) minY = newCoordinates[1]
                }
            }

            maxX -= minX
            maxY -= minY

            val newImageData = ImageData(IntArray((maxX + 1) * (maxY + 1)), maxX + 1, maxY + 1)

            for (x in 0 until width - 1) {
                for (y in 0 until height - 1) {
                    val curPixel = imageData.getPixel(x, y)
                    val newCoordinates = getTransformedCoordinates(
                        x.toFloat(),
                        y.toFloat(),
                        factorMatrix,
                        translationMatrix
                    )
                    newImageData.setPixel(
                        curPixel,
                        newCoordinates[0] - minX,
                        newCoordinates[1] - minY
                    )
                }
            }

            Bitmap.createBitmap(
                newImageData.pixels,
                newImageData.width,
                newImageData.height,
                Bitmap.Config.ARGB_8888
            )
        }
    }

    private fun getTransformedCoordinates(
        x: Float,
        y: Float,
        factorMatrix: FloatArray,
        translationMatrix: FloatArray
    ): IntArray {
        val newX = factorMatrix[0] * x + factorMatrix[1] * y + translationMatrix[0]
        val newY = factorMatrix[2] * x + factorMatrix[3] * y + translationMatrix[1]

        return intArrayOf(newX.toInt(), newY.toInt())
    }

    private fun calculateFactorMatrix(determinant: Float): FloatArray {
        val xFactor = calculateXFactor(determinant)
        val yFactor = calculateYFactor(determinant)

        return floatArrayOf(xFactor[0], yFactor[0], xFactor[1], yFactor[1])
    }

    private fun calculateXFactor(determinant: Float): FloatArray {
        val startYs = Points.getPoints(PointsState.START).getYs()
        val yPoints = listOf(Point(startYs[0], 1F), Point(startYs[1], 1F), Point(startYs[2], 1F))

        return calculateMatrix(determinant, yPoints)
    }

    private fun calculateYFactor(determinant: Float): FloatArray {
        val startXs = Points.getPoints(PointsState.START).getXs()
        val xPoints = listOf(Point(startXs[0], 1F), Point(startXs[1], 1F), Point(startXs[2], 1F))
        val factor = calculateMatrix(determinant, xPoints)

        return floatArrayOf(-factor[0], -factor[1])
    }

    private fun calculateTranslationMatrix(determinant: Float): FloatArray {
        val startPoints = Points.getPoints(PointsState.START)

        return calculateMatrix(determinant, startPoints)
    }

    private fun calculateMatrix(determinant: Float, points: List<Point>): FloatArray {
        val finishXCoordinates = Points.getPoints(PointsState.FINISH).getXs()
        val finishYCoordinates = Points.getPoints(PointsState.FINISH).getYs()

        return floatArrayOf(
            getComponent(finishXCoordinates, points) / determinant,
            getComponent(finishYCoordinates, points) / determinant
        )
    }

    private fun getComponent(finishCoordinates: FloatArray, points: List<Point>): Float {
        return finishCoordinates[0] * determinant2d(points[1], points[2]) -
                finishCoordinates[1] * determinant2d(points[0], points[2]) +
                finishCoordinates[2] * determinant2d(points[0], points[1])
    }

    private fun List<Point>.getXs(): FloatArray {
        return floatArrayOf(this[0].x, this[1].x, this[2].x)
    }

    private fun List<Point>.getYs(): FloatArray {
        return floatArrayOf(this[0].y, this[1].y, this[2].y)
    }

    private fun List<Point>.determinant3d(): Float {
        val plusTriples = this[0].x * this[1].y + this[0].y * this[2].x + this[1].x * this[2].y
        val minusTriples = this[1].y * this[2].x + this[2].y * this[0].x + this[0].x * this[1].x
        return plusTriples - minusTriples
    }

    private fun determinant2d(point1: Point, point2: Point): Float {
        return point1.x * point2.y - point1.y * point2.x
    }
}