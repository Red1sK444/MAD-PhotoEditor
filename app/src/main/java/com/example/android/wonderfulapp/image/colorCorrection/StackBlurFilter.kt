package com.example.android.wonderfulapp.image.colorCorrection

import android.graphics.Bitmap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.*

class StackBlurFilter : Filter() {
    override suspend fun apply(
        curBitmap: Bitmap
    ): Bitmap {
        return withContext(Dispatchers.IO) {
            val radius = 8
            val width = curBitmap.width
            val height = curBitmap.height
            val pixels = IntArray(width * height)
            val newPixels = IntArray(width * height)

            curBitmap.getPixels(pixels, 0, width, 0, 0, width, height)
            stackBlur(width, height, radius, pixels, newPixels)

            Bitmap.createBitmap(newPixels, width, height, Bitmap.Config.ARGB_8888)
        }
    }

    private fun stackBlur(
        width: Int,
        height: Int,
        radius: Int,
        pixels: IntArray,
        newPixels: IntArray
    ) {
        val denominator = radius * (radius + 2) + 1
        val tempPixels = IntArray(width * height)

        for (row in 0 until height) {

            val redPrevQueue = LinkedList<Int>()
            val greenPrevQueue = LinkedList<Int>()
            val bluePrevQueue = LinkedList<Int>()

            val redNextQueue = LinkedList<Int>()
            val greenNextQueue = LinkedList<Int>()
            val blueNextQueue = LinkedList<Int>()

            var redPrevSum = 0
            var redCurSum = 0
            var redNextSum = 0

            var greenPrevSum = 0
            var greenCurSum = 0
            var greenNextSum = 0

            var bluePrevSum = 0
            var blueCurSum = 0
            var blueNextSum = 0

            val startInd = row * width
            val curPixel = pixels[startInd]
            val red = curPixel ushr 16 and 0xFF
            val green = curPixel ushr 8 and 0xFF
            val blue = curPixel and 0xFF

            repeat(radius + 1) {

                redPrevQueue.add(red)
                greenPrevQueue.add(green)
                bluePrevQueue.add(blue)

                redPrevSum += red
                greenPrevSum += green
                bluePrevSum += blue

                redCurSum += (it + 1) * red
                greenCurSum += (it + 1) * green
                blueCurSum += (it + 1) * blue
            }

            for (addNum in 1..radius) {

                val nextInd = startInd + if (addNum > width - 1) width - 1 else addNum
                val nextPixel = pixels[nextInd]
                val nextRed = nextPixel ushr 16 and 0xFF
                val nextGreen = nextPixel ushr 8 and 0xFF
                val nextBlue = nextPixel and 0xFF

                redNextQueue.add(nextRed)
                greenNextQueue.add(nextGreen)
                blueNextQueue.add(nextBlue)

                redNextSum += nextRed
                greenNextSum += nextGreen
                blueNextSum += nextBlue

                val multiplier = radius - addNum + 1
                redCurSum += nextRed * multiplier
                greenCurSum += nextGreen * multiplier
                blueCurSum += nextBlue * multiplier
            }

            for (col in 0 until width) {

                val newInd = width * row + col
                tempPixels[newInd] = (pixels[newInd] and -0x1000000) or
                        (((redCurSum / denominator) and 0xff shl 16)) or
                        (((greenCurSum / denominator) and 0xff shl 8)) or
                        (((blueCurSum / denominator) and 0xff))

                redCurSum -= redPrevSum
                greenCurSum -= greenPrevSum
                blueCurSum -= bluePrevSum

                val nextPixelInd =
                    if (col + radius + 1 > width - 1) row * width + width - 1 else newInd + radius + 1
                val nextPixel = pixels[nextPixelInd]
                val nextRed = nextPixel ushr 16 and 0xFF
                val nextGreen = nextPixel ushr 8 and 0xFF
                val nextBlue = nextPixel and 0xFF

                redNextQueue.add(nextRed)
                greenNextQueue.add(nextGreen)
                blueNextQueue.add(nextBlue)

                redNextSum += nextRed
                greenNextSum += nextGreen
                blueNextSum += nextBlue

                redCurSum += redNextSum
                greenCurSum += greenNextSum
                blueCurSum += blueNextSum

                redPrevSum -= redPrevQueue.remove()
                greenPrevSum -= greenPrevQueue.remove()
                bluePrevSum -= bluePrevQueue.remove()

                val deltaRed = redNextQueue.remove()
                val deltaGreen = greenNextQueue.remove()
                val deltaBlue = blueNextQueue.remove()

                redNextSum -= deltaRed
                greenNextSum -= deltaGreen
                blueNextSum -= deltaBlue

                redPrevQueue.add(deltaRed)
                greenPrevQueue.add(deltaGreen)
                bluePrevQueue.add(deltaBlue)

                redPrevSum += deltaRed
                greenPrevSum += deltaGreen
                bluePrevSum += deltaBlue
            }
        }

        for (col in 0 until width) {

            val redPrevQueue = LinkedList<Int>()
            val greenPrevQueue = LinkedList<Int>()
            val bluePrevQueue = LinkedList<Int>()

            val redNextQueue = LinkedList<Int>()
            val greenNextQueue = LinkedList<Int>()
            val blueNextQueue = LinkedList<Int>()

            var redPrevSum = 0
            var redCurSum = 0
            var redNextSum = 0

            var greenPrevSum = 0
            var greenCurSum = 0
            var greenNextSum = 0

            var bluePrevSum = 0
            var blueCurSum = 0
            var blueNextSum = 0

            val curPixel = tempPixels[col]
            val red = curPixel ushr 16 and 0xFF
            val green = curPixel ushr 8 and 0xFF
            val blue = curPixel and 0xFF

            repeat(radius + 1) {

                redPrevQueue.add(red)
                greenPrevQueue.add(green)
                bluePrevQueue.add(blue)

                redPrevSum += red
                greenPrevSum += green
                bluePrevSum += blue

                redCurSum += (it + 1) * red
                greenCurSum += (it + 1) * green
                blueCurSum += (it + 1) * blue
            }

            for (addNum in 1..radius) {

                val nextInd =
                    col + if (addNum > height - 1) width * (height - 1) else addNum * width
                val nextPixel = tempPixels[nextInd]
                val nextRed = nextPixel ushr 16 and 0xFF
                val nextGreen = nextPixel ushr 8 and 0xFF
                val nextBlue = nextPixel and 0xFF

                redNextQueue.add(nextRed)
                greenNextQueue.add(nextGreen)
                blueNextQueue.add(nextBlue)

                redNextSum += nextRed
                greenNextSum += nextGreen
                blueNextSum += nextBlue

                val multiplier = radius - addNum + 1
                redCurSum += nextRed * multiplier
                greenCurSum += nextGreen * multiplier
                blueCurSum += nextBlue * multiplier
            }

            for (row in 0 until height) {

                val newInd = width * row + col
                newPixels[newInd] = (tempPixels[newInd] and -0x1000000) or
                        (((redCurSum / denominator) and 0xff shl 16)) or
                        (((greenCurSum / denominator) and 0xff shl 8)) or
                        (((blueCurSum / denominator) and 0xff))

                redCurSum -= redPrevSum
                greenCurSum -= greenPrevSum
                blueCurSum -= bluePrevSum

                val nextPixelInd =
                    col + if (row + radius + 1 > height - 1) width * (row + 1) else width * (row + radius + 1) // row + 1

                if (nextPixelInd >= width * height) break

                val nextPixel = tempPixels[nextPixelInd]
                val nextRed = nextPixel ushr 16 and 0xFF
                val nextGreen = nextPixel ushr 8 and 0xFF
                val nextBlue = nextPixel and 0xFF

                redNextQueue.add(nextRed)
                greenNextQueue.add(nextGreen)
                blueNextQueue.add(nextBlue)

                redNextSum += nextRed
                greenNextSum += nextGreen
                blueNextSum += nextBlue

                redCurSum += redNextSum
                greenCurSum += greenNextSum
                blueCurSum += blueNextSum

                redPrevSum -= redPrevQueue.remove()
                greenPrevSum -= greenPrevQueue.remove()
                bluePrevSum -= bluePrevQueue.remove()

                val deltaRed = redNextQueue.remove()
                val deltaGreen = greenNextQueue.remove()
                val deltaBlue = blueNextQueue.remove()

                redNextSum -= deltaRed
                greenNextSum -= deltaGreen
                blueNextSum -= deltaBlue

                redPrevQueue.add(deltaRed)
                greenPrevQueue.add(deltaGreen)
                bluePrevQueue.add(deltaBlue)

                redPrevSum += deltaRed
                greenPrevSum += deltaGreen
                bluePrevSum += deltaBlue
            }
        }
    }
}