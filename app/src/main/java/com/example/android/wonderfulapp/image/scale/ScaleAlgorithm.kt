package com.example.android.wonderfulapp.image.scale

import android.graphics.Bitmap
import android.graphics.Color
import androidx.core.content.res.ResourcesCompat
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlin.math.floor
import kotlin.math.pow

class ScaleAlgorithm(private val image: Bitmap) {
    private val width: Int = image.width
    private val height: Int = image.height
    private val imageData: ImageData = ImageData(IntArray(width * height), width, height)
    private lateinit var mipmap: List<ImageData>

    init {
        image.getPixels(imageData.pixels, 0, width, 0, 0, width, height)
    }

    fun initMipmap() {
        val tempMipmap: MutableList<ImageData> = mutableListOf(imageData)
        var reducedWidth = width / 2
        var reducedHeight = height / 2

        while (reducedWidth >= 1 && reducedHeight >= 1) {
            tempMipmap.add(bilinearArrayScale(reducedWidth, reducedHeight))
            reducedWidth /= 2
            reducedHeight /= 2
        }

        mipmap = tempMipmap
    }

    suspend fun scaleImage(ratio: Float): Bitmap {
        val newWidth = (width * ratio).toInt()
        val newHeight = (height * ratio).toInt()
        if (newWidth == 0 || newHeight == 0 || ratio == 1F) return image

        return withContext(Dispatchers.IO) {
            if (ratio > 1) {
                bilinearScale(newWidth, newHeight)
            } else {
                var index = findLargerImageIndex(ratio)
                try {
                    index = index.coerceAtMost(mipmap.size - 2)
                    trilinearScale(mipmap[index], mipmap[index + 1], newWidth, newHeight)
                } catch (e: Exception) {
                    index = index.coerceAtMost(getMipmapSize() - 2)
                    val references = getReferences(index)
                    trilinearScale(
                        references[0],
                        references[1],
                        newWidth,
                        newHeight
                    )
                }
            }
        }
    }

    private fun findLargerImageIndex(ratio: Float): Int {
        val index = kotlin.math.log(1F / ratio, 2F)
        return floor(index).toInt()
    }

    private fun getMipmapSize(): Int {
        var reducedWidth = width / 2
        var reducedHeight = height / 2
        var size = 1

        while (reducedWidth >= 1 && reducedHeight >= 1) {
            size++
            reducedWidth /= 2
            reducedHeight /= 2
        }

        return size
    }

    private fun getReferences(pow: Int): Array<ImageData> {
        val factor = 2F.pow(pow).toInt()
        val largerImage = bilinearArrayScale(width / factor, height / factor)
        val smallerImage = bilinearArrayScale(width / (factor * 2), height / (factor * 2))

        return arrayOf(largerImage, smallerImage)
    }

    private fun bilinearScale(newWidth: Int, newHeight: Int): Bitmap {
        return Bitmap.createBitmap(
            bilinearArrayScale(newWidth, newHeight).pixels,
            newWidth,
            newHeight,
            Bitmap.Config.ARGB_8888
        )
    }

    private fun bilinearArrayScale(newWidth: Int, newHeight: Int): ImageData {
        val wRatio = (width - 1).toFloat() / newWidth
        val hRatio = (height - 1).toFloat() / newHeight
        val newImageData = ImageData(IntArray(newWidth * newHeight), newWidth, newHeight)

        for (row in 0 until newHeight) {
            for (col in 0 until newWidth) {
                val x = (wRatio * col).toInt()
                val y = (hRatio * row).toInt()

                val xDiff = (wRatio * col) - x
                val yDiff = (hRatio * row) - y

                val a = imageData.getPixel(x, y)
                val b = imageData.getPixel(x + 1, y)
                val c = imageData.getPixel(x, y + 1)
                val d = imageData.getPixel(x + 1, y + 1)

                val blue: Int =
                    (a.blue() * (1 - xDiff) * (1 - yDiff) + b.blue() * xDiff * (1 - yDiff) + c.blue() * yDiff * (1 - xDiff) + d.blue() * xDiff * yDiff).toInt()

                val green: Int =
                    (a.green() * (1 - xDiff) * (1 - yDiff) + b.green() * xDiff * (1 - yDiff) + c.green() * yDiff * (1 - xDiff) + d.green() * xDiff * yDiff).toInt()

                val red: Int =
                    (a.red() * (1 - xDiff) * (1 - yDiff) + b.red() * xDiff * (1 - yDiff) + c.red() * yDiff * (1 - xDiff) + d.red() * xDiff * yDiff).toInt()

                val alpha: Int =
                    (a.alpha() * (1 - xDiff) * (1 - yDiff) + b.alpha() * xDiff * (1 - yDiff) + c.alpha() * yDiff * (1 - xDiff) + d.alpha() * xDiff * yDiff).toInt()

                val color = Color.argb(alpha, red, green, blue)
                newImageData.setPixel(color, col, row)
            }
        }

        return newImageData
    }

    private fun trilinearScale(
        largerImage: ImageData,
        smallerImage: ImageData,
        newWidth: Int,
        newHeight: Int
    ): Bitmap {
        val newImagePixels = IntArray(newWidth * newHeight)
        var offset = 0

        val largerWRatio = (largerImage.width - 1).toFloat() / newWidth
        val largerHRatio = (largerImage.height - 1).toFloat() / newHeight
        val smallerWRatio = (smallerImage.width - 1).toFloat() / newWidth
        val smallerHRatio = (smallerImage.height - 1).toFloat() / newHeight

        val hDiff =
            (largerImage.width - newWidth).toFloat() / (largerImage.width - smallerImage.width)

        for (row in 0 until newHeight) {
            for (col in 0 until newWidth) {
                val xLarger = (largerWRatio * col).toInt()
                val yLarger = (largerHRatio * row).toInt()

                val xSmaller = (smallerWRatio * col).toInt()
                val ySmaller = (smallerHRatio * row).toInt()

                val wLargerDiff = (largerWRatio * col) - xLarger
                val hLargerDiff = (largerHRatio * row) - yLarger

                val wSmallerDiff = (smallerWRatio * col) - xSmaller
                val hSmallerDiff = (smallerHRatio * row) - ySmaller

                val a = largerImage.getPixel(xLarger, yLarger)
                val b = largerImage.getPixel(xLarger + 1, yLarger)
                val c = largerImage.getPixel(xLarger, yLarger + 1)
                val d = largerImage.getPixel(xLarger + 1, yLarger + 1)

                val e = smallerImage.getPixel(xSmaller, ySmaller)
                val f = smallerImage.getPixel(xSmaller + 1, ySmaller)
                val g = smallerImage.getPixel(xSmaller, ySmaller + 1)
                val h = smallerImage.getPixel(xSmaller + 1, ySmaller + 1)

                val blue: Int =
                    (a.blue() * (1 - wLargerDiff) * (1 - hLargerDiff) * (1 - hDiff) + b.blue() * wLargerDiff * (1 - hLargerDiff) * (1 - hDiff) + c.blue() * hLargerDiff * (1 - wLargerDiff) * (1 - hDiff) + d.blue() * wLargerDiff * hLargerDiff * (1 - hDiff) + e.blue() * (1 - wSmallerDiff) * (1 - hSmallerDiff) * hDiff + f.blue() * wSmallerDiff * (1 - hSmallerDiff) * hDiff + g.blue() * hSmallerDiff * (1 - wSmallerDiff) * hDiff + h.blue() * wSmallerDiff * hSmallerDiff * hDiff).toInt()

                val green: Int =
                    (a.green() * (1 - wLargerDiff) * (1 - hLargerDiff) * (1 - hDiff) + b.green() * wLargerDiff * (1 - hLargerDiff) * (1 - hDiff) + c.green() * hLargerDiff * (1 - wLargerDiff) * (1 - hDiff) + d.green() * wLargerDiff * hLargerDiff * (1 - hDiff) + e.green() * (1 - wSmallerDiff) * (1 - hSmallerDiff) * hDiff + f.green() * wSmallerDiff * (1 - hSmallerDiff) * hDiff + g.green() * hSmallerDiff * (1 - wSmallerDiff) * hDiff + h.green() * wSmallerDiff * hSmallerDiff * hDiff).toInt()

                val red: Int =
                    (a.red() * (1 - wLargerDiff) * (1 - hLargerDiff) * (1 - hDiff) + b.red() * wLargerDiff * (1 - hLargerDiff) * (1 - hDiff) + c.red() * hLargerDiff * (1 - wLargerDiff) * (1 - hDiff) + d.red() * wLargerDiff * hLargerDiff * (1 - hDiff) + e.green() * (1 - wSmallerDiff) * (1 - hSmallerDiff) * hDiff + f.red() * wSmallerDiff * (1 - hSmallerDiff) * hDiff + g.red() * hSmallerDiff * (1 - wSmallerDiff) * hDiff + h.red() * wSmallerDiff * hSmallerDiff * hDiff).toInt()

                val alpha: Int =
                    (a.alpha() * (1 - wLargerDiff) * (1 - hLargerDiff) * (1 - hDiff) + b.alpha() * wLargerDiff * (1 - hLargerDiff) * (1 - hDiff) + c.alpha() * hLargerDiff * (1 - wLargerDiff) * (1 - hDiff) + d.alpha() * wLargerDiff * hLargerDiff * (1 - hDiff) + e.alpha() * (1 - wSmallerDiff) * (1 - hSmallerDiff) * hDiff + f.alpha() * wSmallerDiff * (1 - hSmallerDiff) * hDiff + g.alpha() * hSmallerDiff * (1 - wSmallerDiff) * hDiff + h.alpha() * wSmallerDiff * hSmallerDiff * hDiff).toInt()

                val color = Color.argb(alpha, red, green, blue)
                newImagePixels[offset++] = color
            }
        }


        return Bitmap.createBitmap(
            newImagePixels,
            newWidth,
            newHeight,
            Bitmap.Config.ARGB_8888
        )
    }

    private fun Int.alpha(): Int {
        return this shr 24 and 0xff
    }

    private fun Int.red(): Int {
        return this shr 16 and 0xff
    }

    private fun Int.green(): Int {
        return this shr 8 and 0xff
    }

    private fun Int.blue(): Int {
        return this and 0xff
    }
}