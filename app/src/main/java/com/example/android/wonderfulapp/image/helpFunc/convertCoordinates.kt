package com.example.android.wonderfulapp.image.helpFunc

import com.example.android.wonderfulapp.image.rotating.ImageParameters

class ConvertCoordinates {
    fun isNegative(coord: Coordinates): Boolean {
        return coord.x < 0 || coord.y < 0
    }

    fun convert(
        radius: Double,
        coord: Coordinates,
        imParam: ImageParameters,
        viewParam: ImageParameters
    ): Double {
        val scaleOrigTo: Double = if (viewParam.height - imParam.height > viewParam.width - imParam.width) {
            viewParam.width.toDouble() / imParam.width
        } else {
            viewParam.height.toDouble() / imParam.height
        }

        val newW = (imParam.width * scaleOrigTo).toInt()
        val newH = (imParam.height * scaleOrigTo).toInt()

        coord.y = (coord.y - ((viewParam.height - newH).toDouble() / 2)).toInt()
        coord.x = (coord.x - ((viewParam.width - newW).toDouble() / 2)).toInt()

        coord.y = (coord.y * (imParam.height.toDouble() / newH)).toInt()
        coord.x = (coord.x * (imParam.width.toDouble() / newW)).toInt()

        return radius / scaleOrigTo
    }
}