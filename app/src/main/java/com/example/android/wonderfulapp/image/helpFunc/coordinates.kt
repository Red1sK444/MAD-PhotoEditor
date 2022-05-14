package com.example.android.wonderfulapp.image.helpFunc


import kotlin.math.pow

class Coordinates(var x: Int = 0, var y: Int = 0) {
    companion object {
        fun euclideanDist(x1: Int, y1: Int, x2: Int, y2: Int): Double {
            return kotlin.math.sqrt((x1.toDouble() - x2).pow(2.0) + (y1.toDouble() - y2).pow(2.0))
        }
    }
}