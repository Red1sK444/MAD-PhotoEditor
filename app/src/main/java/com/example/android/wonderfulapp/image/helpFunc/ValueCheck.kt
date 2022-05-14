package com.example.android.wonderfulapp.image.helpFunc

class ValueCheck {
    fun inRange(x: Int, xMax: Int): Boolean {
        return (0 <= x) && (x < xMax)
    }
}