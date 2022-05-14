package com.example.android.wonderfulapp.figure.values

object Digits {
    private val digits: MutableList<Array<Int>> = mutableListOf(
        arrayOf(6, 8),
        arrayOf(5, 6, 7, 4, 3, 8),
        arrayOf(5, 6, 7, 4, 7, 8, 3),
        arrayOf(5, 4, 7, 6, 8),
        arrayOf(6, 5, 4, 7, 8, 3),
        arrayOf(6, 5, 3, 8, 7, 4),
        arrayOf(5, 6, 8),
        arrayOf(5, 6, 8, 3, 5, 4, 7),
        arrayOf(3, 8, 6, 5, 4, 7),
        arrayOf(5, 3, 666, 6, 11, 9, 8, 6),
        arrayOf(5, 3, 666, 11, 9),
        arrayOf(5, 3, 666, 6, 11, 10, 7, 8, 9),
        arrayOf(5, 3, 666, 6, 11, 10, 7, 10, 9, 8),
        arrayOf(5, 3, 666, 6, 7, 10, 11, 9),
        arrayOf(5, 3, 666, 11, 6, 7, 10, 9, 8),
        arrayOf(5, 3, 666, 11, 6, 8, 9, 10, 7),
        arrayOf(5, 3, 666, 6, 11, 9),
        arrayOf(5, 3, 666, 6, 11, 9, 8, 6, 7, 10),
        arrayOf(5, 3, 666, 8, 9, 11, 6, 7, 10),
        arrayOf(0, 5, 4, 1, 2, 3, 666, 6, 11, 9, 8, 6)
    )

    fun digitByInd(index: Int): Array<Int> {
        return digits[index]
    }
}