package com.example.android.wonderfulapp.figure.values

import android.graphics.Color

object Colors {
    private val colors: MutableList<Int> = mutableListOf(
        Color.parseColor("#364F6B"),
        Color.parseColor("#6A2C70"),
        Color.parseColor("#B83B5E"),
        Color.parseColor("#F08A5D"),
        Color.parseColor("#F9ED69"),
        Color.parseColor("#00B8A9"),
        Color.parseColor("#FFB6B9"),
        Color.parseColor("#8D6262"),
        Color.parseColor("#F25D9C"),
        Color.parseColor("#A8D8EA"),
        Color.parseColor("#928A97"),
        Color.parseColor("#B61AAE"),
        Color.parseColor("#283149"),
        Color.parseColor("#FF6464"),
        Color.parseColor("#EBCBAE"),
        Color.parseColor("#D72323"),
        Color.parseColor("#A7FF83"),
        Color.parseColor("#FDFFAB"),
        Color.parseColor("#0D7377"),
        Color.parseColor("#FF2E63")
    )

    fun colorByInd(index: Int): Int {
        return colors[index]
    }
}