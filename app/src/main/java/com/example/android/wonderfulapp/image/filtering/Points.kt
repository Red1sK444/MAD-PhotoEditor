package com.example.android.wonderfulapp.image.filtering

import android.graphics.Color

data class Point(var x: Float, var y: Float, var radius: Float = 16F, var color: Int = Color.RED)

enum class PointsState {
    START, FINISH;

    fun toggle(): PointsState {
        return if (this == START)
            FINISH
        else
            START
    }
}

object Points {
    private val pointColors = listOf(Color.RED, Color.GREEN, Color.BLUE)
    private var state = PointsState.START
    private val startPoints: MutableList<Point> = mutableListOf()
    private val finishPoints: MutableList<Point> = mutableListOf()

    fun getCurrentPoints(): List<Point> {
        return getPoints(state)
    }

    fun getPoints(thisState: PointsState): List<Point> {
        return when (thisState) {
            PointsState.START -> {
                startPoints.toList()
            }
            PointsState.FINISH -> {
                finishPoints.toList()
            }
        }
    }

    fun addPoint(element: Point): Boolean {
        return when (state) {
            PointsState.START -> {
                addPointToList(element, startPoints)
            }
            PointsState.FINISH -> {
                addPointToList(element, finishPoints)
            }
        }
    }

    fun clearAllPoints() {
        clearPoints(startPoints)
        clearPoints(finishPoints)
    }

    fun clearCurrentStatePoints() {
        when (state) {
            PointsState.START -> {
                clearPoints(startPoints)
            }
            PointsState.FINISH -> {
                clearPoints(finishPoints)
            }
        }
    }

    fun toggleState(): PointsState {
        state = state.toggle()
        return state
    }

    private fun addPointToList(element: Point, list: MutableList<Point>): Boolean {
        return if (list.size < 3) {
            element.color = pointColors[list.size]
            list.add(element)
            true
        } else {
            false
        }
    }

    private fun clearPoints(list: MutableList<Point>) {
        list.clear()
    }
}