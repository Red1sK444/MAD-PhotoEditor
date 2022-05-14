package com.example.android.wonderfulapp.figure.model3D

import android.graphics.Canvas
import android.graphics.Color
import com.example.android.wonderfulapp.figure.model3D.Drawer
import com.example.android.wonderfulapp.figure.model3D.Face
import com.example.android.wonderfulapp.figure.model3D.Node

abstract class Figure {
    abstract var nodes: MutableList<Node>
    abstract var faces: MutableList<Face>
    var rotationZ = 0

    fun draw(canvas: Canvas) {
        canvas.translate((canvas.width / 2).toFloat(), (canvas.height / 2).toFloat())
        canvas.drawColor(Color.WHITE)

        Drawer.drawVisible(canvas, faces)
    }
}