package com.example.android.wonderfulapp.figure.polyhedra

import com.example.android.wonderfulapp.figure.model3D.Drawer
import com.example.android.wonderfulapp.figure.model3D.Face
import com.example.android.wonderfulapp.figure.model3D.Figure
import com.example.android.wonderfulapp.figure.model3D.Node
import com.example.android.wonderfulapp.figure.values.Colors
import com.example.android.wonderfulapp.figure.values.Digits
import kotlin.math.sqrt

object Tetrahedron : Figure() {
    private const val H = Drawer.REMOTENESS * 3F / 2F
    private val a = H * 3F / sqrt(6F)
    private val h = a * sqrt(3F) / 2F

    override var nodes = mutableListOf(
        Node(-a / 2F, H / 3F, -h / 3F),
        Node(a / 2F, H / 3F, -h / 3F),
        Node(0F, H / 3F, h * 2F / 3F),
        Node(0F, -H * 2F / 3F, 0F)
    )

    override var faces = mutableListOf(
        Face(
            mutableListOf(nodes[0], nodes[1], nodes[2]),
            Colors.colorByInd(0),
            Digits.digitByInd(0)
        ),
        Face(
            mutableListOf(nodes[1], nodes[0], nodes[3]),
            Colors.colorByInd(1),
            Digits.digitByInd(1)
        ),
        Face(
            mutableListOf(nodes[0], nodes[2], nodes[3]),
            Colors.colorByInd(2),
            Digits.digitByInd(2)
        ),
        Face(
            mutableListOf(nodes[2], nodes[1], nodes[3]),
            Colors.colorByInd(3),
            Digits.digitByInd(3)
        )
    )
}