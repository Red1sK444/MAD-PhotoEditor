package com.example.android.wonderfulapp.figure.polyhedra

import com.example.android.wonderfulapp.figure.model3D.Drawer
import com.example.android.wonderfulapp.figure.model3D.Face
import com.example.android.wonderfulapp.figure.model3D.Figure
import com.example.android.wonderfulapp.figure.model3D.Node
import com.example.android.wonderfulapp.figure.values.Colors
import com.example.android.wonderfulapp.figure.values.Digits
import kotlin.math.sqrt

object Cube : Figure() {
    private val halfA = Drawer.REMOTENESS / sqrt(3F)

    override var nodes = mutableListOf(
        Node(
            -halfA,
            -halfA,
            -halfA
        ),
        Node(
            -halfA,
            halfA,
            -halfA
        ),
        Node(
            halfA,
            halfA,
            -halfA
        ),
        Node(
            halfA,
            -halfA,
            -halfA
        ),
        Node(
            -halfA,
            -halfA,
            halfA
        ),
        Node(
            -halfA,
            halfA,
            halfA
        ),
        Node(
            halfA,
            halfA,
            halfA
        ),
        Node(
            halfA,
            -halfA,
            halfA
        ),
        Node(0f, -halfA / 2, halfA),
        Node(0f, halfA / 2, halfA),
        Node(-halfA / 2, 0f, halfA),
        Node(halfA / 2, 0f, halfA)
    )

    override var faces = mutableListOf(
        Face(
            mutableListOf(nodes[1], nodes[0], nodes[3], nodes[2]),
            Colors.colorByInd(0),
            Digits.digitByInd(0)
        ),
        Face(
            mutableListOf(nodes[0], nodes[1], nodes[5], nodes[4]),
            Colors.colorByInd(1),
            Digits.digitByInd(1)
        ),
        Face(
            mutableListOf(nodes[1], nodes[2], nodes[6], nodes[5]),
            Colors.colorByInd(2),
            Digits.digitByInd(2)
        ),
        Face(
            mutableListOf(nodes[2], nodes[3], nodes[7], nodes[6]),
            Colors.colorByInd(3),
            Digits.digitByInd(3)
        ),
        Face(
            mutableListOf(nodes[4], nodes[7], nodes[3], nodes[0]),
            Colors.colorByInd(4),
            Digits.digitByInd(4)
        ),
        Face(
            mutableListOf(nodes[5], nodes[6], nodes[7], nodes[4]),
            Colors.colorByInd(5),
            Digits.digitByInd(5)
        )
    )
}