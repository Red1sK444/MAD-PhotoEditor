package com.example.android.wonderfulapp.figure.polyhedra

import com.example.android.wonderfulapp.figure.model3D.Drawer
import com.example.android.wonderfulapp.figure.model3D.Face
import com.example.android.wonderfulapp.figure.model3D.Figure
import com.example.android.wonderfulapp.figure.model3D.Node
import com.example.android.wonderfulapp.figure.values.Colors
import com.example.android.wonderfulapp.figure.values.Digits
import kotlin.math.sqrt

object Octahedron : Figure() {
    private val halfA = Drawer.REMOTENESS * sqrt(2F) / 2

    override var nodes = mutableListOf(
        Node(-halfA, 0F, -halfA),
        Node(-halfA, 0F, halfA),
        Node(halfA, 0F, halfA),
        Node(halfA, 0F, -halfA),
        Node(0F, Drawer.REMOTENESS, 0F),
        Node(0F, -Drawer.REMOTENESS, 0F)
    )

    override var faces = mutableListOf(
        Face(
            mutableListOf(nodes[1], nodes[0], nodes[4]),
            Colors.colorByInd(0),
            Digits.digitByInd(0)
        ),
        Face(
            mutableListOf(nodes[0], nodes[1], nodes[5]),
            Colors.colorByInd(1),
            Digits.digitByInd(1)
        ),
        Face(
            mutableListOf(nodes[2], nodes[1], nodes[4]),
            Colors.colorByInd(2),
            Digits.digitByInd(2)
        ),
        Face(
            mutableListOf(nodes[1], nodes[2], nodes[5]),
            Colors.colorByInd(3),
            Digits.digitByInd(3)
        ),
        Face(
            mutableListOf(nodes[3], nodes[2], nodes[4]),
            Colors.colorByInd(4),
            Digits.digitByInd(4)
        ),
        Face(
            mutableListOf(nodes[2], nodes[3], nodes[5]),
            Colors.colorByInd(5),
            Digits.digitByInd(5)
        ),
        Face(
            mutableListOf(nodes[0], nodes[3], nodes[4]),
            Colors.colorByInd(6),
            Digits.digitByInd(6)
        ),
        Face(
            mutableListOf(nodes[3], nodes[0], nodes[5]),
            Colors.colorByInd(7),
            Digits.digitByInd(7)
        )
    )
}