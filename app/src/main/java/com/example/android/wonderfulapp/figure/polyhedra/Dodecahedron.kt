package com.example.android.wonderfulapp.figure.polyhedra

import com.example.android.wonderfulapp.figure.model3D.Face
import com.example.android.wonderfulapp.figure.model3D.Figure
import com.example.android.wonderfulapp.figure.model3D.GeometricalFunctions
import com.example.android.wonderfulapp.figure.model3D.Node
import com.example.android.wonderfulapp.figure.values.Colors
import com.example.android.wonderfulapp.figure.values.Digits

object Dodecahedron : Figure() {
    override var nodes = MutableList(20) { Node(0f, 0f, 0f) }
    override var faces: MutableList<Face> = mutableListOf()

    init {
        fillNodes()
        faces = fillFaces()
    }

    private fun fillNodes() {
        for (i in 0..19) {
            val faceCenter = GeometricalFunctions.center(
                mutableListOf(
                    Icosahedron.faces[i].nodes[0],
                    Icosahedron.faces[i].nodes[1],
                    Icosahedron.faces[i].nodes[2]
                )
            )
            nodes[i] = Node(
                faceCenter.x,
                faceCenter.y,
                faceCenter.z
            )
        }
    }

    private fun fillFaces(): MutableList<Face> = mutableListOf(
        Face(
            mutableListOf(nodes[0], nodes[1], nodes[2], nodes[3], nodes[4]),
            Colors.colorByInd(0),
            Digits.digitByInd(0)
        ),
        Face(
            mutableListOf(nodes[0], nodes[1], nodes[7], nodes[6], nodes[5]),
            Colors.colorByInd(2),
            Digits.digitByInd(2)
        ),
        Face(
            mutableListOf(nodes[4], nodes[0], nodes[5], nodes[14], nodes[13]),
            Colors.colorByInd(1),
            Digits.digitByInd(1)
        ),
        Face(
            mutableListOf(nodes[1], nodes[2], nodes[9], nodes[8], nodes[7]),
            Colors.colorByInd(3),
            Digits.digitByInd(3)
        ),
        Face(
            mutableListOf(nodes[6], nodes[7], nodes[8], nodes[17], nodes[16]),
            Colors.colorByInd(4),
            Digits.digitByInd(4)
        ),
        Face(
            mutableListOf(nodes[5], nodes[6], nodes[16], nodes[15], nodes[14]),
            Colors.colorByInd(5),
            Digits.digitByInd(5)
        ),
        Face(
            mutableListOf(nodes[11], nodes[10], nodes[9], nodes[2], nodes[3]),
            Colors.colorByInd(11),
            Digits.digitByInd(11)
        ),
        Face(
            mutableListOf(nodes[12], nodes[11], nodes[3], nodes[4], nodes[13]),
            Colors.colorByInd(10),
            Digits.digitByInd(10)
        ),
        Face(
            mutableListOf(nodes[19], nodes[18], nodes[10], nodes[11], nodes[12]),
            Colors.colorByInd(8),
            Digits.digitByInd(8)
        ),
        Face(
            mutableListOf(nodes[14], nodes[15], nodes[19], nodes[12], nodes[13]),
            Colors.colorByInd(9),
            Digits.digitByInd(9)
        ),
        Face(
            mutableListOf(nodes[15], nodes[16], nodes[17], nodes[18], nodes[19]),
            Colors.colorByInd(6),
            Digits.digitByInd(6)
        ),
        Face(
            mutableListOf(nodes[17], nodes[8], nodes[9], nodes[10], nodes[18]),
            Colors.colorByInd(7),
            Digits.digitByInd(7)
        )
    )
}