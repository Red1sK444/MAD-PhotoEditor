package com.example.android.wonderfulapp.figure.polyhedra

import com.example.android.wonderfulapp.figure.model3D.*
import com.example.android.wonderfulapp.figure.values.Colors
import com.example.android.wonderfulapp.figure.values.Digits
import kotlin.math.*

object Icosahedron : Figure() {
    override var nodes = MutableList(12) { Node(0f, 0f, 0f) }

    override var faces = MutableList(20) {
        Face(
            mutableListOf(
                Node(0f, 0f, 0f),
                Node(0f, 0f, 0f),
                Node(0f, 0f, 0f)
            ), 1, arrayOf(1)
        )
    }

    init {
        fillNodes()
        fillFaces()
    }

    private fun fillNodes() {
        val sideLength = GeometricalFunctions.calcLength(
            Node(
                (cos(36 * PI / 180).toFloat() - cos(36 * 3 * PI / 180).toFloat()) * 0.866f * Drawer.REMOTENESS,
                0F,
                (sin(36 * PI / 180).toFloat() - sin(36 * 3 * PI / 180).toFloat()) * 0.866f * Drawer.REMOTENESS
            )
        )
        val height = sqrt(
            (sideLength.pow(2) - ((cos(36 * PI / 180) - cos(360 * PI / 180)) * 0.866f * Drawer.REMOTENESS).pow(
                2
            ) - ((sin(36 * PI / 180) - sin(
                360 * PI / 180
            )) * 0.866f * Drawer.REMOTENESS).pow(2)) / 4
        ).toFloat()
        val sphereRadius =
            sqrt(
                (cos(36 * PI / 180) * 0.866f * Drawer.REMOTENESS).pow(2)
                        + height.pow(2)
                        + (sin(36 * PI / 180) * 0.866f * Drawer.REMOTENESS).pow(
                    2
                )
            ).toFloat()

        for (i in 1..10) {
            nodes[i - 1] = Node(
                cos(i * 36 * PI / 180).toFloat() * 0.866f * Drawer.REMOTENESS,
                (-1f).pow(i) * height,
                sin(i * 36 * PI / 180).toFloat() * 0.866f * Drawer.REMOTENESS
            )
        }
        nodes[10] = Node(0f, -sphereRadius, 0f)
        nodes[11] = Node(0f, sphereRadius, 0f)
    }

    private fun fillFaces() {
        var faceInd = 0
        for (i in 2..10 step 2) {
            faces[faceInd] = Face(
                mutableListOf(nodes[i % 10], nodes[i - 2], nodes[10]),
                Colors.colorByInd(faceInd),
                Digits.digitByInd(faceInd)
            )
            faceInd++
        }

        for (i in 2..11) {
            if (i % 2 == 0) {
                faces[faceInd] = Face(
                    mutableListOf(nodes[i - 2], nodes[i % 10], nodes[(i - 1) % 10]),
                    Colors.colorByInd(faceInd),
                    Digits.digitByInd(faceInd)
                )
            } else {
                faces[faceInd] = Face(
                    mutableListOf(nodes[i % 10], nodes[i - 2], nodes[(i - 1) % 10]),
                    Colors.colorByInd(faceInd),
                    Digits.digitByInd(faceInd)
                )
            }
            faceInd++
        }

        faces[faceInd] = Face(
            mutableListOf(nodes[9], nodes[1], nodes[11]),
            Colors.colorByInd(faceInd),
            Digits.digitByInd(faceInd)
        )
        faceInd++

        for (i in 3..9 step 2) {
            faces[faceInd] = Face(
                mutableListOf(nodes[i - 2], nodes[i], nodes[11]),
                Colors.colorByInd(faceInd),
                Digits.digitByInd(faceInd)
            )
            faceInd++
        }
    }
}