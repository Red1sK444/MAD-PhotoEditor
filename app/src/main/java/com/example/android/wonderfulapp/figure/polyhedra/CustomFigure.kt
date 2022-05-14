package com.example.android.wonderfulapp.figure.polyhedra

import com.example.android.wonderfulapp.figure.model3D.Drawer
import com.example.android.wonderfulapp.figure.model3D.Face
import com.example.android.wonderfulapp.figure.model3D.Figure
import com.example.android.wonderfulapp.figure.model3D.Node
import com.example.android.wonderfulapp.figure.values.Colors
import com.example.android.wonderfulapp.figure.values.Digits
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

class CustomFigure(private val faceCount: Int) : Figure() {

    override var nodes = when (faceCount % 2) {
        0 -> {
            MutableList(faceCount / 2 + 2) { Node() }
        }
        else -> {
            MutableList(faceCount) { Node() }
        }
    }

    override var faces = MutableList(faceCount) { Face() }

    init {
        fillNodes()
        fillFaces()
    }

    private fun fillNodes() {
        when (faceCount % 2) {
            0 -> {
                for (k in 0 until (nodes.size - 2)) {
                    nodes[k].x = cos(2 * PI * k / (nodes.size - 2)).toFloat() * Drawer.REMOTENESS
                    nodes[k].z = sin(2 * PI * k / (nodes.size - 2)).toFloat() * Drawer.REMOTENESS
                }

                nodes[nodes.size - 2].y = Drawer.REMOTENESS
                nodes[nodes.size - 1].y = -Drawer.REMOTENESS
            }
            1 -> {
                for (k in 0 until (nodes.size - 1)) {
                    nodes[k].x = cos(2 * PI * k / (nodes.size - 1)).toFloat() * 0.866f * Drawer.REMOTENESS
                    nodes[k].z = sin(2 * PI * k / (nodes.size - 1)).toFloat() * 0.866f * Drawer.REMOTENESS
                    nodes[k].y = Drawer.REMOTENESS / 2
                }

                nodes[nodes.size - 1].y = -Drawer.REMOTENESS
            }
        }
    }

    private fun fillFaces() {
        when (faceCount % 2) {
            0 -> {
                var faceInd = 0
                for (nodeInd in 0 until (nodes.size - 2)) {
                    faces[faceInd] = Face(
                        mutableListOf(
                            nodes[(nodeInd + 1) % (nodes.size - 2)],
                            nodes[nodeInd],
                            nodes[nodes.size - 1]
                        ), Colors.colorByInd(faceInd), Digits.digitByInd(faceInd)
                    )
                    faces[faceInd + 1] = Face(
                        mutableListOf(
                            nodes[nodeInd],
                            nodes[(nodeInd + 1) % (nodes.size - 2)],
                            nodes[nodes.size - 2]
                        ), Colors.colorByInd(faceInd + 1), Digits.digitByInd(faceInd + 1)
                    )
                    faceInd += 2
                }
            }
            1 -> {
                var faceInd = 0
                val base = MutableList(0) { Node(0f, 0f, 0f) }
                for (nodeInd in 0 until (nodes.size - 1)) {
                    base.add(nodeInd, nodes[nodeInd])
                }
                faces[faceInd] = Face(base, Colors.colorByInd(faceInd), Digits.digitByInd(faceInd))

                faceInd++
                for (nodeInd in 0 until (nodes.size - 1)) {
                    faces[faceInd] = Face(
                        mutableListOf(
                            nodes[(nodeInd + 1) % (nodes.size - 1)],
                            nodes[nodeInd],
                            nodes[nodes.size - 1]
                        ), Colors.colorByInd(faceInd), Digits.digitByInd(faceInd)
                    )
                    faceInd++
                }
            }
        }
    }
}