package com.example.android.wonderfulapp.figure.model3D

import kotlin.math.pow
import kotlin.math.sqrt

object GeometricalFunctions {

    fun center(nodes: MutableList<Node>): Node {
        val center = Node(0f, 0f, 0f)
        for (node in nodes) {
            center.x += node.x / nodes.size
            center.y += node.y / nodes.size
            center.z += node.z / nodes.size
        }
        return center
    }

    fun findBasePoint(nodes: MutableList<Node>, q: Float): Node {
        val dirVector =
            Node(nodes[0].x - nodes[1].x, nodes[0].y - nodes[1].y, nodes[0].z - nodes[1].z)
        val normVector = normalize(dirVector)
        val multiVector = vectorMulti(normVector, q)
        return Node(
            multiVector.x + nodes[1].x,
            multiVector.y + nodes[1].y,
            multiVector.z + nodes[1].z
        )
    }

    fun calcLength(vector: Node): Float =
        sqrt(vector.x.pow(2) + vector.y.pow(2) + vector.z.pow(2))

    fun normalize(node: Node): Node {
        val length = calcLength(node)
        return Node(node.x / length, node.y / length, node.z / length)
    }

    fun vectorMulti(vector: Node, q: Float): Node =
        Node(vector.x * q, vector.y * q, vector.z * q)

    fun shiftVectorStart(vector: Node, toPoint: Node): Node =
        Node(vector.x + toPoint.x, vector.y + toPoint.y, vector.z + toPoint.z)
}