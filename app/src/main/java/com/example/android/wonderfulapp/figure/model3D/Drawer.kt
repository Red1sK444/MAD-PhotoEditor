package com.example.android.wonderfulapp.figure.model3D

import android.graphics.*

object Drawer {
    private const val NODE_COLOR = "#1976d2"
    private const val EDGE_COLOR = Color.BLACK
    private const val NODE_SIZE = 10F
    private const val EDGE_WIDTH = 7F

    private val paint: Paint = Paint(Paint.ANTI_ALIAS_FLAG)
    const val REMOTENESS = 500F
    private const val DISTANCE = REMOTENESS * 4

    private fun drawNodes(canvas: Canvas, nodes: MutableList<Node>) {
        paint.color = Color.parseColor(NODE_COLOR)
        paint.style = Paint.Style.FILL

        //Draw nodes
        for (node in nodes) {
            canvas.drawCircle(node.x, node.y, NODE_SIZE, paint)
        }
    }

    private fun drawFace(canvas: Canvas, face: Face) {
        val path = Path().apply {
            fillType = Path.FillType.EVEN_ODD
            moveTo(face.perspectiveNodes[0].x, face.perspectiveNodes[0].y)
        }

        for (node in face.perspectiveNodes) {
            path.lineTo(node.x, node.y)
        }
        path.close()

        //Fill face
        paint.color = face.color
        paint.strokeWidth = EDGE_WIDTH
        paint.style = Paint.Style.FILL
        canvas.drawPath(path, paint)

        //Stroke face
        paint.color = EDGE_COLOR
        paint.style = Paint.Style.STROKE
        canvas.drawPath(path, paint)

        //Stroke digit
        val digitPath = Path()
        val z0 = (DISTANCE * 0.9F) / (DISTANCE - face.digitPoints[face.digit[0]].z)
        digitPath.moveTo(face.digitPoints[face.digit[0]].x * z0, z0 * face.digitPoints[face.digit[0]].y)

        var move = false
        for (dPoint in face.digit) {
            if (dPoint == 666) {
                move = true
                continue
            }

            val z = (DISTANCE * 0.9F) / (DISTANCE - face.digitPoints[dPoint].z)
            if (move) {
                digitPath.moveTo(face.digitPoints[dPoint].x * z, face.digitPoints[dPoint].y * z)
                move = false
            } else {
                digitPath.lineTo(face.digitPoints[dPoint].x * z, face.digitPoints[dPoint].y * z)
            }
        }
        canvas.drawPath(digitPath, paint)
    }

    private fun isVisible(face: Face): Boolean {

        for ((index, node) in face.nodes.withIndex()) {
            val z = (DISTANCE * 0.9F) / (DISTANCE - node.z)
            face.perspectiveNodes.add(index, Node(node.x * z, node.y * z, node.z))
        }

        val nodeA = face.perspectiveNodes[0]
        val nodeB = face.perspectiveNodes[1]
        val nodeC = face.perspectiveNodes[2]

        val quotientX = (nodeB.y * nodeC.z - nodeB.z * nodeC.y) +
                (nodeA.z * nodeC.y - nodeA.y * nodeC.z) +
                (nodeA.y * nodeB.z - nodeA.z * nodeB.y)

        val quotientY = (nodeB.z * nodeC.x - nodeB.x * nodeC.z) +
                (nodeA.x * nodeC.z - nodeA.z * nodeC.x) +
                (nodeA.z * nodeB.x - nodeA.x * nodeB.z)

        val quotientZ = (nodeB.x * nodeC.y - nodeB.y * nodeC.x) +
                (nodeA.y * nodeC.x - nodeA.x * nodeC.y) +
                (nodeA.x * nodeB.y - nodeA.y * nodeB.x)

        val d = -(quotientX * face.perspectiveNodes[0].x
                + quotientY * face.perspectiveNodes[0].y
                + quotientZ * face.perspectiveNodes[0].z)

        var visibility = if (quotientX * 0 + quotientY * 0 + quotientZ * 1 >= 0) 1 else -1
        if (quotientX + quotientY + quotientZ + d > 0) visibility *= -1

        return visibility == 1
    }

    fun drawVisible(canvas: Canvas, faces: MutableList<Face>) {
        for (face in faces) {
            if (isVisible(face)) {
                drawFace(canvas, face)
                drawNodes(canvas, face.perspectiveNodes)
            }
            face.perspectiveNodes.clear()
        }
    }
}