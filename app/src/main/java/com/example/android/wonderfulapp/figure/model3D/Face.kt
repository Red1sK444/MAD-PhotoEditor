package com.example.android.wonderfulapp.figure.model3D

class Face(
    val nodes: MutableList<Node> = mutableListOf(Node(), Node(), Node()),
    val color: Int = 1,
    val digit: Array<Int> = arrayOf(1)
) {
    val perspectiveNodes: MutableList<Node> = MutableList(0) { Node(0F, 0F, 0F) }
    val digitPoints: Array<Node>

    init {
        val myCenter = GeometricalFunctions.center(nodes)

        val baseCenter = Node(
            (nodes[0].x + nodes[1].x) / 2,
            (nodes[0].y + nodes[1].y) / 2,
            (nodes[0].z + nodes[1].z) / 2
        )

        val directionVector =
            Node(myCenter.x - baseCenter.x, myCenter.y - baseCenter.y, myCenter.z - baseCenter.z)
        val normalizedDir = GeometricalFunctions.normalize(directionVector)

        val sideLength = GeometricalFunctions.calcLength(
            Node(
                nodes[0].x - nodes[1].x,
                nodes[0].y - nodes[1].y,
                nodes[0].z - nodes[1].z
            )
        )

        val leftBase = GeometricalFunctions.findBasePoint(nodes, sideLength * 0.65f)
        val leftCenterBase = GeometricalFunctions.findBasePoint(nodes, sideLength * 0.55f)
        val rightCenterBase = GeometricalFunctions.findBasePoint(nodes, sideLength * 0.45f)
        val rightBase = GeometricalFunctions.findBasePoint(nodes, sideLength * 0.35f)

        val basicLength = GeometricalFunctions.calcLength(directionVector)

        val upperPoint = GeometricalFunctions.vectorMulti(normalizedDir, basicLength * 5f / 4)
        val medianPoint = GeometricalFunctions.vectorMulti(normalizedDir, basicLength)
        val lowerPoint = GeometricalFunctions.vectorMulti(normalizedDir, basicLength * 3f / 4)

        digitPoints = arrayOf(
            GeometricalFunctions.shiftVectorStart(upperPoint, leftBase),
            GeometricalFunctions.shiftVectorStart(medianPoint, leftBase),
            GeometricalFunctions.shiftVectorStart(lowerPoint, leftBase),
            GeometricalFunctions.shiftVectorStart(lowerPoint, leftCenterBase),
            GeometricalFunctions.shiftVectorStart(medianPoint, leftCenterBase),
            GeometricalFunctions.shiftVectorStart(upperPoint, leftCenterBase),
            GeometricalFunctions.shiftVectorStart(upperPoint, rightCenterBase),
            GeometricalFunctions.shiftVectorStart(medianPoint, rightCenterBase),
            GeometricalFunctions.shiftVectorStart(lowerPoint, rightCenterBase),
            GeometricalFunctions.shiftVectorStart(lowerPoint, rightBase),
            GeometricalFunctions.shiftVectorStart(medianPoint, rightBase),
            GeometricalFunctions.shiftVectorStart(upperPoint, rightBase)
        )
    }

    // 0 5 6 11
    // 1 4 7 10 - number
    // 2 3 8 9
}