package com.example.android.wonderfulapp.figure.model3D

import kotlin.math.cos
import kotlin.math.sin

object Rotation3D {
    
    fun rotateZ3D(figure: Figure, theta: Float) {
        val sinTheta = sin(theta)
        val cosTheta = cos(theta)

        for (node in figure.nodes) {
            val newX = node.x * cosTheta - node.y * sinTheta
            val newY = node.y * cosTheta + node.x * sinTheta

            node.x = newX
            node.y = newY
        }

        for (face in figure.faces) {
            for (point in face.digitPoints) {
                val newX = point.x * cosTheta - point.y * sinTheta
                val newY = point.y * cosTheta + point.x * sinTheta

                point.x = newX
                point.y = newY
            }
        }
    }

    fun rotateX3D(figure: Figure, theta: Float) {
        val sinTheta = sin(theta)
        val cosTheta = cos(theta)

        for (node in figure.nodes) {
            val newY = node.y * cosTheta - node.z * sinTheta
            val newZ = node.z * cosTheta + node.y * sinTheta

            node.y = newY
            node.z = newZ
        }

        for (face in figure.faces) {
            for (point in face.digitPoints) {
                val newY = point.y * cosTheta - point.z * sinTheta
                val newZ = point.z * cosTheta + point.y * sinTheta

                point.y = newY
                point.z = newZ
            }
        }
    }

    fun rotateY3D(figure: Figure, theta: Float) {
        val sinTheta = sin(theta)
        val cosTheta = cos(theta)

        for (node in figure.nodes) {
            val newX = node.x * cosTheta - node.z * sinTheta
            val newZ = node.z * cosTheta + node.x * sinTheta

            node.z = newZ
            node.x = newX
        }

        for (face in figure.faces) {
            for (point in face.digitPoints) {
                val newX = point.x * cosTheta - point.z * sinTheta
                val newZ = point.z * cosTheta + point.x * sinTheta

                point.z = newZ
                point.x = newX
            }
        }
    }
}