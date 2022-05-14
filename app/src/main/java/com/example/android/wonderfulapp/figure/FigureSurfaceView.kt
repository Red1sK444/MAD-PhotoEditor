package com.example.android.wonderfulapp.figure

import android.content.Context
import android.graphics.Canvas
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.SurfaceHolder
import android.view.SurfaceView
import com.example.android.wonderfulapp.figure.model3D.Rotation3D
import com.example.android.wonderfulapp.figure.polyhedra.Cube
import com.example.android.wonderfulapp.figure.model3D.Figure
import kotlin.math.PI
import kotlin.properties.Delegates

class FigureSurfaceView(
    context: Context,
    attributes: AttributeSet? = null,
) : SurfaceView(context, attributes), SurfaceHolder.Callback {

    private var figure: Figure = Cube

    fun changeFigure(figure: Figure) {
        this.figure = figure
        surfaceCreated(holder)
    }

    fun rotateByZ(direction: Int) {
        Rotation3D.rotateZ3D(figure, direction * (PI / 180).toFloat())
        figure.rotationZ += direction

        val canvas: Canvas = holder.lockCanvas()
        figure.draw(canvas)
        holder.unlockCanvasAndPost(canvas)
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
        val canvas: Canvas = holder.lockCanvas()
        figure.draw(canvas)
        holder.unlockCanvasAndPost(canvas)
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {}

    override fun surfaceDestroyed(holder: SurfaceHolder) {}

    private var prevX = 0f
    private var prevY = 0f

    override fun onTouchEvent(event: MotionEvent?): Boolean {
        when (event?.action) {
            MotionEvent.ACTION_DOWN -> {
                prevX = event.x
                prevY = event.y
            }
            MotionEvent.ACTION_MOVE -> {
                Rotation3D.rotateY3D(figure, -(event.x - prevX) / 100)
                Rotation3D.rotateX3D(figure, -(event.y - prevY) / 100)

                val canvas: Canvas = holder.lockCanvas()
                figure.draw(canvas)
                holder.unlockCanvasAndPost(canvas)

                prevX = event.x
                prevY = event.y
            }
            else -> {
                return super.onTouchEvent(event)
            }
        }
        return true
    }
}