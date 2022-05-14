package com.example.android.wonderfulapp.image.filtering

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.SurfaceHolder
import android.view.SurfaceView
import com.example.android.wonderfulapp.image.ImageManager

class FilteringSurfaceView(
    context: Context,
    attributes: AttributeSet? = null,
) : SurfaceView(context, attributes), SurfaceHolder.Callback {

    private val paint: Paint = Paint(Paint.ANTI_ALIAS_FLAG)

    init {
        holder.addCallback(this)
        paint.color = Color.RED
        paint.style = Paint.Style.FILL
    }

    override fun surfaceCreated(holder: SurfaceHolder) {
        updateCanvas()
    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {}

    override fun surfaceDestroyed(holder: SurfaceHolder) {
        Points.clearAllPoints()
    }

    override fun onTouchEvent(event: MotionEvent?): Boolean {
        when (event?.action) {
            MotionEvent.ACTION_DOWN -> {
                Points.addPoint(Point(event.x, event.y))
                updateCanvas()
            }
            else -> {
                return super.onTouchEvent(event)
            }
        }
        return true
    }

    fun updateCanvas() {
        val canvas: Canvas = holder.lockCanvas()

        canvas.drawColor(Color.WHITE)
        ImageManager.thumbnail?.let {
            holder.setFixedSize(it.width, it.height)

            canvas.drawBitmap(it, 0F, 0F, paint)
            for (point in Points.getCurrentPoints()) {
                paint.color = point.color
                canvas.drawCircle(point.x, point.y, point.radius, paint)
            }
        }

        holder.unlockCanvasAndPost(canvas)
    }
}