package com.example.android.wonderfulapp.spline

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import android.widget.Toast
import com.example.android.wonderfulapp.R
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.sqrt

enum class State {
    ADD_POINT, DELETE_POINT, MOVE_POINT
}

class SplineView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    companion object {
        private const val POINT_COLOR = Color.RED
        private const val FL_POINT_COLOR = Color.MAGENTA
        private const val FL_STROKE_COLOR = Color.GRAY
        private const val STROKE_COLOR = Color.BLACK
        private const val SPLINE_COLOR = Color.RED
        private const val SELECTED_POINT_COLOR = Color.BLUE
        private const val POINT_SIZE = 15F
        private const val FL_POINT_SIZE = 10F
    }

    class Point(var x: Float, var y: Float)
    class Spline(val controlPoints: MutableList<Point>)

    private var points = MutableList(0) { Point(0F, 0F) }
    private var spline = Spline(points)
    private var transformed = false

    var state = State.ADD_POINT

    object SelectedPoint {
        var ind: Int = -1
        var show: Boolean = false
    }

    private val bitmap: Bitmap by lazy {
        Bitmap.createBitmap(this.width, this.height, Bitmap.Config.ARGB_8888)
    }
    lateinit var canvas: Canvas
    private val paint: Paint = Paint()

    override fun onDraw(canva: Canvas?) {
        canvas = Canvas(bitmap)
        drawCanvas()
        canva?.drawBitmap(bitmap, 0f, 0f, Paint())
    }

    override fun onTouchEvent(event: MotionEvent?): Boolean {

        when (event?.action) {
            MotionEvent.ACTION_DOWN -> {

                when (state) {
                    State.ADD_POINT -> {

                        val point = Point(
                            floor(event.x - canvas.width / 2),
                            floor(event.y - canvas.height / 2)
                        )
                        if (checkPoint(point)) {
                            spline.controlPoints.add(point)
                        }

                        invalidate()
                    }
                    State.DELETE_POINT -> {

                        val point = Point(
                            floor(event.x - canvas.width / 2),
                            floor(event.y - canvas.height / 2)
                        )
                        deletePoint(point)

                        invalidate()
                    }
                    State.MOVE_POINT -> {
                        getSelectedPoint(
                            floor(event.x - canvas.width / 2),
                            floor(event.y - canvas.height / 2)
                        )

                        invalidate()
                    }
                }
            }
            MotionEvent.ACTION_MOVE -> {
                if (state == State.MOVE_POINT && SelectedPoint.show) {
                    SelectedPoint.ind.let {
                        spline.controlPoints[SelectedPoint.ind].x =
                            when {
                                floor(event.x - canvas.width / 2) >= canvas.width / 2 - 15 ->
                                    (canvas.width / 2 - 15).toFloat()
                                floor(event.x - canvas.width / 2) <= -canvas.width / 2 + 15 ->
                                    (-canvas.width / 2 + 15).toFloat()
                                else ->
                                    floor(event.x - canvas.width / 2)
                            }

                        spline.controlPoints[SelectedPoint.ind].y =
                            when {
                                floor(event.y - canvas.height / 2) >= canvas.height / 2 - 15 ->
                                    (canvas.height / 2 - 15).toFloat()
                                floor(event.y - canvas.height / 2) <= -canvas.height / 2 + 15 ->
                                    (-canvas.height / 2 + 15).toFloat()
                                else ->
                                    floor(event.y - canvas.height / 2)
                            }

                        invalidate()
                    }
                }
            }
            else -> {
                return super.onTouchEvent(event)
            }
        }
        return true
    }

    private fun getSelectedPoint(x0: Float, y0: Float) {

        for (i in spline.controlPoints.indices) {
            if (abs(spline.controlPoints[i].x - x0) < 3 * POINT_SIZE && abs(spline.controlPoints[i].y - y0) < 3 * POINT_SIZE) {

                SelectedPoint.ind = i
                SelectedPoint.show = true
            }
        }
    }

    private fun drawCanvas() {

        canvas.drawColor(Color.WHITE)
        canvas.translate((canvas.width / 2).toFloat(), (canvas.height / 2).toFloat())
        if (!transformed) {
            drawCurve()
        } else {
            drawSpline()
        }
    }

    private fun drawCurve() {

        if (spline.controlPoints.size > 1) {
            drawAntiAliasedLine(
                spline.controlPoints[0],
                spline.controlPoints[1],
                FL_STROKE_COLOR
            )

            if (spline.controlPoints.size > 2) {
                drawAntiAliasedLine(
                    spline.controlPoints[spline.controlPoints.size - 2],
                    spline.controlPoints[spline.controlPoints.size - 1],
                    FL_STROKE_COLOR
                )
            }

            if (spline.controlPoints.size > 3) {
                var i = 1
                while (i < spline.controlPoints.size - 2) {
                    drawAntiAliasedLine(
                        spline.controlPoints[i],
                        spline.controlPoints[i + 1],
                        STROKE_COLOR
                    )
                    i++
                }
            }
        }

        if (spline.controlPoints.size > 0) {
            drawBorderPoints()
        }

        paint.color = POINT_COLOR
        var i = 1
        while (i < spline.controlPoints.size - 1) {
            canvas.drawCircle(
                spline.controlPoints[i].x,
                spline.controlPoints[i].y,
                POINT_SIZE,
                paint
            )
            i++
        }

        drawSelectedPoint()
    }

    private fun drawBorderPoints() {
        paint.color = FL_POINT_COLOR
        paint.style = Paint.Style.FILL
        canvas.drawCircle(
            spline.controlPoints[0].x, spline.controlPoints[0].y,
            FL_POINT_SIZE, paint
        )
        canvas.drawCircle(
            spline.controlPoints[spline.controlPoints.size - 1].x,
            spline.controlPoints[spline.controlPoints.size - 1].y,
            FL_POINT_SIZE,
            paint
        )
    }

    private fun drawSelectedPoint() {

        if (SelectedPoint.show) {
            paint.color = SELECTED_POINT_COLOR
            canvas.drawCircle(
                spline.controlPoints[SelectedPoint.ind].x,
                spline.controlPoints[SelectedPoint.ind].y,
                POINT_SIZE,
                paint
            )
        }
    }

    private fun drawSpline() {

        var firstPoint = spline.controlPoints[0]
        var t = 0.0f
        while (t <= 1f) {
            val point = getBorderSplinePoints(t, "left")
            val checker = when {
                abs(point.x) <= canvas.width / 2 - 3 && abs(point.y) <= canvas.height / 2 - 3 &&
                        abs(firstPoint.x) <= canvas.width / 2 - 3 && abs(firstPoint.y) <= canvas.height / 2 - 3 -> true
                else -> false
            }

            t += 0.025f
            if (checker) {
                drawAntiAliasedLine(
                    firstPoint,
                    point,
                    SPLINE_COLOR
                )
            }
            firstPoint = point
        }

        firstPoint = spline.controlPoints[spline.controlPoints.size - 2]
        t = 0.0f
        while (t <= 1f) {
            val point = getBorderSplinePoints(t, "right")
            val checker = when {
                abs(point.x) <= canvas.width / 2 - 3 && abs(point.y) <= canvas.height / 2 - 3 &&
                        abs(firstPoint.x) <= canvas.width / 2 - 3 && abs(firstPoint.y) <= canvas.height / 2 - 3 -> true
                else -> false
            }

            t += 0.025f
            if (checker) {
                drawAntiAliasedLine(
                    firstPoint,
                    point,
                    SPLINE_COLOR
                )
            }
            firstPoint = point
        }

        t = 0f
        firstPoint = spline.controlPoints[1]
        while (t <= spline.controlPoints.size - 3.0f) {

            val point = getSplinePoint(t)
            val checker = when {
                abs(point.x) <= canvas.width / 2 - 3 && abs(point.y) <= canvas.height / 2 - 3 &&
                        abs(firstPoint.x) <= canvas.width / 2 - 3 && abs(firstPoint.y) <= canvas.height / 2 - 3 -> true
                else -> false
            }

            t += 0.025f
            if (checker) {
                drawAntiAliasedLine(
                    firstPoint,
                    point,
                    SPLINE_COLOR
                )
            }
            firstPoint = point
        }

        drawCurve()
    }

    private fun checkPoint(curPoint: Point): Boolean {

        for (point in spline.controlPoints) {

            if (abs(point.x - curPoint.x) < 3 * POINT_SIZE && abs(point.y - curPoint.y) < 3 * POINT_SIZE) {
                return false
            }
        }

        return true
    }

    private fun deletePoint(curPoint: Point) {

        for (point in spline.controlPoints) {
            if (abs(point.x - curPoint.x) < 3 * POINT_SIZE && abs(point.y - curPoint.y) < 3 * POINT_SIZE) {
                spline.controlPoints.remove(point)
                break
            }
        }

        if (spline.controlPoints.size <= 3) {
            transformed = false
        }
    }

    private fun getSplinePoint(t0: Float): Point {

        val p1 = floor(t0).toInt() + 1
        val p0 = p1 - 1
        val p2 = p1 + 1
        val p3 = p1 + 2

        val t = t0 - t0.toInt()
        val tSquare = t * t
        val tCube = tSquare * t

        val q0 = -tCube + 2 * tSquare - t
        val q1 = 3 * tCube - 5 * tSquare + 2
        val q2 = -3 * tCube + 4 * tSquare + t
        val q3 = tCube - tSquare

        val x = 0.5f * (spline.controlPoints[p0].x * q0 + spline.controlPoints[p1].x * q1
                + spline.controlPoints[p2].x * q2 + spline.controlPoints[p3].x * q3)
        val y = 0.5f * (spline.controlPoints[p0].y * q0 + spline.controlPoints[p1].y * q1
                + spline.controlPoints[p2].y * q2 + spline.controlPoints[p3].y * q3)

        return Point(x, y)
    }

    private fun getBorderSplinePoints(t0: Float, border: String): Point {
        if (border == "left") {
            val p1 = 0
            val p0 = spline.controlPoints.size - 1
            val p2 = p1 + 1
            val p3 = p1 + 2

            val t = t0 - t0.toInt()
            val tSquare = t * t
            val tCube = tSquare * t

            val q0 = -tCube + 2 * tSquare - t
            val q1 = 3 * tCube - 5 * tSquare + 2
            val q2 = -3 * tCube + 4 * tSquare + t
            val q3 = tCube - tSquare

            val x = 0.5f * (spline.controlPoints[p0].x * q0 + spline.controlPoints[p1].x * q1
                    + spline.controlPoints[p2].x * q2 + spline.controlPoints[p3].x * q3)
            val y = 0.5f * (spline.controlPoints[p0].y * q0 + spline.controlPoints[p1].y * q1
                    + spline.controlPoints[p2].y * q2 + spline.controlPoints[p3].y * q3)

            return Point(x, y)
        } else {
            val p0 = spline.controlPoints.size - 3
            val p1 = p0 + 1
            val p2 = p0 + 2
            val p3 = 0

            val t = t0 - t0.toInt()
            val tSquare = t * t
            val tCube = tSquare * t

            val q0 = -tCube + 2 * tSquare - t
            val q1 = 3 * tCube - 5 * tSquare + 2
            val q2 = -3 * tCube + 4 * tSquare + t
            val q3 = tCube - tSquare

            val x = 0.5f * (spline.controlPoints[p0].x * q0 + spline.controlPoints[p1].x * q1
                    + spline.controlPoints[p2].x * q2 + spline.controlPoints[p3].x * q3)
            val y = 0.5f * (spline.controlPoints[p0].y * q0 + spline.controlPoints[p1].y * q1
                    + spline.controlPoints[p2].y * q2 + spline.controlPoints[p3].y * q3)

            return Point(x, y)
        }

    }

    private fun drawAntiAliasedLine(
        firstPoint: Point,
        secondPoint: Point,
        color: Int
    ) {

        var startX = firstPoint.x.toInt() + canvas.width / 2
        var startY = firstPoint.y.toInt() + canvas.height / 2
        var endX = secondPoint.x.toInt() + canvas.width / 2
        var endY = secondPoint.y.toInt() + canvas.height / 2
        val yMoreThenX = abs(startY - endY) > abs(startX - endX)

        if (yMoreThenX) {
            var temp = startX
            startX = startY
            startY = temp
            temp = endX
            endX = endY
            endY = temp
        }
        if (startX > endX) {
            var temp = startX
            startX = endX
            endX = temp
            temp = startY
            startY = endY
            endY = temp
        }

        val dx = (endX - startX).toFloat()
        val dy = (endY - startY).toFloat()

        var gradient = dy / dx
        if (dx == 0f) {
            gradient = 1f
        }

        var interY = startY + gradient

        if (yMoreThenX) {
            setPixel(startY, startX, color, 1f)
            setPixel(endY, endX, color, 1f)

            for (x in startX + 1 until endX) {
                setPixel(floor(interY).toInt(), x, color, 1 - floatPart(interY))
                setPixel(floor(interY).toInt() + 1, x, color, floatPart(interY))
                interY += gradient
            }

        } else {
            setPixel(startX, startY, color, 1f)
            setPixel(endX, endY, color, 1f)

            for (x in startX + 1 until endX) {
                setPixel(x, floor(interY).toInt(), color, 1 - floatPart(interY))
                setPixel(x, floor(interY).toInt() + 1, color, floatPart(interY))
                interY += gradient
            }
        }
    }

    private fun floatPart(x: Float): Float {
        return x - x.toInt()
    }

    private fun setPixel(
        x: Int,
        y: Int,
        color: Int,
        brightness: Float
    ) {
            val color = Color.argb((brightness * 255f).toInt(), Color.red(color), Color.green(color), Color.blue(color))
            bitmap.setPixel(x, y, color)
    }

    fun transformToSpline() {

        if (!transformed && spline.controlPoints.size > 3) {
            transformed = true
            invalidate()
        } else if (spline.controlPoints.size <= 3) {
            Toast.makeText(context, context.getString(R.string.spline_error), Toast.LENGTH_SHORT).show()
        }
    }

    fun clearCanvas() {
        spline.controlPoints.clear()

        canvas.drawColor(Color.WHITE)
        invalidate()

        transformed = false

        SelectedPoint.show = false
    }

    fun setAddPointState() {
        state = State.ADD_POINT
        SelectedPoint.show = false
    }

    fun setMovePointState() {
        state = State.MOVE_POINT
        SelectedPoint.show = false
    }

    fun setDeletePointState() {
        state = State.DELETE_POINT
        SelectedPoint.show = false
    }
}