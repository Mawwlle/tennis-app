package com.example.ballrealtime

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.hypot

class CalibrationOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {
    enum class Mode { TABLE, PEOPLE, HIDDEN }

    var mode: Mode = Mode.HIDDEN
        set(value) {
            field = value
            visibility = if (value == Mode.HIDDEN) GONE else VISIBLE
            isClickable = value != Mode.HIDDEN
            invalidate()
        }
    var onPeopleChanged: (() -> Unit)? = null

    private val tablePoints = mutableListOf<PointF>()
    private val peoplePoints = mutableListOf<PointF>()
    private var draggingCorner = -1

    private val fillPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.argb(65, 255, 128, 0)
    }
    private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        color = Color.rgb(255, 150, 30)
    }
    private val cornerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.rgb(255, 220, 80)
    }
    private val personPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.FILL
        color = Color.rgb(0, 230, 160)
    }
    private val personStrokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 4f
        color = Color.BLACK
    }

    fun tablePolygon(): List<PointF> = tablePoints.map { PointF(it.x, it.y) }
    fun peoplePositions(): List<PointF> = peoplePoints.map { PointF(it.x, it.y) }

    fun resetPeople() {
        peoplePoints.clear()
        invalidate()
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        if (tablePoints.isEmpty() && w > 0 && h > 0) {
            tablePoints.add(PointF(w * 0.28f, h * 0.38f))
            tablePoints.add(PointF(w * 0.72f, h * 0.38f))
            tablePoints.add(PointF(w * 0.86f, h * 0.72f))
            tablePoints.add(PointF(w * 0.14f, h * 0.72f))
        }
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (mode == Mode.HIDDEN) return
        drawTable(canvas)
        drawPeople(canvas)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (mode == Mode.HIDDEN) return false
        return when (mode) {
            Mode.TABLE -> handleTableTouch(event)
            Mode.PEOPLE -> handlePeopleTouch(event)
            Mode.HIDDEN -> false
        }
    }

    private fun handleTableTouch(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                draggingCorner = nearestCorner(event.x, event.y)
                return draggingCorner >= 0
            }
            MotionEvent.ACTION_MOVE -> {
                if (draggingCorner >= 0) {
                    tablePoints[draggingCorner].x = event.x.coerceIn(0f, width.toFloat())
                    tablePoints[draggingCorner].y = event.y.coerceIn(0f, height.toFloat())
                    invalidate()
                }
                return true
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                draggingCorner = -1
                return true
            }
        }
        return true
    }

    private fun handlePeopleTouch(event: MotionEvent): Boolean {
        if (event.actionMasked != MotionEvent.ACTION_UP) return true
        if (peoplePoints.size >= 2) {
            peoplePoints.clear()
        }
        peoplePoints.add(PointF(event.x, event.y))
        invalidate()
        onPeopleChanged?.invoke()
        return true
    }

    private fun nearestCorner(x: Float, y: Float): Int {
        var bestIndex = -1
        var bestDistance = 64f
        for (i in tablePoints.indices) {
            val p = tablePoints[i]
            val d = hypot((p.x - x).toDouble(), (p.y - y).toDouble()).toFloat()
            if (d < bestDistance) {
                bestDistance = d
                bestIndex = i
            }
        }
        return bestIndex
    }

    private fun drawTable(canvas: Canvas) {
        if (tablePoints.size < 4) return
        val path = Path()
        path.moveTo(tablePoints[0].x, tablePoints[0].y)
        for (i in 1 until tablePoints.size) {
            path.lineTo(tablePoints[i].x, tablePoints[i].y)
        }
        path.close()
        canvas.drawPath(path, fillPaint)
        canvas.drawPath(path, strokePaint)
        for (point in tablePoints) {
            canvas.drawCircle(point.x, point.y, 18f, cornerPaint)
            canvas.drawCircle(point.x, point.y, 18f, strokePaint)
        }
    }

    private fun drawPeople(canvas: Canvas) {
        for ((index, point) in peoplePoints.withIndex()) {
            canvas.drawCircle(point.x, point.y, 20f, personPaint)
            canvas.drawCircle(point.x, point.y, 20f, personStrokePaint)
            val label = if (index == 0) "L" else "R"
            val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                color = Color.BLACK
                textSize = 24f
                textAlign = Paint.Align.CENTER
                isFakeBoldText = true
            }
            canvas.drawText(label, point.x, point.y + 8f, textPaint)
        }
    }
}
