package com.example.ballrealtime

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

class DetectionOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    data class DetectionItem(
        val rect: RectF,
        val label: String,
        val classIndex: Int
    )

    data class BallOverlay(
        val center: PointF,
        val detected: Boolean,
        val label: String
    )

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        color = Color.GREEN
        strokeWidth = 8f
        isAntiAlias = true
    }

    private val textBgPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.argb(190, 0, 0, 0)
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        style = Paint.Style.FILL
        color = Color.WHITE
        textSize = 42f
        isAntiAlias = true
    }

    private val trailPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        strokeCap = Paint.Cap.ROUND
        isAntiAlias = true
    }

    private val ballFillPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private val ballStrokePaint = Paint().apply {
        style = Paint.Style.STROKE
        color = Color.BLACK
        strokeWidth = 2f
        isAntiAlias = true
    }

    private var items: List<DetectionItem> = emptyList()
    private var ball: BallOverlay? = null
    private var trail: List<PointF> = emptyList()
    private var score: GameState.Score = GameState.Score()
    private var events: List<GameState.Event> = emptyList()
    private var netX: Float? = null

    fun showDetections(newItems: List<DetectionItem>) {
        items = newItems.map { it.copy(rect = RectF(it.rect)) }
        invalidate()
    }

    fun showTrack(ball: BallOverlay, trail: List<PointF>) {
        this.ball = ball.copy(center = PointF(ball.center.x, ball.center.y))
        this.trail = trail.map { PointF(it.x, it.y) }
        invalidate()
    }

    fun showGame(
        ball: BallOverlay?,
        trail: List<PointF>,
        score: GameState.Score,
        events: List<GameState.Event>,
        netX: Float?
    ) {
        this.ball = ball?.copy(center = PointF(ball.center.x, ball.center.y))
        this.trail = trail.map { PointF(it.x, it.y) }
        this.score = score
        this.events = events
        this.netX = netX
        invalidate()
    }

    fun clearDetection() {
        items = emptyList()
        ball = null
        trail = emptyList()
        score = GameState.Score()
        events = emptyList()
        netX = null
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        drawNet(canvas)
        drawScore(canvas)

        if (trail.size >= 2) {
            for (i in 1 until trail.size) {
                val alpha = (i.toFloat() / trail.size.toFloat()).coerceIn(0.1f, 1f)
                trailPaint.color = Color.argb((alpha * 255).toInt(), 255, 220, 0)
                val p0 = trail[i - 1]
                val p1 = trail[i]
                canvas.drawLine(p0.x, p0.y, p1.x, p1.y, trailPaint)
            }
        }

        drawEvents(canvas)

        ball?.let { current ->
            val radius = if (current.detected) 10f else 8f
            ballFillPaint.color = if (current.detected) {
                Color.rgb(0, 220, 255)
            } else {
                Color.rgb(180, 180, 255)
            }
            canvas.drawCircle(current.center.x, current.center.y, radius, ballFillPaint)
            canvas.drawCircle(current.center.x, current.center.y, radius, ballStrokePaint)

            if (current.label.isNotEmpty()) {
                val textPadding = 12f
                val textHeight = textPaint.textSize + textPadding * 2
                val textWidth = textPaint.measureText(current.label) + textPadding * 2
                val top = (current.center.y - radius - textHeight).coerceAtLeast(0f)
                val left = (current.center.x - textWidth / 2f).coerceAtLeast(0f)
                val bgRect = RectF(
                    left,
                    top,
                    (left + textWidth).coerceAtMost(width.toFloat()),
                    (top + textHeight).coerceAtMost(height.toFloat())
                )
                canvas.drawRect(bgRect, textBgPaint)
                canvas.drawText(
                    current.label,
                    bgRect.left + textPadding,
                    bgRect.bottom - textPadding,
                    textPaint
                )
            }
        }

        if (items.isEmpty()) return

        for (item in items) {
            val r = item.rect
            boxPaint.color = colorForClass(item.classIndex)
            canvas.drawRect(r, boxPaint)
            if (item.label.isNotEmpty()) {
                val textPadding = 12f
                val textHeight = textPaint.textSize + textPadding * 2
                val textWidth = textPaint.measureText(item.label) + textPadding * 2
                val top = (r.top - textHeight).coerceAtLeast(0f)
                val left = r.left.coerceAtLeast(0f)
                val bgRect = RectF(
                    left,
                    top,
                    (left + textWidth).coerceAtMost(width.toFloat()),
                    (top + textHeight).coerceAtMost(height.toFloat())
                )
                canvas.drawRect(bgRect, textBgPaint)
                canvas.drawText(
                    item.label,
                    bgRect.left + textPadding,
                    bgRect.bottom - textPadding,
                    textPaint
                )
            }
        }
    }

    private fun drawScore(canvas: Canvas) {
        val barHeight = 70f
        val bg = if (score.pointWinner != null) Color.rgb(50, 170, 90) else Color.argb(210, 20, 24, 30)
        textBgPaint.color = bg
        canvas.drawRect(0f, 0f, width.toFloat(), barHeight, textBgPaint)

        textPaint.textSize = 34f
        textPaint.color = Color.WHITE
        val scoreText = "${score.left}  :  ${score.right}"
        val scoreX = (width - textPaint.measureText(scoreText)) / 2f
        canvas.drawText(scoreText, scoreX, 46f, textPaint)

        textPaint.textSize = 22f
        textPaint.color = Color.rgb(255, 190, 70)
        canvas.drawText("LEFT", 18f, 45f, textPaint)

        val right = "RIGHT"
        textPaint.color = Color.rgb(70, 220, 255)
        canvas.drawText(right, width - textPaint.measureText(right) - 18f, 45f, textPaint)

        score.eventLabel?.let { label ->
            textPaint.textSize = 28f
            textPaint.color = Color.WHITE
            canvas.drawText(label.uppercase(), 18f, height - 26f, textPaint)
        }
    }

    private fun drawNet(canvas: Canvas) {
        val net = netX ?: return
        if (width == 0) return
        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = 3f
            color = Color.argb(140, 180, 180, 255)
        }
        canvas.drawLine(net.coerceIn(0f, width.toFloat()), 0f, net.coerceIn(0f, width.toFloat()), height.toFloat(), paint)
    }

    private fun drawEvents(canvas: Canvas) {
        for (event in events) {
            val eventColor = when (event.kind) {
                GameState.EventKind.HIT -> Color.rgb(80, 255, 180)
                GameState.EventKind.BOUNCE -> Color.rgb(255, 220, 0)
                GameState.EventKind.NET -> Color.rgb(0, 80, 230)
                GameState.EventKind.MISS -> Color.rgb(40, 80, 255)
            }
            ballFillPaint.color = eventColor
            ballStrokePaint.color = Color.BLACK
            val x = event.cx
            val y = event.cy
            when (event.kind) {
                GameState.EventKind.HIT -> {
                    ballStrokePaint.color = eventColor
                    ballStrokePaint.strokeWidth = 5f
                    canvas.drawCircle(x, y, 18f, ballStrokePaint)
                    ballStrokePaint.color = Color.BLACK
                    ballStrokePaint.strokeWidth = 2f
                }
                GameState.EventKind.BOUNCE -> {
                    canvas.drawCircle(x, y, 13f, ballFillPaint)
                    canvas.drawCircle(x, y, 13f, ballStrokePaint)
                }
                GameState.EventKind.NET -> {
                    val r = 16f
                    val p = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                        style = Paint.Style.STROKE
                        strokeWidth = 5f
                        color = eventColor
                    }
                    canvas.drawLine(x - r, y - r, x + r, y + r, p)
                    canvas.drawLine(x - r, y + r, x + r, y - r, p)
                }
                GameState.EventKind.MISS -> {
                    val p = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                        style = Paint.Style.STROKE
                        strokeWidth = 5f
                        color = eventColor
                    }
                    canvas.drawCircle(x, y, 18f, p)
                    textPaint.textSize = 22f
                    textPaint.color = eventColor
                    canvas.drawText("OUT", x + 12f, y - 10f, textPaint)
                }
            }
        }
    }

    private fun colorForClass(classIndex: Int): Int {
        val palette = intArrayOf(
            Color.GREEN,
            Color.CYAN,
            Color.YELLOW,
            Color.MAGENTA,
            Color.RED,
            Color.BLUE
        )
        return palette[(classIndex and Int.MAX_VALUE) % palette.size]
    }

}
