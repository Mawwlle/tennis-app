package com.example.ballrealtime

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sign

class GameState {
    enum class Side { LEFT, RIGHT }
    enum class EventKind { HIT, BOUNCE, NET, MISS }

    data class Detection(
        val frame: Int,
        val cx: Float,
        val cy: Float,
        val confidence: Float
    )

    data class Event(
        val frame: Int,
        val kind: EventKind,
        val side: Side,
        val cx: Float,
        val cy: Float,
        val confidence: Float
    )

    data class Score(
        val left: Int = 0,
        val right: Int = 0,
        val pointWinner: Side? = null,
        val eventLabel: String? = null
    )

    data class Snapshot(
        val score: Score,
        val currentEvent: Event?,
        val recentEvents: List<Event>,
        val netX: Float
    )

    private enum class Phase { IDLE, AWAIT_BOUNCE, AWAIT_RETURN }

    private var phase = Phase.IDLE
    private var striker: Side? = null
    private var targetSide: Side? = null
    private var returnerSide: Side? = null
    private var strikerSide: Side? = null
    private var lastStateEventFrame = -1

    private var score = Score()
    private val detections = ArrayDeque<Detection>()
    private val events = ArrayDeque<Event>()
    private var lastHit: Event? = null
    private var lastBounce: Event? = null
    private var lastEventLabelFrame = -EVENT_TEXT_FADE_FRAMES * 2
    private var lastEventLabel: String? = null
    private var missingFrames = 0
    private var frameWidth = 640f
    private var frameHeight = 640f
    private var netCxRatio = NET_CX_RATIO
    private var tableLeftRatio = 0.05f
    private var tableTopRatio = 0.38f
    private var tableRightRatio = 0.95f
    private var tableBottomRatio = 0.88f

    fun updateFrameSize(width: Int, height: Int) {
        if (width > 0) frameWidth = width.toFloat()
        if (height > 0) frameHeight = height.toFloat()
    }

    fun updateNetXRatio(ratio: Float) {
        netCxRatio = ratio.coerceIn(0.15f, 0.85f)
    }

    fun updateTableBounds(left: Float, top: Float, right: Float, bottom: Float) {
        tableLeftRatio = left.coerceIn(0f, 0.9f)
        tableTopRatio = top.coerceIn(0f, 0.9f)
        tableRightRatio = right.coerceIn(tableLeftRatio + 0.05f, 1f)
        tableBottomRatio = bottom.coerceIn(tableTopRatio + 0.05f, 1f)
        updateNetXRatio((tableLeftRatio + tableRightRatio) * 0.5f)
    }

    fun addDetection(detection: Detection): Snapshot {
        detections.addLast(detection)
        missingFrames = 0
        while (detections.size > MAX_HISTORY) {
            detections.removeFirst()
        }

        val event = detectNewEvent()
        if (event != null) {
            events.addLast(event)
            while (events.size > MAX_EVENTS) events.removeFirst()
            applyEvent(event)
            lastEventLabel = "${event.kind.name.lowercase()} ${event.side.name.lowercase()}"
            lastEventLabelFrame = event.frame
        } else {
            applyTimeout(detection.frame)
        }

        return snapshot(detection.frame, event)
    }

    fun addMissingFrame(frame: Int): Snapshot {
        missingFrames += 1
        if (missingFrames >= BALL_LOST_FRAMES) {
            val winner = feedBallLost()
            missingFrames = 0
            if (winner != null) addPoint(winner, frame, "point ${winner.name.lowercase()}")
        } else {
            applyTimeout(frame)
        }
        return snapshot(frame, null)
    }

    fun reset() {
        phase = Phase.IDLE
        striker = null
        targetSide = null
        returnerSide = null
        strikerSide = null
        lastStateEventFrame = -1
        score = Score()
        detections.clear()
        events.clear()
        lastHit = null
        lastBounce = null
        lastEventLabel = null
        lastEventLabelFrame = -EVENT_TEXT_FADE_FRAMES * 2
        missingFrames = 0
        netCxRatio = NET_CX_RATIO
        tableLeftRatio = 0.05f
        tableTopRatio = 0.38f
        tableRightRatio = 0.95f
        tableBottomRatio = 0.88f
    }

    private fun snapshot(frame: Int, event: Event?): Snapshot {
        val label = if (frame - lastEventLabelFrame <= EVENT_TEXT_FADE_FRAMES) lastEventLabel else null
        val pointWinner = if (label?.startsWith("point") == true) score.pointWinner else null
        val currentScore = score.copy(pointWinner = pointWinner, eventLabel = label)
        return Snapshot(
            score = currentScore,
            currentEvent = event,
            recentEvents = events.filter { frame - it.frame <= EVENT_MARKER_FRAMES },
            netX = netX()
        )
    }

    private fun detectNewEvent(): Event? {
        if (detections.size < LOCAL_WINDOW * 2 + 3) return null

        val list = detections.toList()
        val i = list.size - LOCAL_WINDOW - 1
        if (i <= LOCAL_WINDOW) return null

        val current = list[i]
        val prev = list[i - 1]
        val next = list[i + 1]
        val before = list.subList(i - LOCAL_WINDOW, i)
        val after = list.subList(i + 1, i + 1 + LOCAL_WINDOW)
        val dxBefore = median(before.zip(before.drop(1)).map { it.second.cx - it.first.cx })
        val dxAfter = median(after.zip(after.drop(1)).map { it.second.cx - it.first.cx })
        val dyBefore = median(before.zip(before.drop(1)).map { it.second.cy - it.first.cy })
        val dyAfter = median(after.zip(after.drop(1)).map { it.second.cy - it.first.cy })
        val travelBefore = abs(current.cx - list[i - LOCAL_WINDOW].cx)
        val travelAfter = abs(list[i + LOCAL_WINDOW].cx - current.cx)
        detectBounce(current, dyBefore, dyAfter)?.let { return it }
        detectHit(current, dxBefore, dxAfter, travelBefore, travelAfter)?.let { return it }
        detectNet(current, prev, next)?.let { return it }
        detectMiss(current)?.let { return it }

        return null
    }

    private fun detectBounce(current: Detection, dyBefore: Float, dyAfter: Float): Event? {
        val curvature = dyBefore - dyAfter
        val recentBounce = lastBounce
        if (dyBefore <= BOUNCE_MIN_DY || dyAfter >= -BOUNCE_MIN_DY) return null
        if (curvature < BOUNCE_MIN_CURVATURE) return null
        if (!isOnTable(current.cx, current.cy, marginRatio = 0.035f)) return null
        if (recentBounce != null && current.frame - recentBounce.frame < BOUNCE_DEDUP_FRAMES) return null

        return Event(current.frame, EventKind.BOUNCE, sideFor(current.cx), current.cx, current.cy, curvature).also {
            lastBounce = it
        }
    }

    private fun detectHit(
        current: Detection,
        dxBefore: Float,
        dxAfter: Float,
        travelBefore: Float,
        travelAfter: Float
    ): Event? {
        val netX = netX()
        val netMargin = HIT_NET_MARGIN_RATIO * frameWidth
        val side = when {
            dxBefore < -HIT_MIN_X_SPEED && dxAfter > HIT_MIN_X_SPEED -> {
                if (current.cx >= netX - netMargin) return null
                Side.LEFT
            }
            dxBefore > HIT_MIN_X_SPEED && dxAfter < -HIT_MIN_X_SPEED -> {
                if (current.cx <= netX + netMargin) return null
                Side.RIGHT
            }
            else -> return null
        }
        val recentHit = lastHit
        if (min(travelBefore, travelAfter) < HIT_MIN_TRAVEL_PX) return null
        if (current.cy > tableBottom() + 0.08f * frameHeight) return null
        if (recentHit != null && current.frame - recentHit.frame < HIT_DEDUP_FRAMES) return null

        return Event(current.frame, EventKind.HIT, side, current.cx, current.cy, travelBefore + travelAfter).also {
            lastHit = it
        }
    }

    private fun detectNet(current: Detection, prev: Detection, next: Detection): Event? {
        val hit = lastHit ?: return null
        if (current.frame - hit.frame !in 2..NET_POST_HIT_FRAMES) return null
        if (lastBounce != null && lastBounce!!.frame > hit.frame) return null
        if (abs(current.cx - netX()) > NET_NEAR_RATIO * frameWidth) return null
        if (current.cy < tableTop() - 0.20f * tableHeight() || current.cy > tableBottom()) return null
        val dx0 = current.cx - prev.cx
        val dx1 = next.cx - current.cx
        if (abs(dx0) < NET_REVERSAL_X_SPEED || abs(dx1) < NET_REVERSAL_X_SPEED) return null
        if (sign(dx0) == sign(dx1)) return null
        if (events.lastOrNull { it.kind == EventKind.NET }?.let { current.frame - it.frame < NET_DEDUP_FRAMES } == true) return null
        return Event(current.frame, EventKind.NET, hit.side, current.cx, current.cy, abs(dx0 - dx1))
    }

    private fun detectMiss(current: Detection): Event? {
        val hit = lastHit ?: return null
        if (current.frame - hit.frame !in 8..MAX_HIT_TO_BOUNCE_FRAMES) return null
        val targetSide = otherSide(hit.side)
        if (sideFor(current.cx) != targetSide) return null
        if (lastBounce != null && lastBounce!!.frame > hit.frame && lastBounce!!.side == targetSide) return null
        val outMargin = MISS_OUT_X_MARGIN_RATIO * tableWidth()
        val low = current.cy >= tableBottom() + outMargin
        val out = current.cx <= tableLeft() - outMargin || current.cx >= tableRight() + outMargin
        if (!low && !out) return null
        if (events.lastOrNull { it.kind == EventKind.MISS }?.let { current.frame - it.frame < HIT_DEDUP_FRAMES } == true) return null
        return Event(current.frame, EventKind.MISS, hit.side, current.cx, current.cy, 1f)
    }

    private fun applyEvent(event: Event) {
        val winner = when (event.kind) {
            EventKind.HIT -> feedHit(event)
            EventKind.BOUNCE -> feedBounce(event)
            EventKind.NET -> feedNet(event)
            EventKind.MISS -> feedMiss(event)
        }
        if (winner != null) addPoint(winner, event.frame, "point ${winner.name.lowercase()}")
    }

    private fun applyTimeout(frame: Int) {
        val winner = feedTimeout(frame)
        if (winner != null) addPoint(winner, frame, "point ${winner.name.lowercase()}")
    }

    private fun addPoint(winner: Side, frame: Int, label: String) {
        score = Score(
            left = score.left + if (winner == Side.LEFT) 1 else 0,
            right = score.right + if (winner == Side.RIGHT) 1 else 0,
            pointWinner = winner,
            eventLabel = label
        )
        lastEventLabel = label
        lastEventLabelFrame = frame
    }

    private fun feedHit(hit: Event): Side? {
        if (phase == Phase.IDLE) {
            phase = Phase.AWAIT_BOUNCE
            striker = hit.side
            targetSide = otherSide(hit.side)
            lastStateEventFrame = hit.frame
            return null
        }

        if (phase == Phase.AWAIT_RETURN && hit.side == returnerSide) {
            phase = Phase.AWAIT_BOUNCE
            striker = hit.side
            targetSide = otherSide(hit.side)
            lastStateEventFrame = hit.frame
            return null
        }

        if (phase == Phase.AWAIT_BOUNCE && hit.side == striker && hit.frame - lastStateEventFrame <= HIT_DEDUP_FRAMES) {
            lastStateEventFrame = hit.frame
            return null
        }

        val winner = otherSide(hit.side)
        resetRally()
        return winner
    }

    private fun feedBounce(bounce: Event): Side? {
        if (phase == Phase.IDLE) return null

        if (phase == Phase.AWAIT_BOUNCE) {
            if (bounce.side == targetSide) {
                phase = Phase.AWAIT_RETURN
                returnerSide = bounce.side
                strikerSide = striker
                lastStateEventFrame = bounce.frame
                return null
            }
            val winner = otherSide(striker ?: bounce.side)
            resetRally()
            return winner
        }

        if (phase == Phase.AWAIT_RETURN) {
            val winner = if (bounce.side == returnerSide) strikerSide else returnerSide
            resetRally()
            return winner
        }

        return null
    }

    private fun feedNet(net: Event): Side {
        resetRally()
        return otherSide(net.side)
    }

    private fun feedMiss(miss: Event): Side {
        resetRally()
        return otherSide(miss.side)
    }

    private fun feedBallLost(): Side? {
        if (phase == Phase.IDLE) return null
        val winner = if (phase == Phase.AWAIT_BOUNCE) {
            otherSide(striker ?: Side.LEFT)
        } else {
            strikerSide ?: Side.LEFT
        }
        resetRally()
        return winner
    }

    private fun feedTimeout(frame: Int): Side? {
        if (phase == Phase.IDLE) return null
        if (phase == Phase.AWAIT_BOUNCE && frame - lastStateEventFrame >= MAX_HIT_TO_BOUNCE_FRAMES) {
            val winner = otherSide(striker ?: Side.LEFT)
            resetRally()
            return winner
        }
        if (phase == Phase.AWAIT_RETURN && frame - lastStateEventFrame >= MAX_BOUNCE_TO_HIT_FRAMES) {
            val winner = strikerSide ?: Side.LEFT
            resetRally()
            return winner
        }
        return null
    }

    private fun resetRally() {
        phase = Phase.IDLE
        striker = null
        targetSide = null
        returnerSide = null
        strikerSide = null
        lastStateEventFrame = -1
    }

    private fun sideFor(cx: Float): Side = if (cx < netX()) Side.LEFT else Side.RIGHT
    private fun otherSide(side: Side): Side = if (side == Side.LEFT) Side.RIGHT else Side.LEFT
    private fun netX(): Float = netCxRatio * frameWidth
    private fun tableLeft(): Float = tableLeftRatio * frameWidth
    private fun tableRight(): Float = tableRightRatio * frameWidth
    private fun tableTop(): Float = tableTopRatio * frameHeight
    private fun tableBottom(): Float = tableBottomRatio * frameHeight
    private fun tableWidth(): Float = max(1f, tableRight() - tableLeft())
    private fun tableHeight(): Float = max(1f, tableBottom() - tableTop())

    private fun isOnTable(cx: Float, cy: Float, marginRatio: Float): Boolean {
        val xMargin = marginRatio * tableWidth()
        val yMargin = marginRatio * tableHeight()
        return cx >= tableLeft() - xMargin &&
                cx <= tableRight() + xMargin &&
                cy >= tableTop() - yMargin &&
                cy <= tableBottom() + yMargin
    }

    private fun median(values: List<Float>): Float {
        if (values.isEmpty()) return 0f
        val sorted = values.sorted()
        return sorted[sorted.size / 2]
    }

    companion object {
        private const val NET_CX_RATIO = 0.50f
        private const val LOCAL_WINDOW = 4
        private const val HIT_DEDUP_FRAMES = 10
        private const val BOUNCE_DEDUP_FRAMES = 12
        private const val NET_DEDUP_FRAMES = 12
        private const val HIT_MIN_X_SPEED = 1.4f
        private const val HIT_MIN_TRAVEL_PX = 18.0f
        private const val HIT_NET_MARGIN_RATIO = 0.10f
        private const val HIT_Y_MAX_RATIO = 0.92f
        private const val BOUNCE_MIN_Y_RATIO = 0.42f
        private const val BOUNCE_MAX_Y_RATIO = 0.90f
        private const val BOUNCE_MIN_DY = 1.0f
        private const val BOUNCE_MIN_CURVATURE = 1.8f
        private const val NET_NEAR_RATIO = 0.07f
        private const val NET_REVERSAL_X_SPEED = 0.90f
        private const val NET_MIN_Y_RATIO = 0.20f
        private const val NET_MAX_Y_RATIO = 0.78f
        private const val NET_POST_HIT_FRAMES = 90
        private const val MAX_HIT_TO_BOUNCE_FRAMES = 110
        private const val MAX_BOUNCE_TO_HIT_FRAMES = 140
        private const val BALL_LOST_FRAMES = 60
        private const val EVENT_TEXT_FADE_FRAMES = 26
        private const val EVENT_MARKER_FRAMES = 36
        private const val MISS_MAX_Y_RATIO = 0.88f
        private const val MISS_OUT_X_MARGIN_RATIO = 0.03f
        private const val MAX_HISTORY = 240
        private const val MAX_EVENTS = 32
    }
}
