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
        val eventLabel: String? = null,
        val server: Side = Side.LEFT
    )

    data class Snapshot(
        val score: Score,
        val currentEvent: Event?,
        val recentEvents: List<Event>,
        val netX: Float
    )

    private data class Trajectory(
        val frames: List<Int>,
        val x: List<Float>,
        val y: List<Float>,
        val xSmooth: List<Float>,
        val ySmooth: List<Float>,
        val dx: List<Float>,
        val dy: List<Float>
    )

    private enum class Phase { IDLE, AWAIT_SERVE_BOUNCE, AWAIT_BOUNCE, AWAIT_RETURN }

    private var phase = Phase.IDLE
    private var striker: Side? = null
    private var targetSide: Side? = null
    private var returnerSide: Side? = null
    private var strikerSide: Side? = null
    private var lastStateEventFrame = -1

    private var score = Score()
    private val detections = ArrayDeque<Detection>()
    private val events = ArrayDeque<Event>()
    private val emittedEventKeys = mutableSetOf<String>()
    private var lastHit: Event? = null
    private var lastBounce: Event? = null
    private var lastAppliedEventFrame = -1
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

    fun addDetection(detection: Detection, eventNetBounce: Event? = null): Snapshot {
        detections.addLast(detection)
        missingFrames = 0
        while (detections.size > MAX_HISTORY) {
            detections.removeFirst()
        }

        val detectedEvents = detectNewEvents(eventNetBounce)
        if (detectedEvents.isNotEmpty()) {
            for (event in detectedEvents) {
                emittedEventKeys.add(eventKey(event))
                lastAppliedEventFrame = max(lastAppliedEventFrame, event.frame)
                events.addLast(event)
                while (events.size > MAX_EVENTS) events.removeFirst()
                applyEvent(event)
                lastEventLabel = "${event.kind.name.lowercase()} ${event.side.name.lowercase()}"
                lastEventLabelFrame = event.frame
            }
        } else {
            applyTimeout(detection.frame)
        }

        return snapshot(detection.frame, detectedEvents.lastOrNull())
    }

    fun currentNetX(): Float = netX()

    fun addMissingFrame(frame: Int, eventNetBounce: Event? = null): Snapshot {
        missingFrames += 1
        val detectedEvents = if (eventNetBounce != null) detectNewEvents(eventNetBounce) else emptyList()
        if (detectedEvents.isNotEmpty()) {
            for (event in detectedEvents) {
                emittedEventKeys.add(eventKey(event))
                lastAppliedEventFrame = max(lastAppliedEventFrame, event.frame)
                events.addLast(event)
                while (events.size > MAX_EVENTS) events.removeFirst()
                applyEvent(event)
                lastEventLabel = "${event.kind.name.lowercase()} ${event.side.name.lowercase()}"
                lastEventLabelFrame = event.frame
            }
            return snapshot(frame, detectedEvents.lastOrNull())
        }
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
        emittedEventKeys.clear()
        lastHit = null
        lastBounce = null
        lastAppliedEventFrame = -1
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

    private fun detectNewEvents(eventNetBounce: Event?): List<Event> {
        val trajectory = buildTrajectory(detections.toList()) ?: return emptyList()
        val stableFrame = detections.lastOrNull()?.frame?.minus(LOCAL_WINDOW) ?: return emptyList()
        val trajectoryBounces = detectTrajectoryBounces(trajectory)
        val bounces = if (eventNetBounce != null) {
            knownBouncesWith(eventNetBounce)
        } else {
            trajectoryBounces
        }
        var hits = filterHitsNearBounces(detectHitsFromBounces(bounces, trajectory), bounces)
        if (hits.isEmpty()) {
            hits = filterHitsNearBounces(detectTrajectoryHits(trajectory), bounces)
        }
        val bounceEvents = if (eventNetBounce != null) listOf(eventNetBounce) else trajectoryBounces
        val merged = mergeEvents(
            hits = hits,
            bounces = bounceEvents,
            nets = detectTrajectoryNets(trajectory, hits, bounces),
            misses = detectTrajectoryMisses(trajectory, hits, bounces)
        )
        return merged.filter { event ->
            val eventStableFrame = if (eventNetBounce != null && event.frame <= eventNetBounce.frame) {
                max(stableFrame, eventNetBounce.frame)
            } else {
                stableFrame
            }
            event.frame <= eventStableFrame &&
                    event.frame >= lastAppliedEventFrame &&
                    !emittedEventKeys.contains(eventKey(event))
        }
    }

    private fun buildTrajectory(points: List<Detection>): Trajectory? {
        if (points.size < LOCAL_WINDOW * 2 + 3) return null
        val frames = points.map { it.frame }
        val x = points.map { it.cx }
        val y = points.map { it.cy }
        val xSmooth = smoothSignal(x)
        val ySmooth = smoothSignal(y)
        return Trajectory(
            frames = frames,
            x = x,
            y = y,
            xSmooth = xSmooth,
            ySmooth = ySmooth,
            dx = gradient(xSmooth),
            dy = gradient(ySmooth)
        )
    }

    private fun detectTrajectoryBounces(traj: Trajectory): List<Event> {
        val bounces = ArrayList<Event>()
        var lastFrame = -BOUNCE_DEDUP_FRAMES * 2
        for (i in LOCAL_WINDOW until traj.frames.size - LOCAL_WINDOW) {
            val leftDy = median(traj.dy.subList(i - LOCAL_WINDOW, i))
            val rightDy = median(traj.dy.subList(i + 1, i + 1 + LOCAL_WINDOW))
            val curvature = leftDy - rightDy
            val xSpeed = abs(traj.dx[i])
            val cx = traj.xSmooth[i]
            val cy = traj.ySmooth[i]
            if (leftDy <= BOUNCE_MIN_DY || rightDy >= -BOUNCE_MIN_DY) continue
            if (curvature < BOUNCE_MIN_CURVATURE) continue
            if (xSpeed < BOUNCE_MIN_X_SPEED) continue
            if (cy < BOUNCE_MIN_Y_RATIO * frameHeight) continue
            if (cy > BOUNCE_MAX_Y_RATIO * frameHeight) continue
            if (cx <= 0.05f * frameWidth || cx >= 0.95f * frameWidth) continue

            val frame = traj.frames[i]
            val event = Event(frame, EventKind.BOUNCE, sideFor(cx), traj.x[i], traj.y[i], curvature)
            if (frame - lastFrame < BOUNCE_DEDUP_FRAMES) {
                if (bounces.isNotEmpty() && curvature > bounces.last().confidence) {
                    bounces[bounces.lastIndex] = event
                    lastFrame = frame
                }
                continue
            }
            bounces.add(event)
            lastFrame = frame
        }
        return bounces
    }

    private fun detectHitsFromBounces(bounces: List<Event>, traj: Trajectory): List<Event> {
        if (bounces.size < 2) return emptyList()
        val hits = ArrayList<Event>()
        val sortedBounces = bounces.sortedBy { it.frame }
        for (i in 0 until sortedBounces.size - 1) {
            val b0 = sortedBounces[i]
            val b1 = sortedBounces[i + 1]
            if (b0.side == b1.side) continue
            val start = searchSorted(traj.frames, b0.frame + 1)
            val stop = searchSorted(traj.frames, b1.frame)
            if (stop - start < 3) continue

            val hitterSide = b0.side
            var hitIdx = start
            for (j in start + 1 until stop) {
                val better = if (hitterSide == Side.RIGHT) traj.x[j] > traj.x[hitIdx] else traj.x[j] < traj.x[hitIdx]
                if (better) hitIdx = j
            }
            if (hitterSide == Side.RIGHT && traj.x[hitIdx] <= netX()) continue
            if (hitterSide == Side.LEFT && traj.x[hitIdx] >= netX()) continue

            hits.add(Event(
                frame = traj.frames[hitIdx],
                kind = EventKind.HIT,
                side = hitterSide,
                cx = traj.x[hitIdx],
                cy = traj.y[hitIdx],
                confidence = abs(traj.x[hitIdx] - netX())
            ))
        }
        return hits
    }

    private fun detectTrajectoryHits(traj: Trajectory): List<Event> {
        val hits = ArrayList<Event>()
        val netMargin = HIT_NET_MARGIN_RATIO * frameWidth
        var lastFrame = -HIT_DEDUP_FRAMES * 2
        for (i in LOCAL_WINDOW until traj.frames.size - LOCAL_WINDOW) {
            val leftDx = median(traj.dx.subList(i - LOCAL_WINDOW, i))
            val rightDx = median(traj.dx.subList(i + 1, i + 1 + LOCAL_WINDOW))
            val travelBefore = abs(traj.xSmooth[i] - traj.xSmooth[i - LOCAL_WINDOW])
            val travelAfter = abs(traj.xSmooth[i + LOCAL_WINDOW] - traj.xSmooth[i])
            val cx = traj.xSmooth[i]
            val cy = traj.ySmooth[i]
            val side = when {
                leftDx < -HIT_MIN_X_SPEED && rightDx > HIT_MIN_X_SPEED -> {
                    if (cx >= netX() - netMargin) continue
                    Side.LEFT
                }
                leftDx > HIT_MIN_X_SPEED && rightDx < -HIT_MIN_X_SPEED -> {
                    if (cx <= netX() + netMargin) continue
                    Side.RIGHT
                }
                else -> continue
            }
            if (min(travelBefore, travelAfter) < HIT_MIN_TRAVEL_PX) continue
            if (cy > HIT_Y_MAX_RATIO * frameHeight) continue

            val frame = traj.frames[i]
            val confidence = travelBefore + travelAfter
            val event = Event(frame, EventKind.HIT, side, traj.x[i], traj.y[i], confidence)
            if (frame - lastFrame < HIT_DEDUP_FRAMES) {
                if (hits.isNotEmpty() && confidence > hits.last().confidence) {
                    hits[hits.lastIndex] = event
                    lastFrame = frame
                }
                continue
            }
            hits.add(event)
            lastFrame = frame
        }
        return hits
    }

    private fun filterHitsNearBounces(hits: List<Event>, bounces: List<Event>): List<Event> {
        return hits.filter { hit -> bounces.none { bounce -> abs(hit.frame - bounce.frame) <= HIT_BOUNCE_FILTER_FRAMES } }
    }

    private fun detectTrajectoryNets(traj: Trajectory, hits: List<Event>, bounces: List<Event>): List<Event> {
        if (hits.isEmpty()) return emptyList()
        val nets = ArrayList<Event>()
        val netNearPx = NET_NEAR_RATIO * frameWidth
        val crossMargin = NET_MAX_CROSS_RATIO * frameWidth
        var lastFrame = -NET_DEDUP_FRAMES * 2
        val sortedHits = hits.sortedBy { it.frame }
        for (idx in sortedHits.indices) {
            val hit = sortedHits[idx]
            val nextHitFrame = sortedHits.getOrNull(idx + 1)?.frame ?: ((traj.frames.lastOrNull() ?: hit.frame) + 1)
            val segmentBounces = bounces.filter { it.frame > hit.frame && it.frame < nextHitFrame }
            if (segmentBounces.any { it.side == otherSide(hit.side) }) continue

            val start = searchSorted(traj.frames, hit.frame)
            val stop = searchSorted(traj.frames, nextHitFrame)
            val segLimit = min(stop - start, NET_POST_HIT_FRAMES)
            if (segLimit < 3) continue

            var nearIdx = 0
            var nearDist = Float.MAX_VALUE
            for (j in 0 until segLimit) {
                val dist = abs(traj.xSmooth[start + j] - netX())
                if (dist < nearDist) {
                    nearDist = dist
                    nearIdx = j
                }
            }
            val nearFrame = traj.frames[start + nearIdx]
            val nearX = traj.xSmooth[start + nearIdx]
            val nearY = traj.ySmooth[start + nearIdx]
            val hitToNetProgress = abs(netX() - hit.cx)
            val progress = abs(nearX - hit.cx)
            val crossed = if (hit.side == Side.LEFT) {
                (0 until segLimit).any { traj.xSmooth[start + it] >= netX() + crossMargin }
            } else {
                (0 until segLimit).any { traj.xSmooth[start + it] <= netX() - crossMargin }
            }
            val reversalNearNet = if (nearIdx in 1 until segLimit - 1) {
                val before = median((max(0, nearIdx - 2)..nearIdx).map { traj.dx[start + it] })
                val after = median((nearIdx until min(segLimit, nearIdx + 3)).map { traj.dx[start + it] })
                abs(before) >= NET_REVERSAL_X_SPEED && abs(after) >= NET_REVERSAL_X_SPEED && sign(before) != sign(after)
            } else {
                false
            }
            val sameSideBounceNearNet = segmentBounces.any {
                it.side == hit.side && abs(it.cx - netX()) <= 1.5f * netNearPx
            }
            val yOk = nearY >= NET_MIN_Y_RATIO * frameHeight && nearY <= NET_MAX_Y_RATIO * frameHeight
            val progressOk = progress >= NET_MIN_PROGRESS_RATIO * max(hitToNetProgress, 1f)
            val stagnatedBeforeCross = !crossed && progressOk
            val candidateNearNet = nearDist <= netNearPx && yOk

            if (crossed && !reversalNearNet && !sameSideBounceNearNet) continue
            if (!candidateNearNet && !sameSideBounceNearNet) continue
            if (!reversalNearNet && !sameSideBounceNearNet && !stagnatedBeforeCross) continue
            if (segmentBounces.any { abs(nearFrame - it.frame) <= 3 }) continue
            if (nearFrame - lastFrame < NET_DEDUP_FRAMES) continue

            nets.add(Event(
                frame = nearFrame,
                kind = EventKind.NET,
                side = hit.side,
                cx = traj.x[start + nearIdx],
                cy = traj.y[start + nearIdx],
                confidence = max(1f, netNearPx - nearDist)
            ))
            lastFrame = nearFrame
        }
        return nets
    }

    private fun detectTrajectoryMisses(traj: Trajectory, hits: List<Event>, bounces: List<Event>): List<Event> {
        if (hits.isEmpty()) return emptyList()
        val misses = ArrayList<Event>()
        val crossMargin = NET_MAX_CROSS_RATIO * frameWidth
        val outMargin = MISS_OUT_X_MARGIN_RATIO * frameWidth
        val sortedHits = hits.sortedBy { it.frame }
        for (idx in sortedHits.indices) {
            val hit = sortedHits[idx]
            val targetSide = otherSide(hit.side)
            val nextHitFrame = sortedHits.getOrNull(idx + 1)?.frame ?: ((traj.frames.lastOrNull() ?: hit.frame) + 1)
            val segmentBounces = bounces.filter { it.frame > hit.frame && it.frame < nextHitFrame }
            if (segmentBounces.any { it.side == targetSide }) continue

            val start = searchSorted(traj.frames, hit.frame)
            val stop = searchSorted(traj.frames, nextHitFrame)
            val segLimit = min(stop - start, MAX_HIT_TO_BOUNCE_FRAMES)
            if (segLimit < 3) continue

            val crossedIdx = (0 until segLimit).firstOrNull {
                if (hit.side == Side.LEFT) traj.xSmooth[start + it] >= netX() + crossMargin
                else traj.xSmooth[start + it] <= netX() - crossMargin
            } ?: continue

            var terminalIdx: Int? = null
            for (j in crossedIdx until segLimit) {
                val x = traj.xSmooth[start + j]
                val y = traj.ySmooth[start + j]
                val returned = if (hit.side == Side.LEFT) traj.dx[start + j] < -MISS_RETURN_X_SPEED else traj.dx[start + j] > MISS_RETURN_X_SPEED
                if (y >= MISS_MAX_Y_RATIO * frameHeight || x <= outMargin || x >= frameWidth - outMargin || returned) {
                    terminalIdx = j
                    break
                }
            }
            if (terminalIdx == null && segLimit == MAX_HIT_TO_BOUNCE_FRAMES) terminalIdx = segLimit - 1
            val terminal = terminalIdx ?: continue
            if (terminal <= crossedIdx) continue

            val terminalFrame = traj.frames[start + terminal]
            if (segmentBounces.any { abs(terminalFrame - it.frame) <= 4 }) continue
            misses.add(Event(
                frame = terminalFrame,
                kind = EventKind.MISS,
                side = hit.side,
                cx = traj.x[start + terminal],
                cy = traj.y[start + terminal],
                confidence = (terminal - crossedIdx).toFloat()
            ))
        }
        return dedupeByFrame(misses, HIT_DEDUP_FRAMES)
    }

    private fun applyEvent(event: Event) {
        when (event.kind) {
            EventKind.HIT -> lastHit = event
            EventKind.BOUNCE -> lastBounce = event
            EventKind.NET, EventKind.MISS -> Unit
        }
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
        val leftScore = score.left + if (winner == Side.LEFT) 1 else 0
        val rightScore = score.right + if (winner == Side.RIGHT) 1 else 0
        score = Score(
            left = leftScore,
            right = rightScore,
            pointWinner = winner,
            eventLabel = label,
            server = servingSide(leftScore, rightScore)
        )
        lastEventLabel = label
        lastEventLabelFrame = frame
    }

    private fun servingSide(leftScore: Int, rightScore: Int): Side {
        val totalPoints = leftScore + rightScore
        val serviceTurn = if (leftScore >= DEUCE_SCORE && rightScore >= DEUCE_SCORE) {
            totalPoints
        } else {
            totalPoints / SERVES_PER_TURN
        }
        return if (serviceTurn % 2 == 0) Side.LEFT else Side.RIGHT
    }

    private fun feedHit(hit: Event): Side? {
        if (phase == Phase.IDLE) {
            phase = if (hit.side == score.server) Phase.AWAIT_SERVE_BOUNCE else Phase.AWAIT_BOUNCE
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

        if (phase == Phase.AWAIT_SERVE_BOUNCE && hit.side == striker && hit.frame - lastStateEventFrame <= HIT_DEDUP_FRAMES) {
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

        if (phase == Phase.AWAIT_SERVE_BOUNCE) {
            val servingSide = striker ?: score.server
            val receivingSide = otherSide(servingSide)
            if (bounce.side == servingSide) {
                phase = Phase.AWAIT_BOUNCE
                targetSide = receivingSide
                lastStateEventFrame = bounce.frame
                return null
            }
            phase = Phase.AWAIT_RETURN
            returnerSide = receivingSide
            strikerSide = servingSide
            lastStateEventFrame = bounce.frame
            return null
        }

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
        val winner = if (phase == Phase.AWAIT_SERVE_BOUNCE || phase == Phase.AWAIT_BOUNCE) {
            otherSide(striker ?: Side.LEFT)
        } else {
            strikerSide ?: Side.LEFT
        }
        resetRally()
        return winner
    }

    private fun feedTimeout(frame: Int): Side? {
        if (phase == Phase.IDLE) return null
        if (
            (phase == Phase.AWAIT_SERVE_BOUNCE || phase == Phase.AWAIT_BOUNCE) &&
            frame - lastStateEventFrame >= MAX_HIT_TO_BOUNCE_FRAMES
        ) {
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

    private fun knownBouncesWith(newBounce: Event): List<Event> {
        val bounces = events.filter { it.kind == EventKind.BOUNCE }.toMutableList()
        if (bounces.none { eventKey(it) == eventKey(newBounce) }) bounces.add(newBounce)
        return bounces.sortedBy { it.frame }
    }

    private fun mergeEvents(
        hits: List<Event>,
        bounces: List<Event>,
        nets: List<Event>,
        misses: List<Event>
    ): List<Event> {
        val priority = mapOf(
            EventKind.NET to 0,
            EventKind.MISS to 1,
            EventKind.BOUNCE to 2,
            EventKind.HIT to 3
        )
        val merged = (hits + bounces + nets + misses)
            .sortedWith(compareBy<Event> { it.frame }.thenBy { priority[it.kind] ?: 99 })
        val deduped = ArrayList<Event>()
        for (event in merged) {
            val previous = deduped.lastOrNull()
            if (previous != null && abs(event.frame - previous.frame) <= 1 && event.kind == previous.kind) {
                if (event.confidence > previous.confidence) deduped[deduped.lastIndex] = event
                continue
            }
            deduped.add(event)
        }
        return deduped
    }

    private fun dedupeByFrame(events: List<Event>, radiusFrames: Int): List<Event> {
        val deduped = ArrayList<Event>()
        for (event in events.sortedBy { it.frame }) {
            val previous = deduped.lastOrNull()
            if (previous != null && event.frame - previous.frame <= radiusFrames) {
                if (event.confidence > previous.confidence) deduped[deduped.lastIndex] = event
                continue
            }
            deduped.add(event)
        }
        return deduped
    }

    private fun smoothSignal(values: List<Float>): List<Float> {
        if (values.size < 5) return values
        return values.mapIndexed { i, value ->
            if (i < 2 || i > values.lastIndex - 2) {
                value
            } else {
                (-3f * values[i - 2] + 12f * values[i - 1] + 17f * values[i] +
                        12f * values[i + 1] - 3f * values[i + 2]) / 35f
            }
        }
    }

    private fun gradient(values: List<Float>): List<Float> {
        if (values.size < 2) return List(values.size) { 0f }
        return values.mapIndexed { i, value ->
            when (i) {
                0 -> values[1] - value
                values.lastIndex -> value - values[i - 1]
                else -> (values[i + 1] - values[i - 1]) * 0.5f
            }
        }
    }

    private fun searchSorted(values: List<Int>, target: Int): Int {
        var lo = 0
        var hi = values.size
        while (lo < hi) {
            val mid = (lo + hi) / 2
            if (values[mid] < target) lo = mid + 1 else hi = mid
        }
        return lo
    }

    private fun eventKey(event: Event): String = "${event.kind.name}:${event.frame}"

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
        private const val HIT_BOUNCE_FILTER_FRAMES = 6
        private const val BOUNCE_MIN_Y_RATIO = 0.42f
        private const val BOUNCE_MAX_Y_RATIO = 0.90f
        private const val BOUNCE_MIN_DY = 1.0f
        private const val BOUNCE_MIN_CURVATURE = 1.8f
        private const val BOUNCE_MIN_X_SPEED = 0.60f
        private const val NET_NEAR_RATIO = 0.07f
        private const val NET_REVERSAL_X_SPEED = 0.90f
        private const val NET_MAX_CROSS_RATIO = 0.03f
        private const val NET_MIN_Y_RATIO = 0.20f
        private const val NET_MAX_Y_RATIO = 0.78f
        private const val NET_POST_HIT_FRAMES = 90
        private const val NET_MIN_PROGRESS_RATIO = 0.35f
        private const val MAX_HIT_TO_BOUNCE_FRAMES = 110
        private const val MAX_BOUNCE_TO_HIT_FRAMES = 140
        private const val BALL_LOST_FRAMES = 60
        private const val EVENT_TEXT_FADE_FRAMES = 26
        private const val EVENT_MARKER_FRAMES = 36
        private const val MISS_MAX_Y_RATIO = 0.88f
        private const val MISS_OUT_X_MARGIN_RATIO = 0.03f
        private const val MISS_RETURN_X_SPEED = 0.75f
        private const val MAX_HISTORY = 240
        private const val MAX_EVENTS = 32
        private const val SERVES_PER_TURN = 2
        private const val DEUCE_SCORE = 10
    }
}
