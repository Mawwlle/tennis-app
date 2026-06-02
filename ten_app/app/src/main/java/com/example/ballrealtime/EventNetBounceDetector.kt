package com.example.ballrealtime

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import kotlin.math.exp

class EventNetBounceDetector(
    context: Context,
    private val modelAssetName: String = "eventnet_best.onnx"
) {
    private data class FramePoint(
        val frame: Int,
        val cx: Float?,
        val cy: Float?,
        val width: Int,
        val height: Int
    )

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val options = OrtSession.SessionOptions()
    private val session: OrtSession
    private val inputName: String
    private val frames = ArrayDeque<FramePoint>()
    private var lastBounceFrame = -EVENTNET_DEDUP_FRAMES * 2

    init {
        val modelFile = ensureModelFile(context, modelAssetName)
        session = env.createSession(modelFile.absolutePath, options)
        inputName = session.inputNames.firstOrNull() ?: "heatmaps"
    }

    fun addDetection(
        frame: Int,
        cx: Float,
        cy: Float,
        width: Int,
        height: Int,
        netX: Float
    ): GameState.Event? {
        return addFrame(FramePoint(frame, cx, cy, width, height), netX)
    }

    fun addMissingFrame(frame: Int, width: Int, height: Int, netX: Float): GameState.Event? {
        return addFrame(FramePoint(frame, null, null, width, height), netX)
    }

    fun reset() {
        frames.clear()
        lastBounceFrame = -EVENTNET_DEDUP_FRAMES * 2
    }

    fun close() {
        session.close()
        options.close()
    }

    private fun addFrame(point: FramePoint, netX: Float): GameState.Event? {
        frames.addLast(point)
        while (frames.size > EVENTNET_WINDOW) frames.removeFirst()
        if (frames.size < EVENTNET_WINDOW) return null

        val window = frames.toList()
        val center = window[EVENTNET_HALF]
        val cx = center.cx ?: return null
        val cy = center.cy ?: return null
        val score = runWindow(window)
        if (score < EVENTNET_THRESHOLD) return null
        if (center.frame - lastBounceFrame < EVENTNET_DEDUP_FRAMES) return null

        lastBounceFrame = center.frame
        val side = if (cx < netX) GameState.Side.LEFT else GameState.Side.RIGHT
        return GameState.Event(
            frame = center.frame,
            kind = GameState.EventKind.BOUNCE,
            side = side,
            cx = cx,
            cy = cy,
            confidence = score
        )
    }

    private fun runWindow(window: List<FramePoint>): Float {
        val input = FloatArray(EVENTNET_WINDOW * HEATMAP_H * HEATMAP_W)
        for ((t, point) in window.withIndex()) {
            fillHeatmap(input, t * HEATMAP_H * HEATMAP_W, point)
        }

        OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(input),
            longArrayOf(1, EVENTNET_WINDOW.toLong(), HEATMAP_H.toLong(), HEATMAP_W.toLong())
        ).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { result ->
                val logits = extractLogits(result[0].value)
                return sigmoid(logits.getOrElse(0) { Float.NEGATIVE_INFINITY })
            }
        }
    }

    private fun fillHeatmap(output: FloatArray, offset: Int, point: FramePoint) {
        val cx = point.cx
        val cy = point.cy
        if (cx == null || cy == null || point.width <= 0 || point.height <= 0) return
        val px = (cx / point.width.toFloat()) * HEATMAP_W
        val py = (cy / point.height.toFloat()) * HEATMAP_H
        for (y in 0 until HEATMAP_H) {
            val dy = y.toFloat() - py
            val row = offset + y * HEATMAP_W
            for (x in 0 until HEATMAP_W) {
                val dx = x.toFloat() - px
                output[row + x] = exp(
                    -((dx * dx + dy * dy) / (2f * HEATMAP_SIGMA * HEATMAP_SIGMA)).toDouble()
                ).toFloat()
            }
        }
    }

    private fun extractLogits(value: Any?): FloatArray {
        if (value is OnnxTensor) {
            val info = value.info
            val size = info.shape.fold(1L) { acc, dim -> acc * dim.coerceAtLeast(1L) }.toInt()
            val out = FloatArray(size)
            value.floatBuffer.get(out)
            return out
        }
        val batch = value as? Array<*> ?: return FloatArray(0)
        val first = batch.firstOrNull()
        return when (first) {
            is FloatArray -> first
            is Array<*> -> FloatArray(first.size) { i -> (first[i] as? Number)?.toFloat() ?: 0f }
            else -> FloatArray(0)
        }
    }

    private fun sigmoid(value: Float): Float {
        return (1.0 / (1.0 + exp(-value.toDouble()))).toFloat()
    }

    private fun ensureModelFile(context: Context, assetName: String): File {
        val outFile = File(context.filesDir, assetName)
        if (outFile.exists() && outFile.length() > 0) return outFile
        context.assets.open(assetName).use { input ->
            FileOutputStream(outFile).use { output -> input.copyTo(output) }
        }
        return outFile
    }

    companion object {
        private const val EVENTNET_WINDOW = 15
        private const val EVENTNET_HALF = EVENTNET_WINDOW / 2
        private const val EVENTNET_THRESHOLD = 0.35f
        private const val EVENTNET_DEDUP_FRAMES = 8
        private const val HEATMAP_H = 36
        private const val HEATMAP_W = 64
        private const val HEATMAP_SIGMA = 2.0f
    }
}
