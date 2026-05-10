package com.example.ballrealtime

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import androidx.camera.core.ImageProxy
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtProvider
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.util.LinkedHashMap
import kotlin.math.max
import kotlin.math.min

class YoloQnnDetector(
    context: Context,
    private val modelAssetName: String = "yolo_det.onnx"
) {
    data class PreparedInput(
        val ready: Boolean,
        val input: FloatArray,
        val modelWidth: Int,
        val modelHeight: Int,
        val preprocessMs: Long
    )

    data class InferenceResult(
        val ready: Boolean,
        val cx: Float,
        val cy: Float,
        val confidence: Float,
        val threshold: Float,
        val rect: RectF,
        val outputShape: String,
        val backend: String,
        val modelWidth: Int,
        val modelHeight: Int,
        val preprocessMs: Long,
        val inferenceMs: Long,
        val postprocessMs: Long,
        val profilePath: String = ""
    )

    private data class Detection(
        val box: RectF,
        val score: Float
    )

    private data class BitmapInput(
        val input: FloatArray,
        val sourceWidth: Int,
        val sourceHeight: Int,
        val scale: Float,
        val padX: Float,
        val padY: Float,
        val preprocessMs: Long
    )

    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val options: OrtSession.SessionOptions
    private val session: OrtSession
    private val inputName: String
    private val backendName = "QNN"
    private val modelSize = 640
    private val confidenceThreshold = 0.2f
    private val iouThreshold = 0.25f
    private var yPlaneBytes = ByteArray(0)
    private var uPlaneBytes = ByteArray(0)
    private var vPlaneBytes = ByteArray(0)

    init {
        if (!OrtEnvironment.getAvailableProviders().contains(OrtProvider.QNN)) {
            throw IllegalStateException("QNNExecutionProvider is not available")
        }

        val modelFile = ensureModelFile(context, modelAssetName)
        options = createQnnOptions(context, "yolo_det")
        session = env.createSession(modelFile.absolutePath, options)
        inputName = session.inputNames.firstOrNull() ?: "images"
    }

    fun warmUp() {
        OnnxTensor.createTensor(
            env,
            FloatBuffer.allocate(3 * modelSize * modelSize),
            longArrayOf(1, 3, modelSize.toLong(), modelSize.toLong())
        ).use { images ->
            session.run(mapOf(inputName to images)).use {
                // Fully materialize the graph once, matching the Qualcomm tester flow.
            }
        }
    }

    fun prepareInput(imageProxy: ImageProxy, copyForInference: Boolean = true): PreparedInput {
        val start = System.nanoTime()
        val input = if (copyForInference) {
            preprocessFrame(imageProxy)
        } else {
            FloatArray(0)
        }
        return PreparedInput(
            ready = copyForInference,
            input = input,
            modelWidth = modelSize,
            modelHeight = modelSize,
            preprocessMs = nanosToMillis(System.nanoTime() - start)
        )
    }

    fun runPreparedInput(prepared: PreparedInput): InferenceResult {
        if (!prepared.ready || prepared.input.isEmpty()) {
            return InferenceResult(
                ready = prepared.ready,
                cx = -1f,
                cy = -1f,
                confidence = 0f,
                threshold = confidenceThreshold,
                rect = RectF(),
                outputShape = if (prepared.ready) "empty" else "warming",
                backend = backendName,
                modelWidth = modelSize,
                modelHeight = modelSize,
                preprocessMs = prepared.preprocessMs,
                inferenceMs = 0L,
                postprocessMs = 0L
            )
        }

        OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(prepared.input),
            longArrayOf(1, 3, modelSize.toLong(), modelSize.toLong())
        ).use { images ->
            val inferenceStart = System.nanoTime()
            session.run(mapOf(inputName to images)).use { result ->
                val inferenceMs = nanosToMillis(System.nanoTime() - inferenceStart)
                val postStart = System.nanoTime()
                val output = extractYoloOutput(result[0].value)
                val detection = output?.let { postprocessYolo(it).firstOrNull() }
                val postMs = nanosToMillis(System.nanoTime() - postStart)
                val shapeText = output?.let { "1x${it.size}x${it.firstOrNull()?.size ?: 0}" } ?: "unknown"
                if (detection == null) {
                    return InferenceResult(
                        ready = true,
                        cx = -1f,
                        cy = -1f,
                        confidence = 0f,
                        threshold = confidenceThreshold,
                        rect = RectF(),
                        outputShape = shapeText,
                        backend = backendName,
                        modelWidth = modelSize,
                        modelHeight = modelSize,
                        preprocessMs = prepared.preprocessMs,
                        inferenceMs = inferenceMs,
                        postprocessMs = postMs
                    )
                }

                return InferenceResult(
                    ready = true,
                    cx = detection.box.centerX(),
                    cy = detection.box.centerY(),
                    confidence = detection.score,
                    threshold = confidenceThreshold,
                    rect = detection.box,
                    outputShape = shapeText,
                    backend = backendName,
                    modelWidth = modelSize,
                    modelHeight = modelSize,
                    preprocessMs = prepared.preprocessMs,
                    inferenceMs = inferenceMs,
                    postprocessMs = postMs
                )
            }
        }
    }

    fun predictFromBitmap(bitmap: Bitmap): InferenceResult? {
        val prepared = preprocessBitmap(bitmap)
        OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(prepared.input),
            longArrayOf(1, 3, modelSize.toLong(), modelSize.toLong())
        ).use { images ->
            val inferenceStart = System.nanoTime()
            session.run(mapOf(inputName to images)).use { result ->
                val inferenceMs = nanosToMillis(System.nanoTime() - inferenceStart)
                val postStart = System.nanoTime()
                val output = extractYoloOutput(result[0].value)
                val detection = output?.let { postprocessYolo(it).firstOrNull() }
                val postMs = nanosToMillis(System.nanoTime() - postStart)
                val shapeText = output?.let { "1x${it.size}x${it.firstOrNull()?.size ?: 0}" } ?: "unknown"
                if (detection == null) {
                    return InferenceResult(
                        ready = true,
                        cx = -1f,
                        cy = -1f,
                        confidence = 0f,
                        threshold = confidenceThreshold,
                        rect = RectF(),
                        outputShape = shapeText,
                        backend = backendName,
                        modelWidth = prepared.sourceWidth,
                        modelHeight = prepared.sourceHeight,
                        preprocessMs = prepared.preprocessMs,
                        inferenceMs = inferenceMs,
                        postprocessMs = postMs
                    )
                }

                val mapped = mapModelBoxToSource(detection.box, prepared)
                return InferenceResult(
                    ready = true,
                    cx = mapped.centerX(),
                    cy = mapped.centerY(),
                    confidence = detection.score,
                    threshold = confidenceThreshold,
                    rect = mapped,
                    outputShape = shapeText,
                    backend = backendName,
                    modelWidth = prepared.sourceWidth,
                    modelHeight = prepared.sourceHeight,
                    preprocessMs = prepared.preprocessMs,
                    inferenceMs = inferenceMs,
                    postprocessMs = postMs
                )
            }
        }
    }

    fun close() {
        session.close()
        options.close()
    }

    private fun postprocessYolo(output: Array<FloatArray>): List<Detection> {
        if (output.size < 5) return emptyList()
        if (output.firstOrNull()?.size == 6) {
            return postprocessYoloEndToEnd(output)
        }

        val anchors = output[0].size
        val candidates = ArrayList<Detection>()
        for (i in 0 until anchors) {
            var ballScore = output[4][i]
            if (ballScore < confidenceThreshold) continue
            for (c in 5 until output.size) {
                if (output[c][i] > ballScore) {
                    ballScore = -1f
                    break
                }
            }
            if (ballScore < confidenceThreshold) continue

            val cx = output[0][i]
            val cy = output[1][i]
            val w = output[2][i]
            val h = output[3][i]
            val box = RectF(
                clamp(cx - w / 2f, 0f, modelSize.toFloat()),
                clamp(cy - h / 2f, 0f, modelSize.toFloat()),
                clamp(cx + w / 2f, 0f, modelSize.toFloat()),
                clamp(cy + h / 2f, 0f, modelSize.toFloat())
            )
            if (box.width() >= 1f && box.height() >= 1f) {
                candidates.add(Detection(box, ballScore))
            }
        }

        candidates.sortByDescending { it.score }
        val kept = ArrayList<Detection>()
        for (candidate in candidates) {
            if (kept.none { iou(candidate.box, it.box) > iouThreshold }) {
                kept.add(candidate)
            }
            if (kept.size >= 20) break
        }
        return kept
    }

    private fun postprocessYoloEndToEnd(output: Array<FloatArray>): List<Detection> {
        val candidates = ArrayList<Detection>()
        for (row in output) {
            if (row.size < 6) continue
            val score = row[4]
            if (score < confidenceThreshold) continue
            val cls = row[5].toInt()
            if (cls != 0) continue

            val box = RectF(
                clamp(row[0], 0f, modelSize.toFloat()),
                clamp(row[1], 0f, modelSize.toFloat()),
                clamp(row[2], 0f, modelSize.toFloat()),
                clamp(row[3], 0f, modelSize.toFloat())
            )
            if (box.width() >= 1f && box.height() >= 1f) {
                candidates.add(Detection(box, score))
            }
        }

        candidates.sortByDescending { it.score }
        return candidates.take(20)
    }

    private fun extractYoloOutput(value: Any?): Array<FloatArray>? {
        if (value is OnnxTensor) {
            val info = value.info as? TensorInfo ?: return null
            val shape = info.shape
            val channels = shape.getOrNull(shape.size - 2)?.toInt() ?: return null
            val anchors = shape.getOrNull(shape.size - 1)?.toInt() ?: return null
            val buffer = value.floatBuffer
            val output = Array(channels) { FloatArray(anchors) }
            for (c in 0 until channels) {
                buffer.get(output[c])
            }
            return output
        }

        val batch = value as? Array<*> ?: return null
        val first = batch.firstOrNull() as? Array<*> ?: return null
        return Array(first.size) { c ->
            when (val channel = first[c]) {
                is FloatArray -> channel
                is Array<*> -> FloatArray(channel.size) { i -> (channel[i] as? Number)?.toFloat() ?: 0f }
                else -> FloatArray(0)
            }
        }
    }

    private fun preprocessFrame(imageProxy: ImageProxy): FloatArray {
        val rotation = ((imageProxy.imageInfo.rotationDegrees % 360) + 360) % 360
        val sourceWidth = if (rotation == 90 || rotation == 270) imageProxy.height else imageProxy.width
        val sourceHeight = if (rotation == 90 || rotation == 270) imageProxy.width else imageProxy.height
        val scale = min(modelSize / sourceWidth.toFloat(), modelSize / sourceHeight.toFloat())
        val padX = (modelSize - sourceWidth * scale) / 2f
        val padY = (modelSize - sourceHeight * scale) / 2f

        val yPlane = imageProxy.planes[0]
        val uPlane = imageProxy.planes[1]
        val vPlane = imageProxy.planes[2]
        yPlaneBytes = ensurePlaneCapacity(yPlaneBytes, yPlane.buffer.capacity())
        uPlaneBytes = ensurePlaneCapacity(uPlaneBytes, uPlane.buffer.capacity())
        vPlaneBytes = ensurePlaneCapacity(vPlaneBytes, vPlane.buffer.capacity())
        copyPlaneBytes(yPlane.buffer, yPlaneBytes)
        copyPlaneBytes(uPlane.buffer, uPlaneBytes)
        copyPlaneBytes(vPlane.buffer, vPlaneBytes)

        val area = modelSize * modelSize
        val data = FloatArray(3 * area)
        val gray = 114f / 255f
        for (dy in 0 until modelSize) {
            val unpaddedY = (dy - padY) / scale
            val rowOffset = dy * modelSize
            for (dx in 0 until modelSize) {
                val idx = rowOffset + dx
                val unpaddedX = (dx - padX) / scale
                if (unpaddedX < 0f || unpaddedY < 0f ||
                    unpaddedX >= sourceWidth || unpaddedY >= sourceHeight
                ) {
                    data[idx] = gray
                    data[area + idx] = gray
                    data[area * 2 + idx] = gray
                    continue
                }

                val sx = unpaddedX.toInt().coerceIn(0, sourceWidth - 1)
                val sy = unpaddedY.toInt().coerceIn(0, sourceHeight - 1)
                val src = mapRotatedPoint(
                    sx,
                    sy,
                    rotation,
                    imageProxy.width,
                    imageProxy.height
                )
                val yIndex = src.second * yPlane.rowStride + src.first * yPlane.pixelStride
                val uvX = src.first / 2
                val uvY = src.second / 2
                val uIndex = uvY * uPlane.rowStride + uvX * uPlane.pixelStride
                val vIndex = uvY * vPlane.rowStride + uvX * vPlane.pixelStride

                val yValue = yPlaneBytes[yIndex].toInt() and 0xFF
                val uValue = (uPlaneBytes[uIndex].toInt() and 0xFF) - 128
                val vValue = (vPlaneBytes[vIndex].toInt() and 0xFF) - 128

                val r = (yValue + 1.402f * vValue).coerceIn(0f, 255f) / 255f
                val g = (yValue - 0.344136f * uValue - 0.714136f * vValue).coerceIn(0f, 255f) / 255f
                val b = (yValue + 1.772f * uValue).coerceIn(0f, 255f) / 255f
                data[idx] = r
                data[area + idx] = g
                data[area * 2 + idx] = b
            }
        }
        return data
    }

    private fun preprocessBitmap(bitmap: Bitmap): BitmapInput {
        val start = System.nanoTime()
        val sourceWidth = bitmap.width
        val sourceHeight = bitmap.height
        val scale = min(modelSize / sourceWidth.toFloat(), modelSize / sourceHeight.toFloat())
        val padX = (modelSize - sourceWidth * scale) / 2f
        val padY = (modelSize - sourceHeight * scale) / 2f
        val pixels = IntArray(sourceWidth * sourceHeight)
        bitmap.getPixels(pixels, 0, sourceWidth, 0, 0, sourceWidth, sourceHeight)

        val area = modelSize * modelSize
        val data = FloatArray(3 * area)
        val gray = 114f / 255f
        for (dy in 0 until modelSize) {
            val unpaddedY = (dy - padY) / scale
            val rowOffset = dy * modelSize
            for (dx in 0 until modelSize) {
                val idx = rowOffset + dx
                val unpaddedX = (dx - padX) / scale
                if (unpaddedX < 0f || unpaddedY < 0f ||
                    unpaddedX >= sourceWidth || unpaddedY >= sourceHeight
                ) {
                    data[idx] = gray
                    data[area + idx] = gray
                    data[area * 2 + idx] = gray
                    continue
                }

                val sx = unpaddedX.toInt().coerceIn(0, sourceWidth - 1)
                val sy = unpaddedY.toInt().coerceIn(0, sourceHeight - 1)
                val color = pixels[sy * sourceWidth + sx]
                data[idx] = ((color shr 16) and 0xFF) / 255f
                data[area + idx] = ((color shr 8) and 0xFF) / 255f
                data[area * 2 + idx] = (color and 0xFF) / 255f
            }
        }

        return BitmapInput(
            input = data,
            sourceWidth = sourceWidth,
            sourceHeight = sourceHeight,
            scale = scale,
            padX = padX,
            padY = padY,
            preprocessMs = nanosToMillis(System.nanoTime() - start)
        )
    }

    private fun mapModelBoxToSource(box: RectF, input: BitmapInput): RectF {
        return RectF(
            clamp((box.left - input.padX) / input.scale, 0f, input.sourceWidth.toFloat()),
            clamp((box.top - input.padY) / input.scale, 0f, input.sourceHeight.toFloat()),
            clamp((box.right - input.padX) / input.scale, 0f, input.sourceWidth.toFloat()),
            clamp((box.bottom - input.padY) / input.scale, 0f, input.sourceHeight.toFloat())
        )
    }

    private fun mapRotatedPoint(
        x: Int,
        y: Int,
        rotation: Int,
        sensorWidth: Int,
        sensorHeight: Int
    ): Pair<Int, Int> {
        return when (rotation) {
            90 -> y to sensorHeight - 1 - x
            180 -> sensorWidth - 1 - x to sensorHeight - 1 - y
            270 -> sensorWidth - 1 - y to x
            else -> x to y
        }
    }

    private fun iou(a: RectF, b: RectF): Float {
        val left = max(a.left, b.left)
        val top = max(a.top, b.top)
        val right = min(a.right, b.right)
        val bottom = min(a.bottom, b.bottom)
        val intersection = max(0f, right - left) * max(0f, bottom - top)
        val union = a.width() * a.height() + b.width() * b.height() - intersection
        return if (union <= 0f) 0f else intersection / union
    }

    private fun createQnnOptions(
        context: Context,
        name: String
    ): OrtSession.SessionOptions {
        val opts = OrtSession.SessionOptions()
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        opts.addConfigEntry("session.disable_cpu_ep_fallback", "0")
        val nativeLibraryDir = context.applicationInfo.nativeLibraryDir
        val qnn = LinkedHashMap<String, String>().apply {
            put("backend_path", File(nativeLibraryDir, "libQnnHtp.so").absolutePath)
            put("htp_performance_mode", "burst")
            put("enable_htp_fp16_precision", "1")
            put("htp_graph_finalization_optimization_mode", "3")
            put("profiling_level", "basic")
            put("profiling_file_path", File(context.filesDir, "${name}_qnn_profile.csv").absolutePath)
        }
        opts.addQnn(qnn)
        return opts
    }

    private fun ensureModelFile(context: Context, assetName: String): File {
        val dir = File(context.filesDir, "weights_detection")
        if (!dir.exists() && !dir.mkdirs()) {
            throw IllegalStateException("Cannot create ${dir.absolutePath}")
        }
        val out = File(dir, assetName)
        val assetLength = runCatching {
            context.assets.openFd(assetName).use { it.length }
        }.getOrDefault(-1L)
        if (out.exists() && out.length() > 0L && (assetLength <= 0L || out.length() == assetLength)) {
            return out
        }

        val tmp = File(dir, "$assetName.tmp")
        context.assets.open(assetName).use { input ->
            FileOutputStream(tmp).use { output ->
                input.copyTo(output, bufferSize = 1024 * 1024)
            }
        }
        if (out.exists() && !out.delete()) {
            throw IllegalStateException("Cannot replace ${out.absolutePath}")
        }
        if (!tmp.renameTo(out)) {
            throw IllegalStateException("Cannot move ${tmp.absolutePath} to ${out.absolutePath}")
        }
        return out
    }

    private fun ensurePlaneCapacity(buffer: ByteArray, required: Int): ByteArray {
        return if (buffer.size >= required) buffer else ByteArray(required)
    }

    private fun copyPlaneBytes(source: java.nio.ByteBuffer, dest: ByteArray) {
        val duplicate = source.duplicate()
        duplicate.rewind()
        duplicate.get(dest, 0, duplicate.remaining())
    }

    private fun clamp(value: Float, lower: Float, upper: Float): Float {
        return max(lower, min(value, upper))
    }

    private fun nanosToMillis(nanos: Long): Long {
        return max(0L, nanos / 1_000_000L)
    }
}
