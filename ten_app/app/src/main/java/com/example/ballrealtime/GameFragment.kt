package com.example.ballrealtime

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.PointF
import android.hardware.camera2.CaptureRequest
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.Process
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.util.Range
import android.util.Size
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.example.ballrealtime.databinding.FragmentGameBinding
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.max

class GameFragment : Fragment() {
    private enum class CalibrationStep { TABLE, PEOPLE, READY }

    private var binding: FragmentGameBinding? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var analysisExecutor: ExecutorService? = null
    private var detector: YoloQnnDetector? = null
    private var eventNetDetector: EventNetBounceDetector? = null
    private val gameState = GameState()
    private val mainHandler = Handler(Looper.getMainLooper())
    private val trailPoints = ArrayDeque<TrailPoint>()

    private var frameIndex = 0
    private var modelWidth = 640
    private var modelHeight = 640
    private var smoothedPoint: PointF? = null
    private var smoothedVelocity: PointF? = null
    private var smoothedAtMs = 0L
    private var lastDetectionAtMs = 0L
    private var avgInferenceMs = 0f
    private var lastRenderedInferenceFrame = 0
    @Volatile
    private var calibrationStep = CalibrationStep.TABLE
    private var calibratedPeoplePositions: List<PointF> = emptyList()
    private var textToSpeech: TextToSpeech? = null
    private var ttsReady = false
    private var lastAnnouncedLeftScore = 0
    private var lastAnnouncedRightScore = 0

    private data class DetectionPoint(
        val frame: Int,
        val cx: Float,
        val cy: Float,
        val capturedAtMs: Long
    )

    private data class TrailPoint(
        val point: PointF,
        val capturedAtMs: Long
    )

    private val modelAssetName = "yolo_det.onnx"
    private val eventNetAssetName = "eventnet_best.onnx"
    private val analysisSize = Size(320, 180)
    private val trailWindow = 18
    private val maxExtrapMs = 220L
    private val trailKeepMs = 1200L
    private val displayLeadMs = 0L
    private val measurementBlend = 1.0f
    private val velocityCorrection = 0.0f

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            startCameraPipeline()
        } else {
            Toast.makeText(requireContext(), "Camera permission denied", Toast.LENGTH_LONG).show()
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        val fragmentBinding = FragmentGameBinding.inflate(inflater, container, false)
        binding = fragmentBinding
        return fragmentBinding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        enablePerformanceMode()
        setupTextToSpeech()
        analysisExecutor = createAnalysisExecutor()
        setupCalibrationControls()
        createDetector()
        if (hasCameraPermission()) {
            startCameraPipeline()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun setupCalibrationControls() {
        val binding = binding ?: return
        calibrationStep = CalibrationStep.TABLE
        binding.calibrationPanel.visibility = View.VISIBLE
        binding.calibrationOverlay.mode = CalibrationOverlayView.Mode.TABLE
        binding.calibrationOverlay.onPeopleChanged = {
            renderCalibrationStep()
        }
        renderCalibrationStep()
        binding.calibrationNextButton.setOnClickListener {
            when (calibrationStep) {
                CalibrationStep.TABLE -> {
                    applyTableCalibration()
                    calibrationStep = CalibrationStep.PEOPLE
                    binding.calibrationOverlay.mode = CalibrationOverlayView.Mode.PEOPLE
                    binding.calibrationOverlay.resetPeople()
                    renderCalibrationStep()
                }
                CalibrationStep.PEOPLE -> {
                    if (binding.calibrationOverlay.peoplePositions().size < 2) {
                        binding.calibrationMessage.text =
                            "Выберите два примерных положения игроков у стола: левый игрок, затем правый игрок."
                        return@setOnClickListener
                    }
                    calibratedPeoplePositions = binding.calibrationOverlay.peoplePositions()
                    resetTrackingState()
                    applyTableCalibration()
                    binding.calibrationOverlay.mode = CalibrationOverlayView.Mode.HIDDEN
                    binding.calibrationPanel.visibility = View.GONE
                    calibrationStep = CalibrationStep.READY
                    speak(serverText(GameState.Side.LEFT))
                }
                CalibrationStep.READY -> Unit
            }
        }
    }

    private fun renderCalibrationStep() {
        val binding = binding ?: return
        when (calibrationStep) {
            CalibrationStep.TABLE -> {
                binding.calibrationTitle.text = "Калибровка стола"
                binding.calibrationMessage.text =
                    "Для калибровки выставите рамки стола, перемещая углы полигона на экране."
                binding.calibrationNextButton.text = "Стол готов"
            }
            CalibrationStep.PEOPLE -> {
                binding.calibrationTitle.text = "Положение людей в кадре"
                binding.calibrationMessage.text =
                    "Нажатием на экран выберите примерное расположение людей в кадре: позиции у стола."
                val count = binding.calibrationOverlay.peoplePositions().size
                binding.calibrationNextButton.text = if (count < 2) "Продолжить" else "Игроки выбраны"
            }
            CalibrationStep.READY -> Unit
        }
    }

    private fun applyTableCalibration() {
        val binding = binding ?: return
        val points = binding.calibrationOverlay.tablePolygon()
        if (points.isEmpty()) return
        val xs = points.map {
            mapXOverlayToSource(
                it.x,
                modelWidth,
                modelHeight,
                binding.detectionOverlay.width,
                binding.detectionOverlay.height
            )
        }
        val ys = points.map {
            mapYOverlayToSource(
                it.y,
                modelWidth,
                modelHeight,
                binding.detectionOverlay.width,
                binding.detectionOverlay.height
            )
        }
        val left = (xs.minOrNull() ?: 0f) / modelWidth.toFloat()
        val right = (xs.maxOrNull() ?: modelWidth.toFloat()) / modelWidth.toFloat()
        val top = (ys.minOrNull() ?: 0f) / modelHeight.toFloat()
        val bottom = (ys.maxOrNull() ?: modelHeight.toFloat()) / modelHeight.toFloat()
        gameState.updateTableBounds(left, top, right, bottom)
    }

    private fun createDetector() {
        try {
            detector = YoloQnnDetector(requireContext(), modelAssetName).also { it.warmUp() }
        } catch (e: Exception) {
            detector?.close()
            detector = null
            Toast.makeText(requireContext(), "ONNX init error: ${e.message}", Toast.LENGTH_LONG).show()
        }
        try {
            eventNetDetector = EventNetBounceDetector(requireContext(), eventNetAssetName)
        } catch (e: Exception) {
            eventNetDetector?.close()
            eventNetDetector = null
            Toast.makeText(requireContext(), "EventNet init error: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun setupTextToSpeech() {
        textToSpeech = TextToSpeech(requireContext()) { status ->
            ttsReady = status == TextToSpeech.SUCCESS
            if (ttsReady) {
                textToSpeech?.language = Locale("ru", "RU")
                textToSpeech?.setSpeechRate(1.0f)
            }
        }
    }

    private fun speak(text: String) {
        if (!ttsReady) return
        textToSpeech?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "game-${SystemClock.uptimeMillis()}")
    }

    private fun announceScoreIfNeeded(score: GameState.Score) {
        if (score.left == lastAnnouncedLeftScore && score.right == lastAnnouncedRightScore) return
        val winner = when {
            score.left > lastAnnouncedLeftScore -> GameState.Side.LEFT
            score.right > lastAnnouncedRightScore -> GameState.Side.RIGHT
            else -> null
        }
        lastAnnouncedLeftScore = score.left
        lastAnnouncedRightScore = score.right
        val winnerText = when (winner) {
            GameState.Side.LEFT -> "Очко левому игроку"
            GameState.Side.RIGHT -> "Очко правому игроку"
            null -> return
        }
        speak("$winnerText. ${serverText(score.server)}")
    }

    private fun serverText(server: GameState.Side): String {
        return if (server == GameState.Side.LEFT) "Подача левого игрока" else "Подача правого игрока"
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            requireContext(),
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun startCameraPipeline() {
        val binding = binding ?: return
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener({
            val localCameraProvider = cameraProviderFuture.get()
            cameraProvider = localCameraProvider

            val fpsCandidates = listOf(Range(30, 40), Range(30, 30), null)
            var lastError: Exception? = null
            for (fpsRange in fpsCandidates) {
                val previewBuilder = Preview.Builder()
                val analysisBuilder = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setTargetResolution(analysisSize)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                fpsRange?.let {
                    Camera2Interop.Extender(previewBuilder)
                        .setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, it)
                    Camera2Interop.Extender(analysisBuilder)
                        .setCaptureRequestOption(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, it)
                }

                val preview = previewBuilder.build().also {
                    it.surfaceProvider = binding.previewView.surfaceProvider
                }
                val imageAnalysis = analysisBuilder.build()
                imageAnalysis.setAnalyzer(analysisExecutor!!) { imageProxy ->
                    analyzeFrame(imageProxy)
                }

                try {
                    localCameraProvider.unbindAll()
                    localCameraProvider.bindToLifecycle(
                        viewLifecycleOwner,
                        CameraSelector.DEFAULT_BACK_CAMERA,
                        preview,
                        imageAnalysis
                    )
                    lastError = null
                    break
                } catch (e: Exception) {
                    lastError = e
                }
            }
            if (lastError != null) {
                Toast.makeText(requireContext(), "Camera start error: ${lastError.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    private fun analyzeFrame(imageProxy: androidx.camera.core.ImageProxy) {
        if (calibrationStep != CalibrationStep.READY) {
            imageProxy.close()
            return
        }

        val localDetector = detector
        if (localDetector == null) {
            imageProxy.close()
            return
        }

        var imageClosed = false
        try {
            frameIndex += 1
            val currentFrame = frameIndex
            val capturedAtMs = SystemClock.uptimeMillis()
            val prepared = localDetector.prepareInput(imageProxy, copyForInference = true)
            imageProxy.close()
            imageClosed = true
            if (prepared.ready) {
                val inference = localDetector.runPreparedInput(prepared)
                avgInferenceMs = if (avgInferenceMs == 0f) {
                    inference.inferenceMs.toFloat()
                } else {
                    avgInferenceMs * 0.8f + inference.inferenceMs.toFloat() * 0.2f
                }
                mainHandler.post {
                    if (currentFrame <= lastRenderedInferenceFrame) return@post
                    lastRenderedInferenceFrame = currentFrame
                    renderDetection(currentFrame, capturedAtMs, inference)
                }
            }
        } catch (e: Exception) {
            if (!imageClosed) {
                imageProxy.close()
            }
            requireActivity().runOnUiThread {
                binding?.detectionOverlay?.clearDetection()
                resetTrackingState()
            }
        }
    }

    private fun renderDetection(
        frame: Int,
        capturedAtMs: Long,
        inference: YoloQnnDetector.InferenceResult
    ) {
        modelWidth = inference.modelWidth
        modelHeight = inference.modelHeight
        gameState.updateFrameSize(modelWidth, modelHeight)

        val detection = if (inference.ready &&
            inference.confidence >= inference.threshold &&
            inference.cx >= 0f
        ) {
            lastDetectionAtMs = SystemClock.uptimeMillis()
            DetectionPoint(frame, inference.cx, inference.cy, capturedAtMs)
        } else {
            null
        }

        val eventNetBounce = if (detection != null) {
            eventNetDetector?.addDetection(
                frame = frame,
                cx = detection.cx,
                cy = detection.cy,
                width = modelWidth,
                height = modelHeight,
                netX = gameState.currentNetX()
            )
        } else {
            eventNetDetector?.addMissingFrame(
                frame = frame,
                width = modelWidth,
                height = modelHeight,
                netX = gameState.currentNetX()
            )
        }

        val snapshot = if (detection != null) {
            gameState.addDetection(
                GameState.Detection(
                    frame = frame,
                    cx = detection.cx,
                    cy = detection.cy,
                    confidence = inference.confidence
                ),
                eventNetBounce = eventNetBounce
            )
        } else {
            gameState.addMissingFrame(frame, eventNetBounce = eventNetBounce)
        }
        if (detection != null) {
            updateTracking(detection)
        }
        renderPredictedOverlay(snapshot)
        announceScoreIfNeeded(snapshot.score)
    }

    private fun updateTracking(detection: DetectionPoint) {
        val measurement = PointF(detection.cx, detection.cy)
        val previousPoint = smoothedPoint
        if (previousPoint == null || smoothedAtMs == 0L) {
            smoothedPoint = measurement
            smoothedVelocity = PointF(0f, 0f)
            smoothedAtMs = detection.capturedAtMs
            return
        }

        val dtMs = max(1L, detection.capturedAtMs - smoothedAtMs)
        val dt = dtMs.toFloat()
        val velocity = smoothedVelocity ?: PointF(0f, 0f)
        val predicted = PointF(
            previousPoint.x + velocity.x * dt,
            previousPoint.y + velocity.y * dt
        )
        val residual = PointF(
            measurement.x - predicted.x,
            measurement.y - predicted.y
        )

        smoothedPoint = PointF(
            predicted.x + residual.x * measurementBlend,
            predicted.y + residual.y * measurementBlend
        )
        smoothedVelocity = PointF(
            velocity.x + (residual.x / dt) * velocityCorrection,
            velocity.y + (residual.y / dt) * velocityCorrection
        )
        smoothedAtMs = detection.capturedAtMs
    }

    private fun predictedPointAt(targetMs: Long): PointF? {
        val point = smoothedPoint ?: return null
        val velocity = smoothedVelocity ?: PointF(0f, 0f)
        if (smoothedAtMs == 0L) return point
        val ageMs = targetMs - smoothedAtMs
        if (ageMs > maxExtrapMs) return null
        val dt = ageMs.coerceAtLeast(0L).toFloat()
        val damping = (1f - dt / maxExtrapMs.toFloat()).coerceIn(0.25f, 1f)
        return PointF(
            point.x + velocity.x * dt * damping,
            point.y + velocity.y * dt * damping
        )
    }

    private fun renderPredictedOverlay(snapshot: GameState.Snapshot) {
        val binding = binding ?: return
        val nowMs = SystemClock.uptimeMillis()
        val predicted = predictedPointAt(nowMs + displayLeadMs)
        val overlayBall = predicted?.let { point ->
            val mappedPoint = mapPointFillCenter(
                point.x,
                point.y,
                modelWidth,
                modelHeight,
                binding.detectionOverlay.width,
                binding.detectionOverlay.height
            )
            trailPoints.addLast(TrailPoint(mappedPoint, nowMs))
            DetectionOverlayView.BallOverlay(
                center = mappedPoint,
                detected = lastDetectionAtMs != 0L && nowMs - lastDetectionAtMs <= avgInferenceMs.toLong() + 48L,
                label = ""
            )
        } ?: trailPoints.lastOrNull()?.point?.let { point ->
            DetectionOverlayView.BallOverlay(center = point, detected = false, label = "")
        }

        pruneTrail(nowMs)
        binding.detectionOverlay.showGame(
            ball = overlayBall,
            trail = trailPoints.map { it.point },
            score = snapshot.score,
            events = mapEventsToOverlay(snapshot.recentEvents),
            netX = mapXFillCenter(
                snapshot.netX,
                modelWidth,
                modelHeight,
                binding.detectionOverlay.width,
                binding.detectionOverlay.height
            )
        )
    }

    private fun pruneTrail(nowMs: Long) {
        while (trailPoints.size > trailWindow) {
            trailPoints.removeFirst()
        }
        while (trailPoints.isNotEmpty() && nowMs - trailPoints.first().capturedAtMs > trailKeepMs) {
            trailPoints.removeFirst()
        }
    }

    private fun resetTrackingState() {
        trailPoints.clear()
        smoothedPoint = null
        smoothedVelocity = null
        smoothedAtMs = 0L
        lastDetectionAtMs = 0L
        lastAnnouncedLeftScore = 0
        lastAnnouncedRightScore = 0
        eventNetDetector?.reset()
        gameState.reset()
    }

    private fun createAnalysisExecutor(): ExecutorService {
        return Executors.newSingleThreadExecutor { runnable ->
            Thread {
                Process.setThreadPriority(Process.THREAD_PRIORITY_DISPLAY)
                runnable.run()
            }.apply {
                name = "ball-analysis"
                priority = Thread.MAX_PRIORITY
            }
        }
    }

    private fun enablePerformanceMode() {
        activity?.window?.let { window ->
            window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            window.setSustainedPerformanceMode(true)
        }
    }

    private fun disablePerformanceMode() {
        activity?.window?.let { window ->
            window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            window.setSustainedPerformanceMode(false)
        }
    }

    private fun mapPointFillCenter(
        cx: Float,
        cy: Float,
        sourceWidth: Int,
        sourceHeight: Int,
        targetWidth: Int,
        targetHeight: Int
    ): PointF {
        if (sourceWidth <= 0 || sourceHeight <= 0 || targetWidth <= 0 || targetHeight <= 0) {
            return PointF(cx, cy)
        }

        val scale = maxOf(
            targetWidth.toFloat() / sourceWidth.toFloat(),
            targetHeight.toFloat() / sourceHeight.toFloat()
        )
        val drawWidth = sourceWidth * scale
        val drawHeight = sourceHeight * scale
        val dx = (targetWidth - drawWidth) / 2f
        val dy = (targetHeight - drawHeight) / 2f

        return PointF(cx * scale + dx, cy * scale + dy)
    }

    private fun mapXFillCenter(
        x: Float,
        sourceWidth: Int,
        sourceHeight: Int,
        targetWidth: Int,
        targetHeight: Int
    ): Float {
        if (sourceWidth <= 0 || sourceHeight <= 0 || targetWidth <= 0 || targetHeight <= 0) return x
        val scale = maxOf(
            targetWidth.toFloat() / sourceWidth.toFloat(),
            targetHeight.toFloat() / sourceHeight.toFloat()
        )
        val drawWidth = sourceWidth * scale
        val dx = (targetWidth - drawWidth) / 2f
        return x * scale + dx
    }

    private fun mapXOverlayToSource(
        x: Float,
        sourceWidth: Int,
        sourceHeight: Int,
        targetWidth: Int,
        targetHeight: Int
    ): Float {
        if (sourceWidth <= 0 || sourceHeight <= 0 || targetWidth <= 0 || targetHeight <= 0) return x
        val scale = maxOf(
            targetWidth.toFloat() / sourceWidth.toFloat(),
            targetHeight.toFloat() / sourceHeight.toFloat()
        )
        val drawWidth = sourceWidth * scale
        val dx = (targetWidth - drawWidth) / 2f
        return ((x - dx) / scale).coerceIn(0f, sourceWidth.toFloat())
    }

    private fun mapYOverlayToSource(
        y: Float,
        sourceWidth: Int,
        sourceHeight: Int,
        targetWidth: Int,
        targetHeight: Int
    ): Float {
        if (sourceWidth <= 0 || sourceHeight <= 0 || targetWidth <= 0 || targetHeight <= 0) return y
        val scale = maxOf(
            targetWidth.toFloat() / sourceWidth.toFloat(),
            targetHeight.toFloat() / sourceHeight.toFloat()
        )
        val drawHeight = sourceHeight * scale
        val dy = (targetHeight - drawHeight) / 2f
        return ((y - dy) / scale).coerceIn(0f, sourceHeight.toFloat())
    }

    private fun mapEventsToOverlay(events: List<GameState.Event>): List<GameState.Event> {
        val binding = binding ?: return emptyList()
        return events.map { event ->
            val point = mapPointFillCenter(
                event.cx,
                event.cy,
                modelWidth,
                modelHeight,
                binding.detectionOverlay.width,
                binding.detectionOverlay.height
            )
            event.copy(cx = point.x, cy = point.y)
        }
    }

    override fun onDestroyView() {
        disablePerformanceMode()
        cameraProvider?.unbindAll()
        cameraProvider = null
        resetTrackingState()
        detector?.close()
        detector = null
        eventNetDetector?.close()
        eventNetDetector = null
        textToSpeech?.stop()
        textToSpeech?.shutdown()
        textToSpeech = null
        ttsReady = false
        analysisExecutor?.shutdown()
        analysisExecutor = null
        binding = null
        super.onDestroyView()
    }
}
