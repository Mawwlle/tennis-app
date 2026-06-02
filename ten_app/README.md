# BallRealtime (`ten_app`)

Android-приложение для realtime-анализа настольного тенниса: камера видит стол, ONNX-модели находят мяч и отскоки, а внутренняя машина состояний считает очки, показывает оверлей поверх превью и озвучивает изменение счета.

Приложение является мобильной частью общего репозитория `tennis-app`. Python-код в корне репозитория используется для подготовки данных, обучения и экспорта моделей, а `ten_app` отвечает за запуск этих моделей на Android-устройстве.

## Содержание

- [Что умеет приложение](#что-умеет-приложение)
- [Как работает пайплайн](#как-работает-пайплайн)
- [Требования](#требования)
- [Быстрый старт](#быстрый-старт)
- [Модели и assets](#модели-и-assets)
- [Сборка и запуск](#сборка-и-запуск)
- [Первый запуск и калибровка](#первый-запуск-и-калибровка)
- [Структура проекта](#структура-проекта)
- [Ключевые классы](#ключевые-классы)
- [Логика счета](#логика-счета)
- [Настройка производительности](#настройка-производительности)
- [Экспорт моделей из корня репозитория](#экспорт-моделей-из-корня-репозитория)
- [Troubleshooting](#troubleshooting)
- [Полезные команды](#полезные-команды)

## Что умеет приложение

`BallRealtime` запускается в landscape-режиме и показывает полноэкранное превью с задней камеры. Поверх видео рисуются:

- текущий счет `LEFT : RIGHT`;
- вертикальная линия сетки;
- текущая позиция мяча;
- короткий trail движения мяча;
- маркеры событий: удар, отскок, попадание в сетку, промах;
- подпись последнего события или начисленного очка.

Также приложение использует Android Text-to-Speech на русском языке:

- после калибровки сообщает, кто подает первым;
- после изменения счета озвучивает, кому начислено очко;
- после очка озвучивает следующего подающего.

## Как работает пайплайн

Общий поток данных:

```text
CameraX frame
    |
    v
YUV_420_888 preprocessing
    |
    v
YOLO ONNX: yolo_det.onnx
    |
    v
ball center: cx, cy, confidence
    |
    +----------------------+
    |                      |
    v                      v
EventNet ONNX          trajectory heuristics
eventnet_best.onnx     hit / net / miss / fallback bounce
    |                      |
    +----------+-----------+
               |
               v
GameState state machine
    |
    v
score + events + server
    |
    v
DetectionOverlayView + TextToSpeech
```

Важная деталь: Android-приложение не запускает тяжелый сервер и не вызывает Python. Все происходит локально в приложении:

- кадры поступают через `CameraX ImageAnalysis`;
- `YoloQnnDetector` запускает ONNX Runtime с QNN Execution Provider;
- `EventNetBounceDetector` запускает небольшую ONNX-модель для детекции отскоков по окну из heatmap-представлений траектории;
- `GameState` объединяет события и начисляет очки.

## Требования

### Для сборки

- Android Studio с Android Gradle Plugin 8.5.2 или совместимой версией.
- JDK 17.
- Android SDK:
  - `compileSdk 35`;
  - `targetSdk 35`;
  - `minSdk 28`.
- Gradle Wrapper уже лежит в проекте: `./gradlew`.

### Для запуска на устройстве

Приложение рассчитано на реальное Android-устройство, а не на эмулятор.

Минимально требуется:

- Android 9+ (`minSdk 28`);
- камера;
- ABI `arm64-v8a`;
- Qualcomm QNN / HTP окружение, доступное через `onnxruntime-android-qnn`;
- наличие системной native library `libcdsprpc.so`.

В `AndroidManifest.xml` библиотека `libcdsprpc.so` объявлена как обязательная:

```xml
<uses-native-library
    android:name="libcdsprpc.so"
    android:required="true" />
```

Поэтому на устройствах без нужного Qualcomm DSP/QNN окружения приложение может не установиться или не сможет инициализировать детектор.

## Быстрый старт

Из корня репозитория:

```bash
cd ten_app
./gradlew :app:assembleDebug
```

Установить debug-сборку на подключенное устройство:

```bash
./gradlew :app:installDebug
```

Или открыть директорию `ten_app` в Android Studio и запустить конфигурацию `app`.

## Модели и assets

Приложение ожидает ONNX-модели в:

```text
ten_app/app/src/main/assets/
```

На текущий момент используются два файла:

| Файл | Примерный размер | Назначение | Где используется |
|---|---:|---|---|
| `yolo_det.onnx` | 12 MB | Детекция мяча на кадре | `YoloQnnDetector` |
| `eventnet_best.onnx` | 175 KB | Детекция отскоков по окну траектории | `EventNetBounceDetector` |

Если assets отсутствуют, приложение соберется, но при запуске детекторы не смогут инициализироваться. В UI появится Toast вида:

```text
ONNX init error: ...
EventNet init error: ...
```

Скопировать модели из общей папки `weights` можно так:

```bash
cd ten_app
mkdir -p app/src/main/assets
cp ../weights/yolo_det.onnx app/src/main/assets/yolo_det.onnx
cp ../weights/eventnet_best.onnx app/src/main/assets/eventnet_best.onnx
```

В `app/build.gradle` для ONNX и связанных файлов включено `noCompress`:

```gradle
androidResources {
    noCompress += ['onnx', 'data', 'json']
}
```

Это важно: Android не должен сжимать модели внутри APK, иначе чтение и копирование больших assets может стать медленнее или сломать ожидаемый доступ к файлам.

## Сборка и запуск

### Debug APK

```bash
cd ten_app
./gradlew :app:assembleDebug
```

APK появится здесь:

```text
ten_app/app/build/outputs/apk/debug/app-debug.apk
```

### Установка на устройство

```bash
cd ten_app
./gradlew :app:installDebug
```

Проверить, что устройство видно через ADB:

```bash
adb devices
```

### Release APK

```bash
cd ten_app
./gradlew :app:assembleRelease
```

В текущей конфигурации `release` не включает минификацию:

```gradle
release {
    minifyEnabled false
}
```

Если нужен публикационный APK/AAB, отдельно настрой signing config.

## Первый запуск и калибровка

При старте приложение:

1. включает landscape-режим;
2. запрашивает разрешение камеры;
3. инициализирует YOLO и EventNet;
4. открывает заднюю камеру;
5. показывает экран калибровки.

### Шаг 1. Калибровка стола

На экране появляется оранжевый четырехугольник. Нужно перетащить его углы так, чтобы он примерно совпал со столом в кадре.

После этого нажми:

```text
Стол готов
```

Калибровка стола задает:

- левую и правую границу стола;
- верхнюю и нижнюю границу стола;
- примерную позицию сетки как середину между левой и правой границей.

### Шаг 2. Положение игроков

На следующем шаге нужно нажать на экран два раза:

1. примерная позиция левого игрока;
2. примерная позиция правого игрока.

После выбора двух точек кнопка меняется на:

```text
Игроки выбраны
```

После нажатия приложение:

- сбрасывает трекинг;
- применяет калибровку стола;
- скрывает калибровочный оверлей;
- начинает анализ кадров;
- озвучивает начальную подачу левого игрока.

### Что считается хорошим кадром

Для стабильной работы желательно:

- стол занимает заметную часть кадра;
- камера не трясется;
- сетка примерно вертикальна на экране;
- мяч не слишком часто перекрывается игроками;
- освещение достаточно яркое;
- камера находится сбоку или под углом, где левая/правая половина стола различимы.

## Структура проекта

```text
ten_app/
├── README.md
├── build.gradle
├── settings.gradle
├── gradlew
├── gradlew.bat
├── gradle/
│   └── wrapper/
│       ├── gradle-wrapper.jar
│       └── gradle-wrapper.properties
└── app/
    ├── build.gradle
    ├── proguard-rules.pro
    └── src/main/
        ├── AndroidManifest.xml
        ├── assets/
        │   ├── yolo_det.onnx
        │   └── eventnet_best.onnx
        ├── java/com/example/ballrealtime/
        │   ├── MainActivity.kt
        │   ├── GameFragment.kt
        │   ├── YoloQnnDetector.kt
        │   ├── EventNetBounceDetector.kt
        │   ├── GameState.kt
        │   ├── DetectionOverlayView.kt
        │   └── CalibrationOverlayView.kt
        └── res/
            ├── layout/
            │   ├── activity_main.xml
            │   └── fragment_game.xml
            └── values/
                ├── colors.xml
                ├── strings.xml
                └── themes.xml
```

## Ключевые классы

### `MainActivity.kt`

Минимальная activity, которая создает `GameFragment`:

- использует ViewBinding;
- устанавливает `activity_main.xml`;
- при первом создании кладет `GameFragment` в контейнер.

### `GameFragment.kt`

Главный координатор приложения.

Отвечает за:

- разрешение камеры;
- запуск CameraX preview и analysis pipeline;
- инициализацию моделей;
- калибровку стола и игроков;
- передачу кадров в YOLO;
- передачу траектории в EventNet;
- обновление `GameState`;
- отрисовку overlay;
- Text-to-Speech;
- включение performance mode и keep-screen-on.

Ключевые параметры:

```kotlin
private val modelAssetName = "yolo_det.onnx"
private val eventNetAssetName = "eventnet_best.onnx"
private val analysisSize = Size(320, 180)
private val trailWindow = 18
private val trailKeepMs = 1200L
```

`analysisSize = 320x180` означает, что CameraX отдает в анализ компактный кадр, а модель внутри `YoloQnnDetector` приводит его к входу `640x640`.

### `YoloQnnDetector.kt`

Обертка над ONNX Runtime для детекции мяча.

Что делает:

- проверяет наличие QNN Execution Provider;
- копирует `yolo_det.onnx` из assets во внутреннюю папку приложения;
- создает `OrtSession` с QNN options;
- конвертирует `ImageProxy` из `YUV_420_888` в RGB tensor;
- приводит кадр к letterbox-входу `640x640`;
- запускает ONNX inference;
- выполняет postprocess YOLO-выхода;
- возвращает центр мяча, bounding box и confidence.

QNN options:

```kotlin
put("backend_path", File(nativeLibraryDir, "libQnnHtp.so").absolutePath)
put("htp_performance_mode", "burst")
put("enable_htp_fp16_precision", "1")
put("htp_graph_finalization_optimization_mode", "3")
put("profiling_level", "basic")
```

Порог confidence:

```kotlin
private val confidenceThreshold = 0.2f
```

IoU-порог для NMS:

```kotlin
private val iouThreshold = 0.25f
```

### `EventNetBounceDetector.kt`

Легкая ONNX-модель для отскоков.

Она не принимает видео напрямую. Вместо этого приложение накапливает окно из координат мяча и превращает каждую координату в маленькую heatmap:

```text
window: 15 frames
heatmap: 36 x 64
sigma: 2.0
input tensor: 1 x 15 x 36 x 64
```

Если `sigmoid(logit)` выше порога, EventNet возвращает событие `BOUNCE`.

Ключевые параметры:

```kotlin
private const val EVENTNET_WINDOW = 15
private const val EVENTNET_THRESHOLD = 0.35f
private const val EVENTNET_DEDUP_FRAMES = 8
```

Сторона отскока определяется относительно текущей линии сетки:

```kotlin
val side = if (cx < netX) GameState.Side.LEFT else GameState.Side.RIGHT
```

### `GameState.kt`

Мозг счета и событий.

Хранит:

- текущий счет;
- подающего;
- историю детекций;
- историю событий;
- фазу розыгрыша;
- границы стола;
- положение сетки.

Поддерживаемые события:

```kotlin
enum class EventKind { HIT, BOUNCE, NET, MISS }
```

Поддерживаемые стороны:

```kotlin
enum class Side { LEFT, RIGHT }
```

Фазы розыгрыша:

```kotlin
private enum class Phase {
    IDLE,
    AWAIT_SERVE_BOUNCE,
    AWAIT_BOUNCE,
    AWAIT_RETURN
}
```

Подача меняется по правилам настольного тенниса:

- до счета 10:10 - каждые 2 очка;
- после 10:10 - после каждого очка.

Это задается константами:

```kotlin
private const val SERVES_PER_TURN = 2
private const val DEUCE_SCORE = 10
```

### `DetectionOverlayView.kt`

Canvas-view для отрисовки поверх камеры.

Рисует:

- верхнюю панель счета;
- подписи `LEFT` и `RIGHT`;
- текущую линию сетки;
- trail мяча;
- текущий мяч;
- события на столе;
- подпись последнего события внизу.

Цвета событий:

| Событие | Визуальный смысл |
|---|---|
| `HIT` | обводка удара |
| `BOUNCE` | желтый маркер отскока |
| `NET` | крест у сетки |
| `MISS` | маркер промаха |

### `CalibrationOverlayView.kt`

Интерактивный оверлей калибровки.

Режимы:

```kotlin
enum class Mode { TABLE, PEOPLE, HIDDEN }
```

В режиме `TABLE` пользователь двигает 4 угла полигона стола.

В режиме `PEOPLE` пользователь отмечает две точки игроков. Если нажать третий раз, список точек очищается и выбор начинается заново.

## Логика счета

`GameState` работает как state machine.

Упрощенно:

```text
IDLE
  |
  | HIT by server
  v
AWAIT_SERVE_BOUNCE
  |
  | first bounce / serve continues
  v
AWAIT_BOUNCE
  |
  | valid bounce on target side
  v
AWAIT_RETURN
  |
  | return hit
  v
AWAIT_BOUNCE
```

Очко начисляется, когда происходит одно из условий:

- мяч попал в сетку после удара;
- мяч улетел в промах;
- после удара слишком долго нет ожидаемого отскока;
- после отскока слишком долго нет ответного удара;
- мяч потерян на достаточное количество кадров в активной фазе розыгрыша;
- обнаружена неверная последовательность событий.

Ключевые timeout-параметры:

```kotlin
private const val MAX_HIT_TO_BOUNCE_FRAMES = 110
private const val MAX_BOUNCE_TO_HIT_FRAMES = 140
private const val BALL_LOST_FRAMES = 60
```

История ограничена, чтобы приложение не держало бесконечные буферы:

```kotlin
private const val MAX_HISTORY = 240
private const val MAX_EVENTS = 32
```

## Настройка производительности

### CameraX

В `GameFragment` анализ кадров настроен так:

```kotlin
ImageAnalysis.Builder()
    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
    .setTargetResolution(Size(320, 180))
    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
```

`STRATEGY_KEEP_ONLY_LATEST` важен для realtime: если inference не успевает обработать все кадры, приложение берет свежий кадр и не накапливает очередь.

FPS пробуется в таком порядке:

```kotlin
listOf(Range(30, 40), Range(30, 30), null)
```

Если устройство не принимает первый диапазон, код пробует следующий.

### Поток анализа

Для анализа создается отдельный single-thread executor:

```kotlin
Thread {
    Process.setThreadPriority(Process.THREAD_PRIORITY_DISPLAY)
    runnable.run()
}.apply {
    name = "ball-analysis"
    priority = Thread.MAX_PRIORITY
}
```

### Режим производительности

При старте фрагмента:

```kotlin
window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
window.setSustainedPerformanceMode(true)
```

При уничтожении view режим выключается.

### QNN profiling

`YoloQnnDetector` включает basic profiling:

```kotlin
put("profiling_file_path", File(context.filesDir, "yolo_det_qnn_profile.csv").absolutePath)
```

Файл профиля находится во внутренней директории приложения. Его можно вытащить через `adb`, если известен package name:

```bash
adb shell run-as com.example.ballrealtime ls files
adb shell run-as com.example.ballrealtime cat files/yolo_det_qnn_profile.csv
```

## Экспорт моделей из корня репозитория

Android-приложение потребляет уже экспортированные ONNX-файлы. Обучение и экспорт находятся на уровень выше, в корне `tennis-app`.

### Python-окружение

В корне репозитория используется `uv`:

```bash
cd ..
uv sync
```

### Экспорт ONNX

Для общего пайплайна в корне есть скрипт:

```bash
uv run python export_onnx.py
```

После экспорта проверь папку:

```bash
ls -lh weights/
```

Для Android нужны именно:

```text
weights/yolo_det.onnx
weights/eventnet_best.onnx
```

Затем скопируй их в assets:

```bash
cd ten_app
cp ../weights/yolo_det.onnx app/src/main/assets/yolo_det.onnx
cp ../weights/eventnet_best.onnx app/src/main/assets/eventnet_best.onnx
```

## Troubleshooting

### Warning про `compileSdk = 35`

При сборке Gradle может показать предупреждение:

```text
This Android Gradle plugin (8.5.2) was tested up to compileSdk = 34.
```

Это не ошибка сборки. Текущая debug-сборка проходит с `compileSdk 35`, но Android Gradle Plugin 8.5.2 официально тестировался до SDK 34.

Варианты:

- оставить как есть, если сборка проходит;
- обновить Android Gradle Plugin в `build.gradle`;
- подавить предупреждение в `gradle.properties`:

```properties
android.suppressUnsupportedCompileSdk=35
```

### `QNNExecutionProvider is not available`

Причина: ONNX Runtime в приложении не видит QNN provider.

Проверь:

- используется dependency `com.microsoft.onnxruntime:onnxruntime-android-qnn:1.24.3`;
- приложение запускается на реальном arm64 Qualcomm-устройстве;
- устройство поддерживает нужное QNN/HTP окружение;
- APK не был собран с другим ABI;
- в `defaultConfig.ndk` остался `abiFilters 'arm64-v8a'`.

### `libcdsprpc.so` missing

Причина: на устройстве нет обязательной native library для Qualcomm DSP RPC.

Так как в manifest стоит `android:required="true"`, система может не дать установить приложение на неподходящее устройство.

Для разработки на неподдерживаемом устройстве можно временно сделать library необязательной, но тогда QNN inference все равно, скорее всего, не заработает:

```xml
android:required="false"
```

### `ONNX init error`

Частые причины:

- нет `app/src/main/assets/yolo_det.onnx`;
- файл поврежден или был скопирован не полностью;
- модель имеет несовместимый формат входов/выходов;
- QNN provider не смог скомпилировать граф;
- не хватает памяти для инициализации.

Что проверить:

```bash
ls -lh app/src/main/assets/
```

Ожидается:

```text
yolo_det.onnx
eventnet_best.onnx
```

### `EventNet init error`

Частые причины:

- нет `eventnet_best.onnx`;
- модель экспортирована с другим именем входа или другой формой;
- модель требует opset/operator, которого нет в текущем ONNX Runtime.

Текущий Kotlin-код ожидает окно:

```text
1 x 15 x 36 x 64
```

### Камера запустилась, но анализ не начинается

Анализ кадров начинается только после завершения калибровки. Пока `calibrationStep != READY`, `analyzeFrame` сразу закрывает кадр и ничего не считает.

Проверь, что:

- нажата кнопка `Стол готов`;
- выбраны две позиции игроков;
- нажата кнопка `Игроки выбраны`.

### Мяч рисуется не там, где находится в превью

Возможные причины:

- калибровка стола сделана неточно;
- камера повернута или устройство держится не в ожидаемой ориентации;
- preview использует `fillCenter`, поэтому часть изображения может быть обрезана;
- координаты модели мапятся из `640x640` в размеры overlay с учетом fill-center scale.

Ключевые функции маппинга находятся в `GameFragment`:

```kotlin
mapPointFillCenter(...)
mapXFillCenter(...)
mapXOverlayToSource(...)
mapYOverlayToSource(...)
```

### Счет начисляется слишком рано или слишком поздно

Смотри константы в `GameState.kt`:

```kotlin
MAX_HIT_TO_BOUNCE_FRAMES
MAX_BOUNCE_TO_HIT_FRAMES
BALL_LOST_FRAMES
BOUNCE_MIN_CURVATURE
HIT_MIN_X_SPEED
NET_NEAR_RATIO
```

Также качество счета зависит от:

- точности `yolo_det.onnx`;
- стабильности FPS;
- точности калибровки сетки;
- количества пропусков мяча;
- угла камеры.

### Text-to-Speech молчит

Проверь:

- на устройстве установлен TTS engine;
- русский язык доступен для TTS;
- звук не выключен;
- калибровка завершена.

Код TTS находится в `GameFragment.setupTextToSpeech()`.

## Полезные команды

Собрать debug APK:

```bash
cd ten_app
./gradlew :app:assembleDebug
```

Установить debug APK:

```bash
cd ten_app
./gradlew :app:installDebug
```

Проверить assets:

```bash
cd ten_app
ls -lh app/src/main/assets/
```

Скопировать модели из `weights`:

```bash
cd ten_app
mkdir -p app/src/main/assets
cp ../weights/yolo_det.onnx app/src/main/assets/yolo_det.onnx
cp ../weights/eventnet_best.onnx app/src/main/assets/eventnet_best.onnx
```

Посмотреть подключенные устройства:

```bash
adb devices
```

Почитать logcat только по package name:

```bash
adb logcat | grep com.example.ballrealtime
```

Очистить build outputs:

```bash
cd ten_app
./gradlew clean
```

## Связь с корневым проектом

Корневой `tennis-app` содержит ML-пайплайн:

- подготовка датасетов;
- обучение TrackNet/EventNet/YOLO;
- экспорт ONNX;
- офлайн-инференс по видео;
- webapp для разметки.

`ten_app` - это Android runtime для уже подготовленных моделей. Если меняешь архитектуру модели или формат выхода в Python-коде, почти наверняка нужно синхронно обновить Kotlin postprocess:

- для YOLO: `YoloQnnDetector.extractYoloOutput()` и `postprocessYolo()`;
- для EventNet: `EventNetBounceDetector.runWindow()` и `extractLogits()`;
- для логики счета: `GameState.detectNewEvents()` и state machine.

## Текущие ограничения

- Приложение ориентировано на Qualcomm/QNN и не имеет CPU fallback для YOLO-детектора в UI-пайплайне.
- Эмулятор Android не является целевой средой.
- Калибровка пока ручная.
- Позиции игроков собираются при калибровке, но основная текущая логика счета опирается прежде всего на траекторию мяча, сетку и границы стола.
- Счет является эвристическим и зависит от качества детекции мяча.
- Приложение ожидает landscape-сценарий.

## Мини-чеклист перед демо

Перед показом приложения проверь:

- `yolo_det.onnx` лежит в `app/src/main/assets/`;
- `eventnet_best.onnx` лежит в `app/src/main/assets/`;
- устройство реальное, arm64, с Qualcomm QNN/HTP;
- камера разрешена;
- стол хорошо освещен;
- калибровочный полигон совпадает со столом;
- выбраны две позиции игроков;
- звук включен, если нужна озвучка счета.
