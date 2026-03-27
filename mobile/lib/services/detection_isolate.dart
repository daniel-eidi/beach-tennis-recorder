import 'dart:async';
import 'dart:developer' as developer;
import 'dart:isolate';
import 'dart:typed_data';

import 'package:camera/camera.dart';

import '../models/detection.dart';
import '../utils/image_converter.dart';
import 'tflite_service.dart';

/// Runs YOLOv8 TFLite inference in a separate Isolate to avoid blocking the
/// UI thread.
///
/// Communication flow:
///   Main Isolate ──(frame data)──> Detection Isolate
///   Detection Isolate ──(List<Detection>)──> Main Isolate
///
/// Frame skipping: if the isolate is still processing a previous frame when a
/// new one arrives, the new frame is dropped. This prevents a backlog of frames
/// and keeps latency low.
///
/// Implements TASK-01-06.
class IsolateDetectionService {
  Isolate? _isolate;
  SendPort? _sendPort;
  ReceivePort? _receivePort;

  final _detectionController = StreamController<DetectionResult>.broadcast();
  bool _isBusy = false;
  bool _isRunning = false;
  int _framesProcessed = 0;
  int _framesDropped = 0;

  /// Stream of detection results from the isolate.
  Stream<DetectionResult> get detections => _detectionController.stream;

  /// Whether the detection isolate is running.
  bool get isRunning => _isRunning;

  /// Whether the isolate is currently processing a frame.
  bool get isBusy => _isBusy;

  /// Total frames successfully processed.
  int get framesProcessed => _framesProcessed;

  /// Total frames dropped because the isolate was busy.
  int get framesDropped => _framesDropped;

  /// Spawns the detection isolate and initializes the TFLite model inside it.
  ///
  /// Returns true if the isolate was started and the model loaded successfully.
  /// Returns false if the model could not be loaded (detection will be
  /// unavailable but the app continues to function).
  Future<bool> start() async {
    if (_isRunning) {
      _log('warn', 'Detection isolate already running');
      return true;
    }

    try {
      _receivePort = ReceivePort();
      _framesProcessed = 0;
      _framesDropped = 0;

      // Spawn the isolate.
      _isolate = await Isolate.spawn(
        _isolateEntryPoint,
        _receivePort!.sendPort,
      );

      // Wait for the isolate's SendPort.
      final completer = Completer<bool>();

      _receivePort!.listen((message) {
        if (message is SendPort) {
          _sendPort = message;
          // Signal the isolate to load the model.
          _sendPort!.send('load_model');
        } else if (message is _ModelLoadedMessage) {
          _isRunning = true;
          _log('info', 'Detection isolate started, model loaded: '
              '${message.success}');
          if (!completer.isCompleted) {
            completer.complete(message.success);
          }
        } else if (message is _DetectionResultMessage) {
          _isBusy = false;
          _framesProcessed++;
          _detectionController.add(DetectionResult(
            detections: message.detections,
            inferenceTimeMs: message.inferenceTimeMs,
            frameTimestamp: message.frameTimestamp,
          ));
        } else if (message is _ErrorMessage) {
          _isBusy = false;
          _log('error', 'Isolate error: ${message.error}');
        }
      });

      return await completer.future.timeout(
        const Duration(seconds: 10),
        onTimeout: () {
          _log('error', 'Timeout waiting for detection isolate to start');
          return false;
        },
      );
    } catch (e) {
      _log('error', 'Failed to start detection isolate: $e');
      return false;
    }
  }

  /// Sends a camera frame to the detection isolate for processing.
  ///
  /// If the isolate is still processing a previous frame, this frame is
  /// dropped (not queued). This is critical for maintaining low latency.
  void processFrame(CameraImage image) {
    if (!_isRunning || _sendPort == null) return;

    // Frame skipping: drop if isolate is busy.
    if (_isBusy) {
      _framesDropped++;
      return;
    }

    // Convert CameraImage to RGB bytes on the main isolate.
    // This is lightweight compared to inference and avoids sending raw
    // platform-specific image data across isolate boundaries.
    final rgbBytes = ImageConverter.convertCameraImageToRgb(image);
    if (rgbBytes == null) return;

    final (width, height) = ImageConverter.getImageDimensions(image);

    _isBusy = true;
    _sendPort!.send(_FrameMessage(
      rgbBytes: rgbBytes,
      width: width,
      height: height,
      timestamp: DateTime.now(),
    ));
  }

  /// Stops the detection isolate and cleans up resources.
  Future<void> stop() async {
    if (!_isRunning) return;

    _isRunning = false;
    _isBusy = false;

    _sendPort?.send('dispose');

    // Give the isolate a moment to clean up, then kill it.
    await Future.delayed(const Duration(milliseconds: 100));
    _isolate?.kill(priority: Isolate.beforeNextEvent);
    _isolate = null;
    _sendPort = null;
    _receivePort?.close();
    _receivePort = null;

    _log('info', 'Detection isolate stopped '
        '(processed=$_framesProcessed dropped=$_framesDropped)');
  }

  /// Disposes the service and releases all resources.
  void dispose() {
    stop();
    _detectionController.close();
  }

  static void _log(String level, String message) {
    developer.log(
      message,
      name: 'IsolateDetectionService',
      level: level == 'error' ? 1000 : (level == 'warn' ? 900 : 800),
    );
  }
}

// --- Isolate entry point and internal messages ---

/// Entry point for the detection isolate.
///
/// Creates its own TFLiteService instance (interpreters cannot be shared
/// across isolates) and processes frames as they arrive.
void _isolateEntryPoint(SendPort mainSendPort) {
  final receivePort = ReceivePort();
  mainSendPort.send(receivePort.sendPort);

  TFLiteService? tfliteService;

  receivePort.listen((message) async {
    if (message == 'load_model') {
      tfliteService = TFLiteService();
      final success = await tfliteService!.loadModel();
      mainSendPort.send(_ModelLoadedMessage(success: success));
    } else if (message is _FrameMessage) {
      if (tfliteService == null || !tfliteService!.isLoaded) {
        mainSendPort.send(_DetectionResultMessage(
          detections: [],
          inferenceTimeMs: 0,
          frameTimestamp: message.timestamp,
        ));
        return;
      }

      try {
        final stopwatch = Stopwatch()..start();

        // Preprocess: resize and normalize to float32.
        final inputData = TFLiteService.preprocessFrame(
          message.rgbBytes,
          width: message.width,
          height: message.height,
        );

        // Run inference.
        final detections = tfliteService!.runInference(inputData);

        stopwatch.stop();

        mainSendPort.send(_DetectionResultMessage(
          detections: detections,
          inferenceTimeMs: stopwatch.elapsedMilliseconds,
          frameTimestamp: message.timestamp,
        ));
      } catch (e) {
        mainSendPort.send(_ErrorMessage(error: e.toString()));
      }
    } else if (message == 'dispose') {
      tfliteService?.dispose();
      tfliteService = null;
      receivePort.close();
    }
  });
}

// --- Message classes for isolate communication ---

/// Sent from main isolate to detection isolate with frame data.
class _FrameMessage {
  final Uint8List rgbBytes;
  final int width;
  final int height;
  final DateTime timestamp;

  const _FrameMessage({
    required this.rgbBytes,
    required this.width,
    required this.height,
    required this.timestamp,
  });
}

/// Sent from detection isolate when model loading completes.
class _ModelLoadedMessage {
  final bool success;
  const _ModelLoadedMessage({required this.success});
}

/// Sent from detection isolate with inference results.
class _DetectionResultMessage {
  final List<Detection> detections;
  final int inferenceTimeMs;
  final DateTime frameTimestamp;

  const _DetectionResultMessage({
    required this.detections,
    required this.inferenceTimeMs,
    required this.frameTimestamp,
  });
}

/// Sent from detection isolate when an error occurs.
class _ErrorMessage {
  final String error;
  const _ErrorMessage({required this.error});
}

/// Public result container exposed via the detections stream.
class DetectionResult {
  /// Detections found in this frame.
  final List<Detection> detections;

  /// Time taken for inference in milliseconds.
  final int inferenceTimeMs;

  /// Timestamp of the original camera frame.
  final DateTime frameTimestamp;

  const DetectionResult({
    required this.detections,
    required this.inferenceTimeMs,
    required this.frameTimestamp,
  });
}
