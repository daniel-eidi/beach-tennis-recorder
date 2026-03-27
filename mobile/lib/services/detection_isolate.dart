import 'dart:async';
import 'dart:developer' as developer;

import '../models/detection.dart';

/// Result from the detection isolate.
class DetectionResult {
  final List<Detection> detections;
  final int inferenceTimeMs;
  final DateTime frameTimestamp;

  DetectionResult({
    required this.detections,
    required this.inferenceTimeMs,
    DateTime? frameTimestamp,
  }) : frameTimestamp = frameTimestamp ?? DateTime.now();
}

/// Detection service stub — TFLite disabled for MVP.
///
/// Returns empty detections. The real implementation runs YOLOv8 TFLite
/// inference in a separate Isolate to avoid blocking the UI thread.
///
/// TODO: Re-enable when tflite_flutter is working on iOS.
class IsolateDetectionService {
  final _detectionsController = StreamController<DetectionResult>.broadcast();
  bool _isRunning = false;

  Stream<DetectionResult> get detections => _detectionsController.stream;
  bool get isRunning => _isRunning;

  /// Returns false — model not available for MVP.
  Future<bool> start() async {
    developer.log('Detection isolate disabled for MVP', name: 'DetectionIsolate');
    _isRunning = false;
    return false;
  }

  void processFrame(dynamic image) {
    // No-op — detection disabled
  }

  Future<void> stop() async {
    _isRunning = false;
  }

  void dispose() {
    _detectionsController.close();
  }
}
