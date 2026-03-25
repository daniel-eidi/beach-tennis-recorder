import 'dart:developer' as developer;
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:tflite_flutter/tflite_flutter.dart';

import '../models/detection.dart';

/// Service for running YOLOv8 TFLite inference on camera frames.
///
/// Loads the ball_detector.tflite model from assets and provides methods to
/// preprocess camera frames, run inference, and post-process results into
/// [Detection] objects.
///
/// This class is designed to be used inside a separate Isolate — it does NOT
/// share state with the main UI isolate. Each Isolate that needs inference
/// must create its own [TFLiteService] instance.
///
/// Implements TASK-01-05.
class TFLiteService {
  static const String _modelPath = 'assets/models/ball_detector.tflite';

  /// Model input dimensions (YOLOv8 standard).
  static const int inputSize = 640;

  /// Number of raw detections from YOLOv8 output grid.
  static const int numDetections = 25200;

  /// Values per detection: x, y, w, h, confidence, classId.
  static const int valuesPerDetection = 6;

  /// Minimum confidence to keep a detection before NMS.
  static const double defaultConfidenceThreshold = 0.45;

  /// IoU threshold for non-max suppression.
  static const double nmsIouThreshold = 0.5;

  Interpreter? _interpreter;
  bool _isLoaded = false;

  /// Whether the model has been successfully loaded.
  bool get isLoaded => _isLoaded;

  /// Loads the TFLite model from the assets bundle.
  ///
  /// Returns true if the model was loaded successfully, false otherwise.
  /// The app should continue working without the model for development
  /// (graceful degradation).
  Future<bool> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(_modelPath);
      _isLoaded = true;
      _log('info', 'TFLite model loaded: $_modelPath');

      // Log input/output tensor info for debugging.
      final inputTensors = _interpreter!.getInputTensors();
      final outputTensors = _interpreter!.getOutputTensors();
      _log('info', 'Input shape: ${inputTensors.first.shape}');
      _log('info', 'Output shape: ${outputTensors.first.shape}');

      return true;
    } catch (e) {
      _isLoaded = false;
      _log('warn', 'Failed to load TFLite model (app will continue '
          'without detection): $e');
      return false;
    }
  }

  /// Runs inference on a preprocessed float32 input tensor.
  ///
  /// [inputData] must be a Float32List of shape [1, 640, 640, 3] (flattened),
  /// with values normalized to [0.0, 1.0].
  ///
  /// Returns a list of [Detection] objects after applying confidence
  /// filtering and non-max suppression.
  List<Detection> runInference(
    Float32List inputData, {
    double confidenceThreshold = defaultConfidenceThreshold,
  }) {
    if (!_isLoaded || _interpreter == null) {
      _log('warn', 'Cannot run inference: model not loaded');
      return [];
    }

    try {
      // Reshape input to [1, 640, 640, 3].
      final input = inputData.reshape([1, inputSize, inputSize, 3]);

      // Prepare output buffer: [1, 25200, 6].
      final output = List.generate(
        1,
        (_) => List.generate(
          numDetections,
          (_) => List.filled(valuesPerDetection, 0.0),
        ),
      );

      // Run inference.
      _interpreter!.run(input, output);

      // Post-process: filter by confidence and apply NMS.
      return _postProcess(output[0], confidenceThreshold);
    } catch (e) {
      _log('error', 'Inference failed: $e');
      return [];
    }
  }

  /// Preprocesses raw RGB bytes into a Float32List ready for inference.
  ///
  /// [rgbBytes] is a flat Uint8List of size [width * height * 3] in RGB order.
  /// The image is resized to 640x640 and normalized to [0.0, 1.0].
  ///
  /// This method is static so it can be called from an Isolate without
  /// needing a TFLiteService instance for preprocessing only.
  static Float32List preprocessFrame(
    Uint8List rgbBytes, {
    required int width,
    required int height,
  }) {
    final inputPixels = inputSize * inputSize * 3;
    final float32 = Float32List(inputPixels);

    final xRatio = width / inputSize;
    final yRatio = height / inputSize;

    int outIndex = 0;
    for (int y = 0; y < inputSize; y++) {
      final srcY = (y * yRatio).toInt().clamp(0, height - 1);
      for (int x = 0; x < inputSize; x++) {
        final srcX = (x * xRatio).toInt().clamp(0, width - 1);
        final srcIndex = (srcY * width + srcX) * 3;

        // Normalize to [0.0, 1.0].
        float32[outIndex++] = rgbBytes[srcIndex] / 255.0;
        float32[outIndex++] = rgbBytes[srcIndex + 1] / 255.0;
        float32[outIndex++] = rgbBytes[srcIndex + 2] / 255.0;
      }
    }

    return float32;
  }

  /// Post-processes raw model output into Detection objects.
  ///
  /// Applies confidence threshold filtering and non-max suppression.
  List<Detection> _postProcess(
    List<List<double>> rawDetections,
    double confidenceThreshold,
  ) {
    // Step 1: Filter by confidence.
    final candidates = <Detection>[];
    for (final raw in rawDetections) {
      final confidence = raw[4];
      if (confidence >= confidenceThreshold) {
        candidates.add(Detection.fromModelOutput(raw));
      }
    }

    if (candidates.isEmpty) return [];

    // Step 2: Non-max suppression per class.
    final result = <Detection>[];
    final classIds = candidates.map((d) => d.classId).toSet();

    for (final classId in classIds) {
      final classDetections = candidates
          .where((d) => d.classId == classId)
          .toList()
        ..sort((a, b) => b.confidence.compareTo(a.confidence));

      final kept = <Detection>[];
      final suppressed = List.filled(classDetections.length, false);

      for (int i = 0; i < classDetections.length; i++) {
        if (suppressed[i]) continue;
        kept.add(classDetections[i]);

        for (int j = i + 1; j < classDetections.length; j++) {
          if (suppressed[j]) continue;
          if (_computeIoU(classDetections[i], classDetections[j]) >
              nmsIouThreshold) {
            suppressed[j] = true;
          }
        }
      }

      result.addAll(kept);
    }

    return result;
  }

  /// Computes Intersection over Union (IoU) between two detections.
  static double _computeIoU(Detection a, Detection b) {
    final xOverlapStart = math.max(a.left, b.left);
    final yOverlapStart = math.max(a.top, b.top);
    final xOverlapEnd = math.min(a.right, b.right);
    final yOverlapEnd = math.min(a.bottom, b.bottom);

    if (xOverlapEnd <= xOverlapStart || yOverlapEnd <= yOverlapStart) {
      return 0.0;
    }

    final intersectionArea =
        (xOverlapEnd - xOverlapStart) * (yOverlapEnd - yOverlapStart);
    final unionArea = a.area + b.area - intersectionArea;

    if (unionArea <= 0) return 0.0;
    return intersectionArea / unionArea;
  }

  /// Releases the interpreter resources.
  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _isLoaded = false;
    _log('info', 'TFLite interpreter disposed');
  }

  static void _log(String level, String message) {
    developer.log(
      message,
      name: 'TFLiteService',
      level: level == 'error' ? 1000 : (level == 'warn' ? 900 : 800),
    );
  }
}
