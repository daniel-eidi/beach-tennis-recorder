import 'dart:developer' as developer;

import '../models/detection.dart';

/// TFLite inference service — currently a stub.
///
/// The tflite_flutter package is disabled for MVP due to CocoaPods
/// compatibility issues on iOS. This stub returns empty detections
/// so the rest of the app works without the model.
///
/// TODO: Re-enable when tflite_flutter CocoaPods issues are resolved.
class TFLiteService {
  bool _isLoaded = false;

  bool get isLoaded => _isLoaded;

  Future<bool> loadModel() async {
    _log('info', 'TFLite model loading disabled for MVP');
    _isLoaded = false;
    return false;
  }

  List<Detection> runInference(dynamic inputData, int imageWidth, int imageHeight) {
    return [];
  }

  void dispose() {
    _isLoaded = false;
  }

  void _log(String level, String message) {
    developer.log(
      message,
      name: 'TFLiteService',
      level: level == 'error' ? 1000 : 800,
    );
  }
}
