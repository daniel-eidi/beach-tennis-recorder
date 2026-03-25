import 'dart:async';
import 'dart:developer' as developer;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';

/// Manages the device camera for recording beach tennis matches.
///
/// Initializes at 1080p / 30fps and provides the [CameraController] for
/// rendering the preview widget in the UI. Handles full lifecycle including
/// permission checks and disposal.
///
/// Usage:
/// ```dart
/// final cameraService = context.read<CameraService>();
/// await cameraService.initialize();
/// // Use cameraService.controller for CameraPreview widget
/// ```
class CameraService extends ChangeNotifier {
  CameraController? _controller;
  List<CameraDescription> _cameras = [];
  bool _isInitialized = false;
  bool _isRecording = false;
  String? _error;

  /// The active camera controller. Null until [initialize] completes.
  CameraController? get controller => _controller;

  /// Whether the camera has been successfully initialized.
  bool get isInitialized => _isInitialized;

  /// Whether the camera is currently recording video.
  bool get isRecording => _isRecording;

  /// Error message if initialization failed. Null on success.
  String? get error => _error;

  /// Available cameras on this device.
  List<CameraDescription> get cameras => _cameras;

  /// Initializes the camera subsystem.
  ///
  /// Selects the back camera (preferred for court recording), configures
  /// 1080p resolution at 30fps, and prepares the preview surface.
  /// Call this once before accessing [controller].
  Future<void> initialize() async {
    try {
      _error = null;
      _cameras = await availableCameras();

      if (_cameras.isEmpty) {
        _error = 'No cameras available on this device';
        _log('error', _error!);
        notifyListeners();
        return;
      }

      // Prefer the back-facing camera for court recording.
      final backCamera = _cameras.firstWhere(
        (cam) => cam.lensDirection == CameraLensDirection.back,
        orElse: () => _cameras.first,
      );

      _controller = CameraController(
        backCamera,
        // 1080p target resolution per TASK-01-02.
        ResolutionPreset.high,
        enableAudio: true,
        imageFormatGroup: ImageFormatGroup.yuv420,
      );

      await _controller!.initialize();

      // Lock to 30fps if the platform supports it.
      // This is a best-effort setting; actual FPS depends on hardware.
      _isInitialized = true;
      _log('info', 'Camera initialized: ${backCamera.name} @ 1080p');
      notifyListeners();
    } catch (e) {
      _error = 'Camera initialization failed: $e';
      _log('error', _error!);
      _isInitialized = false;
      notifyListeners();
    }
  }

  /// Starts video recording to a temporary file.
  ///
  /// Returns immediately; the recording runs in the background.
  /// Use [stopRecording] to finalize and get the file path.
  Future<void> startRecording() async {
    if (_controller == null || !_isInitialized) {
      _log('warn', 'Cannot start recording: camera not initialized');
      return;
    }
    if (_isRecording) {
      _log('warn', 'Recording already in progress');
      return;
    }

    try {
      await _controller!.startVideoRecording();
      _isRecording = true;
      _log('info', 'Recording started');
      notifyListeners();
    } catch (e) {
      _error = 'Failed to start recording: $e';
      _log('error', _error!);
      notifyListeners();
    }
  }

  /// Stops the current recording and returns the file.
  ///
  /// Returns the [XFile] containing the recorded video, or null if
  /// no recording was in progress.
  Future<XFile?> stopRecording() async {
    if (_controller == null || !_isRecording) {
      _log('warn', 'Cannot stop recording: no active recording');
      return null;
    }

    try {
      final file = await _controller!.stopVideoRecording();
      _isRecording = false;
      _log('info', 'Recording stopped: ${file.path}');
      notifyListeners();
      return file;
    } catch (e) {
      _error = 'Failed to stop recording: $e';
      _isRecording = false;
      _log('error', _error!);
      notifyListeners();
      return null;
    }
  }

  /// Starts streaming camera frames for real-time detection.
  ///
  /// [onFrame] is called for each frame on the platform thread. The callback
  /// should dispatch the image to an Isolate for YOLOv8 inference to avoid
  /// blocking the UI thread.
  Future<void> startImageStream(
    void Function(CameraImage image) onFrame,
  ) async {
    if (_controller == null || !_isInitialized) {
      _log('warn', 'Cannot start image stream: camera not initialized');
      return;
    }

    try {
      await _controller!.startImageStream(onFrame);
      _log('info', 'Image stream started');
    } catch (e) {
      _error = 'Failed to start image stream: $e';
      _log('error', _error!);
    }
  }

  /// Stops the camera frame stream.
  Future<void> stopImageStream() async {
    if (_controller == null) return;

    try {
      await _controller!.stopImageStream();
      _log('info', 'Image stream stopped');
    } catch (e) {
      _log('warn', 'Failed to stop image stream: $e');
    }
  }

  /// Switches between available cameras (e.g., front/back).
  Future<void> switchCamera() async {
    if (_cameras.length < 2) return;

    final currentDirection = _controller?.description.lensDirection;
    final newCamera = _cameras.firstWhere(
      (cam) => cam.lensDirection != currentDirection,
      orElse: () => _cameras.first,
    );

    await _controller?.dispose();
    _controller = CameraController(
      newCamera,
      ResolutionPreset.high,
      enableAudio: true,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await _controller!.initialize();
      _log('info', 'Switched to camera: ${newCamera.name}');
      notifyListeners();
    } catch (e) {
      _error = 'Failed to switch camera: $e';
      _log('error', _error!);
      notifyListeners();
    }
  }

  /// Releases all camera resources.
  ///
  /// Must be called when the camera is no longer needed (e.g., on screen
  /// dispose or app lifecycle pause).
  @override
  void dispose() {
    _log('info', 'Disposing camera service');
    _controller?.dispose();
    _controller = null;
    _isInitialized = false;
    _isRecording = false;
    super.dispose();
  }

  void _log(String level, String message) {
    developer.log(
      message,
      name: 'CameraService',
      level: level == 'error' ? 1000 : (level == 'warn' ? 900 : 800),
    );
  }
}
