import 'dart:async';
import 'dart:developer' as developer;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';

import '../models/gesture.dart';
import '../models/match.dart';
import '../models/rally.dart';
import 'buffer_service.dart';
import 'camera_service.dart';
import 'clip_service.dart';
import 'detection_isolate.dart';
import 'gesture_detector_service.dart';
import 'match_service.dart';
import 'rally_controller.dart';

/// Orchestrates the full recording and detection pipeline.
///
/// Ties together all services into a single cohesive flow:
///   Camera -> Buffer (disk)
///   Camera -> Detection Isolate -> RallyController -> ClipService
///
/// This is the primary interface used by [RecordingScreen] to control the
/// entire recording session. Exposes real-time stats for the UI overlay.
///
/// Pipeline lifecycle:
///   1. [startRecording] — initializes camera, buffer, detection isolate
///   2. Camera frames stream to both buffer (for video) and isolate (for AI)
///   3. Detection results feed into RallyController state machine
///   4. Rally completion triggers ClipService extraction from buffer
///   5. [stopRecording] — tears down everything cleanly
class PipelineController extends ChangeNotifier {
  final CameraService _cameraService;
  final BufferService _bufferService;
  final ClipService _clipService;
  final RallyController _rallyController;
  final MatchService _matchService;
  final GestureDetectorService _gestureDetectorService;

  late final IsolateDetectionService _detectionService;
  StreamSubscription<DetectionResult>? _detectionSubscription;
  StreamSubscription<GestureEvent>? _gestureSubscription;

  bool _isRecording = false;
  bool _modelAvailable = false;
  Match? _currentMatch;
  DateTime? _recordingStartTime;

  // Highlight bookmarks (timestamps relative to recording start)
  final List<Duration> _highlightMarkers = [];

  // --- Performance stats ---
  int _framesProcessed = 0;
  int _detectionsFound = 0;
  int _inferenceTimeMs = 0;
  double _fps = 0.0;
  DateTime? _lastFpsUpdate;
  int _framesSinceLastFpsUpdate = 0;

  PipelineController({
    required CameraService cameraService,
    required BufferService bufferService,
    required ClipService clipService,
    required RallyController rallyController,
    required MatchService matchService,
    required GestureDetectorService gestureDetectorService,
  })  : _cameraService = cameraService,
        _bufferService = bufferService,
        _clipService = clipService,
        _rallyController = rallyController,
        _matchService = matchService,
        _gestureDetectorService = gestureDetectorService {
    _detectionService = IsolateDetectionService();
  }

  // --- Public getters ---

  /// Whether the full pipeline is active.
  bool get isRecording => _isRecording;

  /// Whether the TFLite model was loaded successfully.
  bool get modelAvailable => _modelAvailable;

  /// Current rally state from the state machine.
  RallyState get currentState => _rallyController.state;

  /// Number of rallies detected in the current match.
  int get rallyCount => _rallyController.rallyCount;

  /// Total frames processed by the detection isolate.
  int get framesProcessed => _framesProcessed;

  /// Total detections found across all frames.
  int get detectionsFound => _detectionsFound;

  /// Most recent inference time in milliseconds.
  int get inferenceTimeMs => _inferenceTimeMs;

  /// Approximate detection FPS (frames per second through the model).
  double get fps => _fps;

  /// The current active match.
  Match? get currentMatch => _currentMatch;

  /// The underlying camera service (for preview widget access).
  CameraService get cameraService => _cameraService;

  /// The underlying rally controller (for detailed rally state).
  RallyController get rallyController => _rallyController;

  /// The gesture detector service (for gesture state).
  GestureDetectorService get gestureDetectorService =>
      _gestureDetectorService;

  /// Current state of the gesture detection state machine.
  GestureState get gestureState => _gestureDetectorService.state;

  /// Number of gestures detected in the current session.
  int get gestureCount => _gestureDetectorService.gestureCount;

  /// Whether gesture detection is currently enabled.
  bool get isGestureEnabled => _gestureDetectorService.enabled;

  /// The clip service (for accessing saved clips).
  ClipService get clipService => _clipService;

  /// Highlight bookmark markers (timestamps from recording start).
  List<Duration> get highlightMarkers => List.unmodifiable(_highlightMarkers);

  /// Number of highlight markers in current session.
  int get highlightCount => _highlightMarkers.length;

  /// Adds a highlight marker at the current recording position.
  /// Does NOT stop the recording — just saves the timestamp.
  void addHighlightMarker() {
    if (!_isRecording || _recordingStartTime == null) return;
    final marker = DateTime.now().difference(_recordingStartTime!);
    _highlightMarkers.add(marker);
    _log('info', 'Highlight marker added at ${marker.inSeconds}s '
        '(total: ${_highlightMarkers.length})');
    notifyListeners();
  }

  // --- Pipeline control ---

  /// Starts the full recording and detection pipeline.
  ///
  /// 1. Creates a new match
  /// 2. Initializes camera (if needed)
  /// 3. Starts disk buffer
  /// 4. Spawns detection isolate and loads model
  /// 5. Begins camera frame streaming
  /// 6. Starts rally state machine
  Future<void> startRecording() async {
    if (_isRecording) {
      _log('warn', 'Pipeline already recording');
      return;
    }

    try {
      _log('info', 'Starting pipeline...');

      // Reset stats and markers.
      _framesProcessed = 0;
      _detectionsFound = 0;
      _inferenceTimeMs = 0;
      _fps = 0.0;
      _lastFpsUpdate = DateTime.now();
      _framesSinceLastFpsUpdate = 0;
      _highlightMarkers.clear();
      _recordingStartTime = DateTime.now();

      // Step 1: Create a new match.
      _currentMatch = await _matchService.createMatch();
      _log('info', 'Match created: ${_currentMatch!.id}');

      // Step 2: Initialize camera.
      if (!_cameraService.isInitialized) {
        await _cameraService.initialize();
      }
      if (!_cameraService.isInitialized) {
        _log('error', 'Camera initialization failed, aborting pipeline');
        return;
      }

      // Step 3: Start disk buffer.
      await _bufferService.initialize();
      await _bufferService.startBuffering();

      // Step 4: Start camera recording.
      // Note: On iOS, startVideoRecording and startImageStream are
      // mutually exclusive. For MVP, we prioritize video recording
      // (so highlights work) over real-time detection.
      await _cameraService.startRecording();

      // Step 5: Start rally state machine.
      _rallyController.startMatch(_currentMatch!.id);

      // Step 6: Reset gesture detector and listen for gesture events.
      _gestureDetectorService.reset();
      _gestureSubscription = _gestureDetectorService.gestureEvents.listen(
        _onGestureEvent,
      );

      // Step 7: Detection isolate (disabled for MVP — conflicts with recording).
      // TODO: Enable when using a separate camera stream or platform channels.
      _modelAvailable = false;
      _log('info', 'Detection disabled for MVP (recording mode)');

      _isRecording = true;
      _log('info', 'Pipeline started (model=$_modelAvailable)');
      notifyListeners();
    } catch (e) {
      _log('error', 'Failed to start pipeline: $e');
      await _cleanupOnError();
    }
  }

  /// Stops the full pipeline and cleans up all resources.
  Future<void> stopRecording() async {
    if (!_isRecording) return;

    _log('info', 'Stopping pipeline...');
    _isRecording = false;

    try {
      // Stop gesture detection.
      await _gestureSubscription?.cancel();
      _gestureSubscription = null;

      // Stop camera recording and get the full video file.
      final videoFile = await _cameraService.stopRecording();

      // Stop buffer.
      await _bufferService.stopBuffering(keepFiles: false);

      // End rally state machine.
      _rallyController.endMatch();

      // Save the full recording to the clip library.
      if (videoFile != null && _currentMatch != null) {
        await _clipService.saveFullRecording(
          sourceFilePath: videoFile.path,
          matchId: _currentMatch!.id,
          highlightMarkers: _highlightMarkers,
        );
        _log('info', 'Full recording saved with '
            '${_highlightMarkers.length} highlight markers');
      }

      // Update match metadata.
      if (_currentMatch != null) {
        final updatedMatch = _currentMatch!.copyWith(
          clipCount: _clipService.clipCountForMatch(_currentMatch!.id),
        );
        await _matchService.updateMatch(updatedMatch);
      }

      _matchService.endCurrentMatch();
      _recordingStartTime = null;
      _currentMatch = null;

      _log('info', 'Pipeline stopped '
          '(highlights=${_highlightMarkers.length} '
          'gestures=${_gestureDetectorService.gestureCount})');
      notifyListeners();
    } catch (e) {
      _log('error', 'Error stopping pipeline: $e');
    }
  }

  // --- Private handlers ---

  /// Called for each camera frame from the image stream.
  void _onCameraFrame(CameraImage image) {
    if (!_isRecording) return;

    // Send frame to detection isolate. If busy, it will be dropped
    // automatically by IsolateDetectionService.
    if (_modelAvailable) {
      _detectionService.processFrame(image);
    }
  }

  /// Called when the detection isolate returns results for a frame.
  void _onDetectionResult(DetectionResult result) {
    if (!_isRecording) return;

    _framesProcessed++;
    _framesSinceLastFpsUpdate++;
    _inferenceTimeMs = result.inferenceTimeMs;

    // Update FPS counter every second.
    final now = DateTime.now();
    if (_lastFpsUpdate != null) {
      final elapsed = now.difference(_lastFpsUpdate!).inMilliseconds;
      if (elapsed >= 1000) {
        _fps = _framesSinceLastFpsUpdate / (elapsed / 1000.0);
        _framesSinceLastFpsUpdate = 0;
        _lastFpsUpdate = now;
      }
    }

    // Count total detections.
    _detectionsFound += result.detections.length;

    // Feed detections into the rally state machine.
    _rallyController.processDetections(
      result.detections,
      timestamp: result.frameTimestamp,
    );

    // Feed detections into the gesture detector (runs in parallel with rally).
    _gestureDetectorService.processDetections(
      result.detections,
      640, // Model input size (detections are in model coordinate space).
      640,
    );

    // Notify UI of updated stats.
    notifyListeners();
  }

  /// Called when the gesture detector emits a gesture event.
  ///
  /// Triggers a highlight clip extraction from the buffer.
  void _onGestureEvent(GestureEvent event) {
    if (!_isRecording || _currentMatch == null) return;

    _log('info', 'Gesture event received — saving '
        '${event.clipDurationSeconds}s highlight');

    _clipService
        .saveHighlight(
          triggerTime: event.timestamp,
          durationSeconds: event.clipDurationSeconds,
          matchId: _currentMatch!.id,
        )
        .then((clip) {
      if (clip != null) {
        _log('info', 'Highlight clip saved: ${clip.fileName}');
      } else {
        _log('warn', 'Highlight clip extraction failed');
      }
      notifyListeners();
    }).catchError((e) {
      _log('error', 'Highlight extraction error: $e');
    });
  }

  /// Cleans up resources after a startup failure.
  Future<void> _cleanupOnError() async {
    try {
      await _detectionService.stop();
      await _detectionSubscription?.cancel();
      await _gestureSubscription?.cancel();
      await _cameraService.stopImageStream();
      await _cameraService.stopRecording();
      await _bufferService.stopBuffering();
      _rallyController.endMatch();
    } catch (_) {
      // Best-effort cleanup.
    }
    _isRecording = false;
    notifyListeners();
  }

  void _log(String level, String message) {
    developer.log(
      message,
      name: 'PipelineController',
      level: level == 'error' ? 1000 : (level == 'warn' ? 900 : 800),
    );
  }

  @override
  void dispose() {
    _detectionSubscription?.cancel();
    _gestureSubscription?.cancel();
    _detectionService.dispose();
    super.dispose();
  }
}
