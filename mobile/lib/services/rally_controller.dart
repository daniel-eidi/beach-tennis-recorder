import 'dart:async';
import 'dart:developer' as developer;
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:uuid/uuid.dart';

import '../models/detection.dart';
import '../models/rally.dart';
import 'clip_service.dart';
import 'settings_service.dart';

/// Implements the rally detection state machine.
///
/// State transitions:
///   IDLE ---(ball detected + velocity > threshold)---> EM_JOGO
///   EM_JOGO ---(ball touches ground / out of bounds / timeout)---> FIM_RALLY
///   FIM_RALLY ---(clip saved, buffer reset)---> IDLE
///
/// This controller processes frame-by-frame detections from the YOLOv8 model
/// (running in a separate Isolate) and determines rally boundaries. It triggers
/// clip extraction via [ClipService] when a rally completes.
///
/// Implements TASK-01-04.
class RallyController extends ChangeNotifier {
  // --- Default constants (from CLAUDE.md contract) ---
  // These are used as fallbacks when no SettingsService is available.

  static const int _defaultPreRallySeconds = 3;
  static const int _defaultPostRallySeconds = 2;
  static const double _defaultVelocityThreshold = 15.0;
  static const double _defaultConfidenceThreshold = 0.45;
  static const int _defaultTimeoutSeconds = 8;
  static const bool _defaultNetCrossRequired = true;

  // --- Dependencies ---

  final ClipService _clipService;
  SettingsService? _settingsService;
  final _uuid = const Uuid();

  // --- Dynamic settings accessors (read from SettingsService or defaults) ---

  int get bufferPreRallySeconds =>
      _settingsService?.preRallySeconds ?? _defaultPreRallySeconds;

  int get bufferPostRallySeconds =>
      _settingsService?.postRallySeconds ?? _defaultPostRallySeconds;

  double get velocityThreshold =>
      _settingsService?.velocityThreshold ?? _defaultVelocityThreshold;

  double get confidenceThreshold =>
      _settingsService?.confidenceThreshold ?? _defaultConfidenceThreshold;

  int get rallyTimeoutSeconds =>
      _settingsService?.timeoutSeconds ?? _defaultTimeoutSeconds;

  bool get netCrossRequired =>
      _settingsService?.netCrossRequired ?? _defaultNetCrossRequired;

  // --- State ---

  RallyState _state = RallyState.idle;
  Rally? _currentRally;
  int _matchId = 0;
  int _rallyCounter = 0;

  /// Track ball positions across frames for velocity calculation.
  final List<_BallPosition> _ballHistory = [];

  /// Tracks whether the ball has crossed the net during the current rally.
  bool _netCrossDetected = false;

  /// Last known net X-coordinate (center of court, determined from net
  /// detections). Null until a net is detected.
  double? _netXPosition;

  /// Timer that fires when no ball detection occurs for [rallyTimeoutSeconds].
  Timer? _timeoutTimer;

  /// Timestamp when the rally started (ball first detected in motion).
  DateTime? _rallyStartTime;

  /// Completed rallies for the current match.
  final List<Rally> _completedRallies = [];

  RallyController({
    required ClipService clipService,
    SettingsService? settingsService,
  })  : _clipService = clipService,
        _settingsService = settingsService;

  /// Updates the settings service reference (called by ProxyProvider).
  void updateSettings(SettingsService settingsService) {
    _settingsService = settingsService;
  }

  // --- Public API ---

  /// Current state of the rally state machine.
  RallyState get state => _state;

  /// The rally currently in progress, or null if idle.
  Rally? get currentRally => _currentRally;

  /// Number of rallies completed in the current match.
  int get rallyCount => _rallyCounter;

  /// All completed rallies for the current match.
  List<Rally> get completedRallies => List.unmodifiable(_completedRallies);

  /// Whether a rally is currently in progress.
  bool get isRallyActive => _state == RallyState.emJogo;

  /// The current match ID.
  int get matchId => _matchId;

  /// Starts a new match recording session.
  ///
  /// Resets all state and sets the match ID for clip naming.
  void startMatch(int matchId) {
    _matchId = matchId;
    _rallyCounter = 0;
    _completedRallies.clear();
    _resetToIdle();
    _log('info', 'Match $matchId started');
    notifyListeners();
  }

  /// Ends the current match session.
  void endMatch() {
    _timeoutTimer?.cancel();
    _log('info', 'Match $_matchId ended with $_rallyCounter rallies');
    _resetToIdle();
    notifyListeners();
  }

  /// Processes a list of detections from a single frame.
  ///
  /// This is the main entry point called by the detection Isolate for each
  /// frame. It updates the state machine based on the detected objects.
  ///
  /// [timestamp] is the capture time of the frame.
  void processDetections(
    List<Detection> detections, {
    DateTime? timestamp,
  }) {
    final now = timestamp ?? DateTime.now();

    // Filter detections by confidence threshold.
    final confident = detections
        .where((d) => d.confidence >= confidenceThreshold)
        .toList();

    // Extract ball and net detections.
    final ballDetections = confident.where((d) => d.isBall).toList();
    final netDetections = confident.where((d) => d.isNet).toList();

    // Update net position if detected.
    if (netDetections.isNotEmpty) {
      _netXPosition = netDetections
              .map((d) => d.x)
              .reduce((a, b) => a + b) /
          netDetections.length;
    }

    switch (_state) {
      case RallyState.idle:
        _handleIdleState(ballDetections, now);
        break;
      case RallyState.emJogo:
        _handleEmJogoState(ballDetections, confident, now);
        break;
      case RallyState.fimRally:
        // FIM_RALLY is transient; should not receive detections in this state.
        break;
    }
  }

  // --- State handlers ---

  void _handleIdleState(List<Detection> balls, DateTime now) {
    if (balls.isEmpty) return;

    final ball = balls.first;
    _recordBallPosition(ball, now);

    final velocity = _calculateVelocity();
    if (velocity != null && velocity > velocityThreshold) {
      // Ball is moving fast enough to indicate a serve or rally start.
      _transitionToEmJogo(now);
    }
  }

  void _handleEmJogoState(
    List<Detection> balls,
    List<Detection> allDetections,
    DateTime now,
  ) {
    if (balls.isEmpty) {
      // No ball detected this frame. The timeout timer is already running.
      return;
    }

    final ball = balls.first;
    _recordBallPosition(ball, now);

    // Reset the timeout timer since we detected the ball.
    _resetTimeoutTimer();

    // Check for net crossing.
    _checkNetCrossing(ball);

    // Check for ground touch (rapid downward Y-direction change).
    if (_detectGroundTouch()) {
      _transitionToFimRally(
        now,
        endedByGroundTouch: true,
      );
      return;
    }

    // Check if ball is out of bounds (would need court calibration data).
    // For now, this is a placeholder for TASK-01-13 calibration integration.
    if (_detectOutOfBounds(ball)) {
      _transitionToFimRally(
        now,
        endedByOutOfBounds: true,
      );
      return;
    }
  }

  // --- State transitions ---

  void _transitionToEmJogo(DateTime now) {
    _state = RallyState.emJogo;
    _rallyStartTime = now;
    _netCrossDetected = false;
    _rallyCounter++;

    _currentRally = Rally(
      id: _uuid.v4(),
      matchId: _matchId,
      rallyNumber: _rallyCounter,
      startTime: now,
      state: RallyState.emJogo,
    );

    _resetTimeoutTimer();
    _log('info', 'Rally $_rallyCounter started');
    notifyListeners();
  }

  void _transitionToFimRally(
    DateTime now, {
    bool endedByGroundTouch = false,
    bool endedByOutOfBounds = false,
    bool endedByTimeout = false,
  }) {
    // If net crossing is required but hasn't happened, demote to false alarm.
    if (netCrossRequired && !_netCrossDetected && !endedByTimeout) {
      _log('info', 'Rally $_rallyCounter dismissed: no net crossing detected');
      _rallyCounter--;
      _resetToIdle();
      notifyListeners();
      return;
    }

    _state = RallyState.fimRally;
    _timeoutTimer?.cancel();

    _currentRally = _currentRally?.copyWith(
      endTime: now,
      state: RallyState.fimRally,
      netCrossings: _netCrossDetected ? 1 : 0,
      endedByGroundTouch: endedByGroundTouch,
      endedByOutOfBounds: endedByOutOfBounds,
      endedByTimeout: endedByTimeout,
    );

    if (_currentRally != null) {
      _completedRallies.add(_currentRally!);
    }

    _log('info', 'Rally $_rallyCounter ended '
        '(ground=$endedByGroundTouch out=$endedByOutOfBounds '
        'timeout=$endedByTimeout)');
    notifyListeners();

    // Trigger clip extraction asynchronously.
    _triggerClipExtraction(now);
  }

  void _resetToIdle() {
    _state = RallyState.idle;
    _currentRally = null;
    _rallyStartTime = null;
    _ballHistory.clear();
    _netCrossDetected = false;
    _timeoutTimer?.cancel();
    _timeoutTimer = null;
  }

  // --- Detection logic ---

  void _recordBallPosition(Detection ball, DateTime timestamp) {
    _ballHistory.add(_BallPosition(
      x: ball.x,
      y: ball.y,
      timestamp: timestamp,
    ));

    // Keep only the last 10 positions for velocity calculation.
    if (_ballHistory.length > 10) {
      _ballHistory.removeAt(0);
    }
  }

  /// Calculates ball velocity in pixels per frame from the last two positions.
  double? _calculateVelocity() {
    if (_ballHistory.length < 2) return null;

    final prev = _ballHistory[_ballHistory.length - 2];
    final curr = _ballHistory.last;

    final dx = curr.x - prev.x;
    final dy = curr.y - prev.y;
    return math.sqrt(dx * dx + dy * dy);
  }

  /// Detects if the ball has crossed the net by checking if consecutive
  /// positions are on opposite sides of the net's X coordinate.
  void _checkNetCrossing(Detection ball) {
    if (_netXPosition == null) return;
    if (_ballHistory.length < 2) return;

    final prev = _ballHistory[_ballHistory.length - 2];
    final curr = _ballHistory.last;

    // Check if the ball crossed from one side of the net to the other.
    final prevSide = prev.x < _netXPosition!;
    final currSide = curr.x < _netXPosition!;

    if (prevSide != currSide) {
      _netCrossDetected = true;
      if (_currentRally != null) {
        _currentRally = _currentRally!.copyWith(
          netCrossings: _currentRally!.netCrossings + 1,
        );
      }
    }
  }

  /// Detects a ground touch by looking for a rapid reversal in Y direction.
  ///
  /// A bounce appears as the ball moving downward (increasing Y) and then
  /// rapidly changing to upward (decreasing Y) or stopping.
  bool _detectGroundTouch() {
    if (_ballHistory.length < 3) return false;

    final p1 = _ballHistory[_ballHistory.length - 3];
    final p2 = _ballHistory[_ballHistory.length - 2];
    final p3 = _ballHistory.last;

    final dy1 = p2.y - p1.y; // Previous vertical movement.
    final dy2 = p3.y - p2.y; // Current vertical movement.

    // Ball was going down (positive dy1) and now going up (negative dy2)
    // with significant magnitude change indicating a bounce.
    const bounceThreshold = 10.0;
    return dy1 > bounceThreshold && dy2 < -bounceThreshold;
  }

  /// Placeholder for out-of-bounds detection.
  ///
  /// Requires court calibration data from TASK-01-13 to determine court
  /// boundaries via homography transform.
  bool _detectOutOfBounds(Detection ball) {
    // TODO(TASK-01-13): Implement using calibrated court corners.
    return false;
  }

  void _resetTimeoutTimer() {
    _timeoutTimer?.cancel();
    _timeoutTimer = Timer(
      Duration(seconds: rallyTimeoutSeconds),
      () {
        if (_state == RallyState.emJogo) {
          _log('info', 'Rally timeout: no detection for ${rallyTimeoutSeconds}s');
          _transitionToFimRally(
            DateTime.now(),
            endedByTimeout: true,
          );
        }
      },
    );
  }

  /// Triggers async clip extraction and transitions back to IDLE.
  void _triggerClipExtraction(DateTime rallyEndTime) {
    if (_rallyStartTime == null) {
      _resetToIdle();
      notifyListeners();
      return;
    }

    final event = RallyEvent(
      startTime: _rallyStartTime!,
      endTime: rallyEndTime,
      bufferFilePath: '', // BufferService handles path internally.
      matchId: _matchId,
      rallyNumber: _rallyCounter,
    );

    // Extract clip in the background. Don't block state machine.
    _clipService.extractClip(event).then((clip) {
      if (clip != null) {
        _log('info', 'Clip extracted for rally $_rallyCounter');
      } else {
        _log('warn', 'Clip extraction failed for rally $_rallyCounter');
      }
    }).catchError((e) {
      _log('error', 'Clip extraction error: $e');
    });

    // Reset to idle immediately so we can detect the next rally.
    _resetToIdle();
    notifyListeners();
  }

  void _log(String level, String message) {
    developer.log(
      message,
      name: 'RallyController',
      level: level == 'error' ? 1000 : (level == 'warn' ? 900 : 800),
    );
  }

  @override
  void dispose() {
    _timeoutTimer?.cancel();
    super.dispose();
  }
}

/// Internal representation of a ball position at a point in time.
class _BallPosition {
  final double x;
  final double y;
  final DateTime timestamp;

  const _BallPosition({
    required this.x,
    required this.y,
    required this.timestamp,
  });
}
