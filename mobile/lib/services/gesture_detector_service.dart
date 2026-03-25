import 'dart:async';
import 'dart:developer' as developer;

import 'package:flutter/foundation.dart';

import '../models/detection.dart';
import '../models/gesture.dart';
import 'settings_service.dart';

/// On-device gesture detection using existing YOLO detections.
///
/// Detects the "palm-to-racket" gesture (player raises arms and claps hand
/// against racket) by analyzing spatial relationships between player and
/// racket bounding boxes across frames. No extra inference is needed — this
/// runs on the same detections as rally detection.
///
/// Detection logic:
///   1. Track player and racket bounding boxes across frames
///   2. Detect "racket raised high": racket bbox is in the upper 30% of the
///      player bbox (arms raised above shoulders)
///   3. Detect "clap motion": sudden change in racket bbox area or position
///      while raised, indicating contact with hand
///
/// State machine:
///   IDLE -> RACKET_RAISED (racket in upper portion for 2+ frames)
///   RACKET_RAISED -> GESTURE_DETECTED (clap motion detected)
///   GESTURE_DETECTED -> COOLDOWN (clip saved, wait cooldown)
///   COOLDOWN -> IDLE (after cooldownSeconds)
class GestureDetectorService extends ChangeNotifier {
  SettingsService? _settingsService;

  // --- Configuration defaults ---

  static const bool _defaultEnabled = true;
  static const int _defaultCooldownSeconds = 5;
  static const int _defaultHighlightDuration = 30;
  static const double _defaultSensitivity = 0.5;

  // --- Detection thresholds ---

  /// Fraction of the player bbox height from the top where the racket must
  /// be to count as "raised". Lower sensitivity = stricter (smaller region).
  double get _upperRegionFraction {
    final sens = sensitivity;
    // High sensitivity (1.0) -> 0.40 (top 40%)
    // Medium (0.5) -> 0.30 (top 30%)
    // Low (0.0) -> 0.20 (top 20%)
    return 0.20 + sens * 0.20;
  }

  /// Minimum number of consecutive frames with racket raised before
  /// transitioning from IDLE to RACKET_RAISED.
  int get _minRaisedFrames {
    final sens = sensitivity;
    // High sensitivity -> 1 frame, Medium -> 2, Low -> 3
    if (sens > 0.7) return 1;
    if (sens > 0.3) return 2;
    return 3;
  }

  /// Minimum relative change in racket bbox area to detect a clap motion.
  double get _areaChangeThreshold {
    final sens = sensitivity;
    // High sensitivity -> 0.10 (10% change), Low -> 0.25 (25% change)
    return 0.25 - sens * 0.15;
  }

  /// Minimum position change (as fraction of racket height) to detect clap.
  double get _positionChangeThreshold {
    final sens = sensitivity;
    // High sensitivity -> 0.05, Low -> 0.15
    return 0.15 - sens * 0.10;
  }

  // --- Dynamic settings accessors ---

  bool get enabled =>
      _settingsService?.gestureEnabled ?? _defaultEnabled;

  int get cooldownSeconds =>
      _settingsService?.gestureCooldownSeconds ?? _defaultCooldownSeconds;

  int get highlightDurationSeconds =>
      _settingsService?.gestureHighlightDuration ?? _defaultHighlightDuration;

  double get sensitivity =>
      _settingsService?.gestureSensitivity ?? _defaultSensitivity;

  // --- State ---

  GestureState _state = GestureState.idle;
  int _consecutiveRaisedFrames = 0;
  int _gestureCount = 0;
  DateTime? _lastGestureTime;
  Timer? _cooldownTimer;

  /// History of racket bounding box snapshots for motion analysis.
  final List<_RacketSnapshot> _racketHistory = [];

  /// Stream controller for gesture events.
  final _gestureEventController = StreamController<GestureEvent>.broadcast();

  GestureDetectorService({SettingsService? settingsService})
      : _settingsService = settingsService;

  // --- Public API ---

  /// Current state of the gesture state machine.
  GestureState get state => _state;

  /// Whether a gesture was recently detected (for UI flash).
  bool get gestureDetected => _state == GestureState.gestureDetected;

  /// Total number of gestures detected in the current session.
  int get gestureCount => _gestureCount;

  /// Stream of gesture events for triggering clip saves.
  Stream<GestureEvent> get gestureEvents => _gestureEventController.stream;

  /// Timestamp of the last detected gesture.
  DateTime? get lastGestureTime => _lastGestureTime;

  /// Updates the settings service reference.
  void updateSettings(SettingsService settingsService) {
    _settingsService = settingsService;
  }

  /// Resets all state for a new recording session.
  void reset() {
    _state = GestureState.idle;
    _consecutiveRaisedFrames = 0;
    _gestureCount = 0;
    _lastGestureTime = null;
    _racketHistory.clear();
    _cooldownTimer?.cancel();
    _cooldownTimer = null;
    notifyListeners();
  }

  /// Processes a list of detections from a single frame.
  ///
  /// Analyzes spatial relationships between player and racket detections to
  /// identify the palm-to-racket gesture. This method runs on the same
  /// detection results as rally detection — no extra inference cost.
  ///
  /// [detections] are the YOLO detections from the current frame.
  /// [frameWidth] and [frameHeight] are the original frame dimensions.
  void processDetections(
    List<Detection> detections,
    int frameWidth,
    int frameHeight,
  ) {
    if (!enabled) return;
    if (_state == GestureState.cooldown) return;

    // Extract player and racket detections.
    final players = detections
        .where((d) => d.classId == DetectionClass.player)
        .toList();
    final rackets = detections
        .where((d) => d.classId == DetectionClass.racket)
        .toList();

    // Need at least one player and one racket to analyze.
    if (players.isEmpty || rackets.isEmpty) {
      _onNoRelevantDetections();
      return;
    }

    // Find the best player-racket pair: the racket most overlapping with or
    // closest to a player bbox.
    final (player, racket) = _findBestPair(players, rackets);
    if (player == null || racket == null) {
      _onNoRelevantDetections();
      return;
    }

    // Record racket snapshot for motion analysis.
    _recordRacketSnapshot(racket);

    // Check if racket is in the upper region of the player bbox.
    final isRacketRaised = _isRacketInUpperRegion(player, racket);

    switch (_state) {
      case GestureState.idle:
        _handleIdleState(isRacketRaised);
        break;
      case GestureState.racketRaised:
        _handleRacketRaisedState(isRacketRaised);
        break;
      case GestureState.gestureDetected:
        // Transient state; should transition to cooldown quickly.
        break;
      case GestureState.cooldown:
        // Handled by timer; skip processing.
        break;
    }
  }

  // --- State handlers ---

  void _handleIdleState(bool isRacketRaised) {
    if (isRacketRaised) {
      _consecutiveRaisedFrames++;
      if (_consecutiveRaisedFrames >= _minRaisedFrames) {
        _state = GestureState.racketRaised;
        _log('info', 'Racket raised detected '
            '(frames=$_consecutiveRaisedFrames)');
        notifyListeners();
      }
    } else {
      _consecutiveRaisedFrames = 0;
    }
  }

  void _handleRacketRaisedState(bool isRacketRaised) {
    if (!isRacketRaised) {
      // Racket lowered without clap — false alarm.
      _consecutiveRaisedFrames = 0;
      _state = GestureState.idle;
      _racketHistory.clear();
      notifyListeners();
      return;
    }

    // Check for clap motion while racket is raised.
    if (_detectClapMotion()) {
      _transitionToGestureDetected();
    }
  }

  void _onNoRelevantDetections() {
    // If we were tracking a potential gesture, reset after a few missing frames.
    if (_state == GestureState.idle) {
      _consecutiveRaisedFrames = 0;
    } else if (_state == GestureState.racketRaised) {
      // Give a 2-frame grace period for detection drops.
      _consecutiveRaisedFrames--;
      if (_consecutiveRaisedFrames <= 0) {
        _state = GestureState.idle;
        _consecutiveRaisedFrames = 0;
        _racketHistory.clear();
        notifyListeners();
      }
    }
  }

  // --- State transitions ---

  void _transitionToGestureDetected() {
    _state = GestureState.gestureDetected;
    _gestureCount++;
    _lastGestureTime = DateTime.now();

    final event = GestureEvent(
      timestamp: _lastGestureTime!,
      clipDurationSeconds: highlightDurationSeconds,
      confidence: 0.8, // Heuristic-based confidence.
    );

    _log('info', 'Gesture detected! (#$_gestureCount) '
        'Saving ${highlightDurationSeconds}s highlight');

    _gestureEventController.add(event);
    notifyListeners();

    // Transition to cooldown after a brief moment for UI feedback.
    Future.delayed(const Duration(milliseconds: 500), () {
      _transitionToCooldown();
    });
  }

  void _transitionToCooldown() {
    _state = GestureState.cooldown;
    _consecutiveRaisedFrames = 0;
    _racketHistory.clear();
    notifyListeners();

    _cooldownTimer?.cancel();
    _cooldownTimer = Timer(
      Duration(seconds: cooldownSeconds),
      () {
        _state = GestureState.idle;
        _log('info', 'Gesture cooldown ended');
        notifyListeners();
      },
    );
  }

  // --- Detection logic ---

  /// Checks whether the racket bbox center is within the upper region of
  /// the player bbox, indicating raised arms.
  bool _isRacketInUpperRegion(Detection player, Detection racket) {
    final playerTop = player.top;
    final playerHeight = player.h;
    final upperBound = playerTop + playerHeight * _upperRegionFraction;

    // The racket center Y must be above the upper bound threshold.
    return racket.y < upperBound;
  }

  /// Detects a clap motion by analyzing sudden changes in the racket bbox.
  ///
  /// A clap produces either:
  ///   - A sudden change in racket bbox area (hand overlapping racket)
  ///   - A rapid position shift (racket being struck)
  bool _detectClapMotion() {
    if (_racketHistory.length < 3) return false;

    final prev = _racketHistory[_racketHistory.length - 2];
    final curr = _racketHistory.last;

    // Check area change ratio.
    if (prev.area > 0) {
      final areaChangeRatio = (curr.area - prev.area).abs() / prev.area;
      if (areaChangeRatio > _areaChangeThreshold) {
        _log('info', 'Clap detected via area change: '
            '${(areaChangeRatio * 100).toStringAsFixed(1)}%');
        return true;
      }
    }

    // Check rapid position change (relative to racket height).
    if (prev.h > 0) {
      final posChangeRatio =
          ((curr.y - prev.y).abs() + (curr.x - prev.x).abs()) / prev.h;
      if (posChangeRatio > _positionChangeThreshold) {
        _log('info', 'Clap detected via position change: '
            '${(posChangeRatio * 100).toStringAsFixed(1)}%');
        return true;
      }
    }

    return false;
  }

  /// Finds the best player-racket pair based on spatial proximity.
  ///
  /// Returns the player whose bbox contains or is closest to a racket,
  /// along with the matching racket.
  (Detection?, Detection?) _findBestPair(
    List<Detection> players,
    List<Detection> rackets,
  ) {
    Detection? bestPlayer;
    Detection? bestRacket;
    double bestScore = double.infinity;

    for (final player in players) {
      for (final racket in rackets) {
        // Score: distance from racket center to player center,
        // with bonus for racket being inside the player bbox.
        final dx = racket.x - player.x;
        final dy = racket.y - player.y;
        double score = dx * dx + dy * dy;

        // Bonus: if racket is within the player bbox, greatly prefer it.
        if (racket.x >= player.left &&
            racket.x <= player.right &&
            racket.y >= player.top &&
            racket.y <= player.bottom) {
          score *= 0.1; // Strong preference for overlapping pairs.
        }

        if (score < bestScore) {
          bestScore = score;
          bestPlayer = player;
          bestRacket = racket;
        }
      }
    }

    return (bestPlayer, bestRacket);
  }

  /// Records a snapshot of the current racket bbox for motion analysis.
  void _recordRacketSnapshot(Detection racket) {
    _racketHistory.add(_RacketSnapshot(
      x: racket.x,
      y: racket.y,
      w: racket.w,
      h: racket.h,
      area: racket.area,
      timestamp: DateTime.now(),
    ));

    // Keep only the last 5 snapshots.
    if (_racketHistory.length > 5) {
      _racketHistory.removeAt(0);
    }
  }

  void _log(String level, String message) {
    developer.log(
      message,
      name: 'GestureDetectorService',
      level: level == 'error' ? 1000 : (level == 'warn' ? 900 : 800),
    );
  }

  @override
  void dispose() {
    _cooldownTimer?.cancel();
    _gestureEventController.close();
    super.dispose();
  }
}

/// Internal snapshot of a racket bounding box for motion tracking.
class _RacketSnapshot {
  final double x;
  final double y;
  final double w;
  final double h;
  final double area;
  final DateTime timestamp;

  const _RacketSnapshot({
    required this.x,
    required this.y,
    required this.w,
    required this.h,
    required this.area,
    required this.timestamp,
  });
}
