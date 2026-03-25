/// State of the gesture detection state machine.
///
/// State transitions:
///   IDLE ---(racket in upper region of player bbox for 2+ frames)---> RACKET_RAISED
///   RACKET_RAISED ---(racket bbox area/position change indicates clap)---> GESTURE_DETECTED
///   GESTURE_DETECTED ---(clip saved)---> COOLDOWN
///   COOLDOWN ---(after cooldownSeconds)---> IDLE
enum GestureState {
  /// No gesture activity detected. Watching for racket raised high.
  idle,

  /// Racket detected in the upper portion of a player's bounding box.
  racketRaised,

  /// Clap gesture confirmed. Highlight clip is being saved.
  gestureDetected,

  /// Cooldown period after a gesture to prevent duplicate triggers.
  cooldown,
}

/// Event emitted when a palm-to-racket gesture is detected.
///
/// Used to trigger highlight clip extraction from the buffer.
class GestureEvent {
  /// Timestamp when the gesture was detected.
  final DateTime timestamp;

  /// Duration in seconds of the highlight clip to extract.
  final int clipDurationSeconds;

  /// Confidence score of the gesture detection (0.0 - 1.0).
  final double confidence;

  const GestureEvent({
    required this.timestamp,
    this.clipDurationSeconds = 30,
    this.confidence = 1.0,
  });

  Map<String, dynamic> toJson() => {
        'timestamp': timestamp.toIso8601String(),
        'clipDurationSeconds': clipDurationSeconds,
        'confidence': confidence,
      };

  @override
  String toString() =>
      'GestureEvent(duration=${clipDurationSeconds}s conf=$confidence)';
}

/// Configuration for gesture detection behavior.
class GestureConfig {
  /// Whether gesture detection is enabled.
  final bool enabled;

  /// Seconds to wait after a gesture before allowing another.
  final int cooldownSeconds;

  /// Duration of the highlight clip in seconds.
  final int highlightDurationSeconds;

  /// Sensitivity of detection (0.0 = low, 0.5 = medium, 1.0 = high).
  /// Affects how strict the position/motion thresholds are.
  final double sensitivity;

  const GestureConfig({
    this.enabled = true,
    this.cooldownSeconds = 5,
    this.highlightDurationSeconds = 30,
    this.sensitivity = 0.5,
  });

  /// Creates a copy with updated fields.
  GestureConfig copyWith({
    bool? enabled,
    int? cooldownSeconds,
    int? highlightDurationSeconds,
    double? sensitivity,
  }) {
    return GestureConfig(
      enabled: enabled ?? this.enabled,
      cooldownSeconds: cooldownSeconds ?? this.cooldownSeconds,
      highlightDurationSeconds:
          highlightDurationSeconds ?? this.highlightDurationSeconds,
      sensitivity: sensitivity ?? this.sensitivity,
    );
  }

  @override
  String toString() =>
      'GestureConfig(enabled=$enabled cooldown=${cooldownSeconds}s '
      'highlight=${highlightDurationSeconds}s sensitivity=$sensitivity)';
}
