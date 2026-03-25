/// The state of a rally within the detection state machine.
///
/// State transitions:
///   IDLE ---(ball detected + velocity > threshold)---> EM_JOGO
///   EM_JOGO ---(ball touched ground / out of bounds / timeout)---> FIM_RALLY
///   FIM_RALLY ---(clip saved, buffer reset)---> IDLE
enum RallyState {
  /// No rally in progress. Waiting for ball detection with sufficient velocity.
  idle,

  /// Rally is actively in progress. Ball is being tracked across frames.
  emJogo,

  /// Rally has ended. Clip extraction is triggered and buffer is reset.
  fimRally,
}

/// Represents a single rally detected during a match recording session.
class Rally {
  /// Unique identifier for this rally.
  final String id;

  /// Identifier of the match this rally belongs to.
  final int matchId;

  /// Sequential rally number within the match (1-based).
  final int rallyNumber;

  /// Timestamp when the rally started (ball first detected in motion).
  final DateTime startTime;

  /// Timestamp when the rally ended (ground touch, out of bounds, or timeout).
  /// Null while rally is still in progress.
  final DateTime? endTime;

  /// Current state of this rally.
  final RallyState state;

  /// Number of times the ball crossed the net during this rally.
  final int netCrossings;

  /// Whether the ball was detected touching the ground to end the rally.
  final bool endedByGroundTouch;

  /// Whether the rally ended due to ball going out of bounds.
  final bool endedByOutOfBounds;

  /// Whether the rally ended due to detection timeout.
  final bool endedByTimeout;

  const Rally({
    required this.id,
    required this.matchId,
    required this.rallyNumber,
    required this.startTime,
    this.endTime,
    this.state = RallyState.idle,
    this.netCrossings = 0,
    this.endedByGroundTouch = false,
    this.endedByOutOfBounds = false,
    this.endedByTimeout = false,
  });

  /// Duration of the rally. Returns null if rally is still in progress.
  Duration? get duration {
    if (endTime == null) return null;
    return endTime!.difference(startTime);
  }

  /// Whether this rally is currently active.
  bool get isActive => state == RallyState.emJogo;

  /// Whether this rally has completed.
  bool get isCompleted => state == RallyState.fimRally;

  /// Creates a copy of this rally with updated fields.
  Rally copyWith({
    String? id,
    int? matchId,
    int? rallyNumber,
    DateTime? startTime,
    DateTime? endTime,
    RallyState? state,
    int? netCrossings,
    bool? endedByGroundTouch,
    bool? endedByOutOfBounds,
    bool? endedByTimeout,
  }) {
    return Rally(
      id: id ?? this.id,
      matchId: matchId ?? this.matchId,
      rallyNumber: rallyNumber ?? this.rallyNumber,
      startTime: startTime ?? this.startTime,
      endTime: endTime ?? this.endTime,
      state: state ?? this.state,
      netCrossings: netCrossings ?? this.netCrossings,
      endedByGroundTouch: endedByGroundTouch ?? this.endedByGroundTouch,
      endedByOutOfBounds: endedByOutOfBounds ?? this.endedByOutOfBounds,
      endedByTimeout: endedByTimeout ?? this.endedByTimeout,
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'matchId': matchId,
        'rallyNumber': rallyNumber,
        'startTime': startTime.toIso8601String(),
        'endTime': endTime?.toIso8601String(),
        'state': state.name,
        'netCrossings': netCrossings,
        'endedByGroundTouch': endedByGroundTouch,
        'endedByOutOfBounds': endedByOutOfBounds,
        'endedByTimeout': endedByTimeout,
      };

  factory Rally.fromJson(Map<String, dynamic> json) => Rally(
        id: json['id'] as String,
        matchId: json['matchId'] as int,
        rallyNumber: json['rallyNumber'] as int,
        startTime: DateTime.parse(json['startTime'] as String),
        endTime: json['endTime'] != null
            ? DateTime.parse(json['endTime'] as String)
            : null,
        state: RallyState.values.byName(json['state'] as String),
        netCrossings: json['netCrossings'] as int? ?? 0,
        endedByGroundTouch: json['endedByGroundTouch'] as bool? ?? false,
        endedByOutOfBounds: json['endedByOutOfBounds'] as bool? ?? false,
        endedByTimeout: json['endedByTimeout'] as bool? ?? false,
      );

  @override
  String toString() =>
      'Rally(#$rallyNumber match=$matchId state=${state.name} '
      'crossings=$netCrossings)';
}

/// Event emitted when a rally completes, used to trigger clip extraction.
///
/// This is the contract between AGENT-01 (RallyController) and AGENT-03
/// (ClipService / clip_processor).
class RallyEvent {
  /// Timestamp marking the start of content to capture (T - preRallySeconds).
  final DateTime startTime;

  /// Timestamp marking the end of content to capture (T + postRallySeconds).
  final DateTime endTime;

  /// Path to the buffer file containing the raw recording.
  final String bufferFilePath;

  /// Match identifier for naming the output clip.
  final int matchId;

  /// Rally number within the match for naming the output clip.
  final int rallyNumber;

  const RallyEvent({
    required this.startTime,
    required this.endTime,
    required this.bufferFilePath,
    required this.matchId,
    required this.rallyNumber,
  });

  /// Duration of the clip to extract.
  Duration get clipDuration => endTime.difference(startTime);

  Map<String, dynamic> toJson() => {
        'startTime': startTime.toIso8601String(),
        'endTime': endTime.toIso8601String(),
        'bufferFilePath': bufferFilePath,
        'matchId': matchId,
        'rallyNumber': rallyNumber,
      };

  @override
  String toString() =>
      'RallyEvent(match=$matchId rally=$rallyNumber '
      'duration=${clipDuration.inSeconds}s)';
}
