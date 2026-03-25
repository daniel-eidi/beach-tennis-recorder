import 'package:intl/intl.dart';

/// Represents a beach tennis match recording session.
///
/// Matches are created when the user starts recording and serve as the
/// top-level container for rally clips. Stored locally as JSON files
/// for offline-first operation.
class Match {
  /// Unique identifier for this match (timestamp-based).
  final int id;

  /// User-facing label for this match.
  final String name;

  /// When this match was created.
  final DateTime createdAt;

  /// Number of rally clips captured so far.
  final int clipCount;

  /// Total duration of all clips in seconds.
  final double totalDurationSeconds;

  const Match({
    required this.id,
    required this.name,
    required this.createdAt,
    this.clipCount = 0,
    this.totalDurationSeconds = 0.0,
  });

  /// Creates a new match with an auto-generated name from the current time.
  ///
  /// Name format: "Match 25 Mar 14:30"
  factory Match.create() {
    final now = DateTime.now();
    final id = now.millisecondsSinceEpoch ~/ 1000;
    final name = 'Match ${DateFormat('dd MMM HH:mm').format(now)}';
    return Match(
      id: id,
      name: name,
      createdAt: now,
    );
  }

  /// Total duration formatted as "Xm Ys".
  String get totalDurationFormatted {
    final totalSeconds = totalDurationSeconds.round();
    final minutes = totalSeconds ~/ 60;
    final seconds = totalSeconds % 60;
    if (minutes > 0) {
      return '${minutes}m ${seconds}s';
    }
    return '${seconds}s';
  }

  /// Creates a copy with updated fields.
  Match copyWith({
    int? id,
    String? name,
    DateTime? createdAt,
    int? clipCount,
    double? totalDurationSeconds,
  }) {
    return Match(
      id: id ?? this.id,
      name: name ?? this.name,
      createdAt: createdAt ?? this.createdAt,
      clipCount: clipCount ?? this.clipCount,
      totalDurationSeconds: totalDurationSeconds ?? this.totalDurationSeconds,
    );
  }

  Map<String, dynamic> toJson() => {
        'id': id,
        'name': name,
        'createdAt': createdAt.toIso8601String(),
        'clipCount': clipCount,
        'totalDurationSeconds': totalDurationSeconds,
      };

  factory Match.fromJson(Map<String, dynamic> json) => Match(
        id: json['id'] as int,
        name: json['name'] as String,
        createdAt: DateTime.parse(json['createdAt'] as String),
        clipCount: json['clipCount'] as int? ?? 0,
        totalDurationSeconds:
            (json['totalDurationSeconds'] as num?)?.toDouble() ?? 0.0,
      );

  @override
  String toString() => 'Match(#$id "$name" clips=$clipCount)';
}
