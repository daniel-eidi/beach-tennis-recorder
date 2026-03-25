import 'dart:async';
import 'dart:io';
import 'dart:developer' as developer;

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'package:ffmpeg_kit_flutter/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter/return_code.dart';

/// Disk-based circular video buffer for continuous recording.
///
/// Records video in fixed-duration segments written to temporary storage.
/// Maintains a rolling window of [maxBufferSeconds] by deleting old segments
/// as new ones are created. This approach keeps memory usage constant
/// regardless of recording duration, making it safe for devices with limited
/// RAM.
///
/// The buffer is designed for TASK-01-03: circular buffer of 60 seconds on
/// disk, not in memory.
///
/// Segment lifecycle:
///   1. Camera writes to current segment file
///   2. When segment reaches [segmentDurationSeconds], a new segment starts
///   3. Segments older than [maxBufferSeconds] are deleted from disk
///   4. [extractSegment] uses FFmpeg to concatenate and trim relevant segments
class BufferService extends ChangeNotifier {
  /// Total buffer duration in seconds. Segments older than this are purged.
  static const int maxBufferSeconds = 60;

  /// Duration of each individual segment file in seconds.
  static const int segmentDurationSeconds = 10;

  /// Maximum number of segments kept on disk.
  static const int maxSegments = maxBufferSeconds ~/ segmentDurationSeconds;

  final List<_BufferSegment> _segments = [];
  Directory? _bufferDir;
  bool _isBuffering = false;
  Timer? _cleanupTimer;
  int _segmentCounter = 0;

  /// Whether the buffer is currently active and recording segments.
  bool get isBuffering => _isBuffering;

  /// Number of segments currently stored on disk.
  int get segmentCount => _segments.length;

  /// Total buffered duration available for extraction.
  Duration get bufferedDuration {
    if (_segments.isEmpty) return Duration.zero;
    final oldest = _segments.first.startTime;
    final newest = _segments.last.endTime ?? DateTime.now();
    return newest.difference(oldest);
  }

  /// Path to the buffer directory on disk.
  String? get bufferDirectoryPath => _bufferDir?.path;

  /// Initializes the buffer directory in the app's temporary storage.
  ///
  /// Must be called before [startBuffering]. Creates a dedicated subdirectory
  /// that is cleaned up on [stopBuffering].
  Future<void> initialize() async {
    final tempDir = await getTemporaryDirectory();
    _bufferDir = Directory(p.join(tempDir.path, 'bt_buffer'));

    if (await _bufferDir!.exists()) {
      // Clean up any stale segments from a previous session.
      await _bufferDir!.delete(recursive: true);
    }
    await _bufferDir!.create(recursive: true);

    _log('info', 'Buffer directory initialized: ${_bufferDir!.path}');
  }

  /// Starts the circular buffering process.
  ///
  /// Returns the path where the camera should write the current segment.
  /// The caller (typically [CameraService]) is responsible for writing video
  /// data to this path and calling [rotateSegment] when each segment is full.
  Future<String> startBuffering() async {
    if (_bufferDir == null) {
      await initialize();
    }

    _isBuffering = true;
    _segments.clear();
    _segmentCounter = 0;

    // Start periodic cleanup of old segments.
    _cleanupTimer = Timer.periodic(
      const Duration(seconds: 5),
      (_) => _purgeOldSegments(),
    );

    final firstSegmentPath = _createSegmentPath();
    _segments.add(_BufferSegment(
      filePath: firstSegmentPath,
      startTime: DateTime.now(),
      index: _segmentCounter,
    ));

    _log('info', 'Buffering started');
    notifyListeners();
    return firstSegmentPath;
  }

  /// Notifies the buffer that the current segment is complete and provides
  /// the path for the next segment.
  ///
  /// Call this when the camera finishes writing [segmentDurationSeconds] of
  /// video to the current segment file. Returns the file path for the next
  /// segment.
  String rotateSegment() {
    if (!_isBuffering) {
      throw StateError('Cannot rotate segment: buffering not active');
    }

    // Mark the current segment as complete.
    if (_segments.isNotEmpty) {
      _segments.last.endTime = DateTime.now();
    }

    // Create a new segment.
    _segmentCounter++;
    final newPath = _createSegmentPath();
    _segments.add(_BufferSegment(
      filePath: newPath,
      startTime: DateTime.now(),
      index: _segmentCounter,
    ));

    _purgeOldSegments();
    _log('info', 'Rotated to segment $_segmentCounter '
        '(${_segments.length} segments on disk)');
    notifyListeners();
    return newPath;
  }

  /// Extracts a video segment from the buffer between [startTime] and
  /// [endTime].
  ///
  /// Uses FFmpeg to concatenate the relevant segment files and trim to the
  /// exact time range requested. Returns the path to the extracted clip file,
  /// or null if extraction fails.
  ///
  /// This is the primary interface used by [ClipService] to create rally clips.
  Future<File?> extractSegment(
    DateTime startTime,
    DateTime endTime,
  ) async {
    if (_segments.isEmpty) {
      _log('warn', 'Cannot extract: no segments in buffer');
      return null;
    }

    // Find segments that overlap with the requested time range.
    final relevantSegments = _segments.where((seg) {
      final segEnd = seg.endTime ?? DateTime.now();
      return seg.startTime.isBefore(endTime) && segEnd.isAfter(startTime);
    }).toList();

    if (relevantSegments.isEmpty) {
      _log('warn', 'No segments overlap with requested time range');
      return null;
    }

    // Verify all segment files exist on disk.
    final existingSegments = <_BufferSegment>[];
    for (final seg in relevantSegments) {
      if (await File(seg.filePath).exists()) {
        existingSegments.add(seg);
      } else {
        _log('warn', 'Segment file missing: ${seg.filePath}');
      }
    }

    if (existingSegments.isEmpty) {
      _log('error', 'All relevant segment files are missing from disk');
      return null;
    }

    try {
      final outputPath = p.join(
        _bufferDir!.path,
        'extract_${DateTime.now().millisecondsSinceEpoch}.mp4',
      );

      if (existingSegments.length == 1) {
        // Single segment: trim directly.
        return await _trimSingleSegment(
          existingSegments.first,
          startTime,
          endTime,
          outputPath,
        );
      } else {
        // Multiple segments: concatenate then trim.
        return await _concatAndTrim(
          existingSegments,
          startTime,
          endTime,
          outputPath,
        );
      }
    } catch (e) {
      _log('error', 'Segment extraction failed: $e');
      return null;
    }
  }

  /// Stops buffering and optionally cleans up all segment files.
  ///
  /// If [keepFiles] is true, segment files remain on disk for final
  /// extraction. Otherwise, the entire buffer directory is purged.
  Future<void> stopBuffering({bool keepFiles = false}) async {
    _isBuffering = false;
    _cleanupTimer?.cancel();
    _cleanupTimer = null;

    if (!keepFiles && _bufferDir != null) {
      try {
        if (await _bufferDir!.exists()) {
          await _bufferDir!.delete(recursive: true);
          await _bufferDir!.create(recursive: true);
        }
      } catch (e) {
        _log('warn', 'Failed to clean buffer directory: $e');
      }
      _segments.clear();
    }

    _log('info', 'Buffering stopped (keepFiles=$keepFiles)');
    notifyListeners();
  }

  /// Returns the file path for the currently active segment.
  String? get currentSegmentPath {
    if (_segments.isEmpty) return null;
    return _segments.last.filePath;
  }

  // --- Private helpers ---

  String _createSegmentPath() {
    return p.join(
      _bufferDir!.path,
      'seg_${_segmentCounter.toString().padLeft(4, '0')}.mp4',
    );
  }

  /// Removes segments that are older than [maxBufferSeconds].
  void _purgeOldSegments() {
    if (_segments.length <= maxSegments) return;

    final cutoff = DateTime.now().subtract(
      const Duration(seconds: maxBufferSeconds),
    );

    final toRemove = <_BufferSegment>[];
    for (final seg in _segments) {
      final segEnd = seg.endTime ?? DateTime.now();
      if (segEnd.isBefore(cutoff)) {
        toRemove.add(seg);
      }
    }

    for (final seg in toRemove) {
      _segments.remove(seg);
      // Delete file asynchronously; don't block the rotation.
      File(seg.filePath).delete().catchError((e) {
        _log('warn', 'Failed to delete old segment: ${seg.filePath}');
        return File(seg.filePath);
      });
    }

    if (toRemove.isNotEmpty) {
      _log('info', 'Purged ${toRemove.length} old segments');
    }
  }

  /// Trims a single segment file to the requested time range using FFmpeg.
  Future<File?> _trimSingleSegment(
    _BufferSegment segment,
    DateTime startTime,
    DateTime endTime,
    String outputPath,
  ) async {
    final segStart = segment.startTime;
    final offsetStart = startTime.difference(segStart).inMilliseconds / 1000.0;
    final duration = endTime.difference(startTime).inMilliseconds / 1000.0;

    // Use -ss for seeking and -t for duration. -c copy avoids re-encoding.
    final command = '-y '
        '-ss ${offsetStart.toStringAsFixed(3)} '
        '-i "${segment.filePath}" '
        '-t ${duration.toStringAsFixed(3)} '
        '-c copy '
        '"$outputPath"';

    _log('info', 'FFmpeg trim: $command');
    final session = await FFmpegKit.execute(command);
    final returnCode = await session.getReturnCode();

    if (ReturnCode.isSuccess(returnCode)) {
      _log('info', 'Segment extracted: $outputPath');
      return File(outputPath);
    } else {
      final logs = await session.getLogsAsString();
      _log('error', 'FFmpeg trim failed: $logs');
      return null;
    }
  }

  /// Concatenates multiple segments and trims to the requested time range.
  Future<File?> _concatAndTrim(
    List<_BufferSegment> segments,
    DateTime startTime,
    DateTime endTime,
    String outputPath,
  ) async {
    // Create a concat list file for FFmpeg.
    final concatListPath = p.join(_bufferDir!.path, 'concat_list.txt');
    final concatFile = File(concatListPath);
    final concatContent = segments
        .map((seg) => "file '${seg.filePath}'")
        .join('\n');
    await concatFile.writeAsString(concatContent);

    // Calculate the offset from the first segment's start time.
    final firstSegStart = segments.first.startTime;
    final offsetStart =
        startTime.difference(firstSegStart).inMilliseconds / 1000.0;
    final duration = endTime.difference(startTime).inMilliseconds / 1000.0;

    final command = '-y '
        '-f concat -safe 0 -i "$concatListPath" '
        '-ss ${offsetStart.toStringAsFixed(3)} '
        '-t ${duration.toStringAsFixed(3)} '
        '-c copy '
        '"$outputPath"';

    _log('info', 'FFmpeg concat+trim: $command');
    final session = await FFmpegKit.execute(command);
    final returnCode = await session.getReturnCode();

    // Clean up the concat list file.
    await concatFile.delete().catchError((_) => concatFile);

    if (ReturnCode.isSuccess(returnCode)) {
      _log('info', 'Concatenated segment extracted: $outputPath');
      return File(outputPath);
    } else {
      final logs = await session.getLogsAsString();
      _log('error', 'FFmpeg concat failed: $logs');
      return null;
    }
  }

  void _log(String level, String message) {
    developer.log(
      message,
      name: 'BufferService',
      level: level == 'error' ? 1000 : (level == 'warn' ? 900 : 800),
    );
  }

  @override
  void dispose() {
    _cleanupTimer?.cancel();
    super.dispose();
  }
}

/// Internal representation of a single video segment on disk.
class _BufferSegment {
  final String filePath;
  final DateTime startTime;
  DateTime? endTime;
  final int index;

  _BufferSegment({
    required this.filePath,
    required this.startTime,
    this.endTime,
    required this.index,
  });

  @override
  String toString() => 'Segment($index @ $filePath)';
}
