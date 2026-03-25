import 'dart:async';
import 'dart:io';
import 'dart:developer' as developer;

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'package:intl/intl.dart';
import 'package:ffmpeg_kit_flutter/ffmpeg_kit.dart';
import 'package:ffmpeg_kit_flutter/return_code.dart';
import 'package:uuid/uuid.dart';

import '../models/clip.dart';
import '../models/rally.dart';
import 'buffer_service.dart';

/// Service responsible for extracting and saving rally clips from the buffer.
///
/// When a rally completes, [ClipService] receives a [RallyEvent] and:
/// 1. Extracts the relevant video segment from [BufferService]
/// 2. Saves it with the naming convention: rally_{matchId}_{number}_{date}_{time}.mp4
/// 3. Generates a thumbnail from the middle frame
/// 4. Registers the clip in the local clip library
///
/// Implements TASK-01-07.
class ClipService extends ChangeNotifier {
  final BufferService _bufferService;
  final _uuid = const Uuid();

  /// Pre-rally padding in seconds (capture before the serve).
  static const int preRallySeconds = 3;

  /// Post-rally padding in seconds (capture after rally ends).
  static const int postRallySeconds = 2;

  /// Minimum valid clip duration in seconds.
  static const double minClipDurationSeconds = 1.0;

  /// Maximum clip file size in bytes (200 MB).
  static const int maxClipSizeBytes = 200 * 1024 * 1024;

  Directory? _clipsDir;
  final List<Clip> _clips = [];
  bool _isProcessing = false;

  ClipService({required BufferService bufferService})
      : _bufferService = bufferService;

  /// All saved clips, newest first.
  List<Clip> get clips => List.unmodifiable(_clips);

  /// Whether a clip is currently being extracted/processed.
  bool get isProcessing => _isProcessing;

  /// Clips filtered by match ID.
  List<Clip> clipsForMatch(int matchId) =>
      _clips.where((c) => c.matchId == matchId).toList();

  /// Number of clips for a given match.
  int clipCountForMatch(int matchId) =>
      _clips.where((c) => c.matchId == matchId).length;

  /// Initializes the clips storage directory.
  Future<void> initialize() async {
    final appDir = await getApplicationDocumentsDirectory();
    _clipsDir = Directory(p.join(appDir.path, 'bt_clips'));

    if (!await _clipsDir!.exists()) {
      await _clipsDir!.create(recursive: true);
    }

    // Load existing clips from disk.
    await _loadExistingClips();
    _log('info', 'ClipService initialized: ${_clips.length} clips found');
  }

  /// Extracts a clip from the buffer based on a rally event.
  ///
  /// Applies pre-rally and post-rally padding to capture context around the
  /// actual rally. Returns the created [Clip] or null if extraction fails.
  Future<Clip?> extractClip(RallyEvent event) async {
    if (_clipsDir == null) {
      await initialize();
    }

    _isProcessing = true;
    notifyListeners();

    try {
      // Apply padding to capture pre-serve and post-rally context.
      final paddedStart = event.startTime.subtract(
        const Duration(seconds: preRallySeconds),
      );
      final paddedEnd = event.endTime.add(
        const Duration(seconds: postRallySeconds),
      );

      _log('info', 'Extracting clip for rally ${event.rallyNumber} '
          'of match ${event.matchId}');

      // Extract the segment from the buffer.
      final extractedFile = await _bufferService.extractSegment(
        paddedStart,
        paddedEnd,
      );

      if (extractedFile == null) {
        _log('error', 'Buffer extraction returned null');
        return null;
      }

      // Generate the output filename.
      final fileName = _generateFileName(
        event.matchId,
        event.rallyNumber,
        event.startTime,
      );
      final outputPath = p.join(_clipsDir!.path, fileName);

      // Move the extracted file to the clips directory.
      final clipFile = await extractedFile.rename(outputPath);
      final fileSize = await clipFile.length();

      // Validate the clip.
      if (fileSize > maxClipSizeBytes) {
        _log('warn', 'Clip exceeds max size: ${fileSize ~/ (1024 * 1024)} MB');
      }

      // Get clip duration using FFmpeg probe.
      final duration = await _probeClipDuration(outputPath);
      if (duration != null && duration < minClipDurationSeconds) {
        _log('warn', 'Clip too short: ${duration}s (min ${minClipDurationSeconds}s)');
        await clipFile.delete();
        return null;
      }

      // Generate thumbnail from middle frame.
      final thumbnailPath = await _generateThumbnail(outputPath);

      // Create the clip model.
      final clip = Clip(
        id: _uuid.v4(),
        matchId: event.matchId,
        rallyNumber: event.rallyNumber,
        filePath: outputPath,
        thumbnailPath: thumbnailPath,
        durationSeconds: duration ?? paddedEnd.difference(paddedStart).inMilliseconds / 1000.0,
        fileSizeBytes: fileSize,
        createdAt: DateTime.now(),
      );

      _clips.insert(0, clip);
      _log('info', 'Clip saved: $fileName (${clip.durationFormatted})');

      return clip;
    } catch (e) {
      _log('error', 'Clip extraction failed: $e');
      return null;
    } finally {
      _isProcessing = false;
      notifyListeners();
    }
  }

  /// Extracts a highlight clip from the buffer based on a gesture trigger.
  ///
  /// Captures the last [durationSeconds] of video from the buffer ending at
  /// [triggerTime]. Returns the created [Clip] or null if extraction fails.
  ///
  /// Highlights are stored alongside rally clips but marked with
  /// [ClipType.highlight]. Naming convention:
  /// highlight_{matchId}_{number}_{date}_{time}.mp4
  Future<Clip?> saveHighlight({
    required DateTime triggerTime,
    required int durationSeconds,
    required int matchId,
  }) async {
    if (_clipsDir == null) {
      await initialize();
    }

    _isProcessing = true;
    notifyListeners();

    try {
      final startTime = triggerTime.subtract(
        Duration(seconds: durationSeconds),
      );

      _log('info', 'Saving highlight: last ${durationSeconds}s '
          'for match $matchId');

      // Extract the segment from the buffer.
      final extractedFile = await _bufferService.extractSegment(
        startTime,
        triggerTime,
      );

      if (extractedFile == null) {
        _log('error', 'Buffer extraction returned null for highlight');
        return null;
      }

      // Count existing highlights for this match for numbering.
      final highlightCount = _clips
          .where((c) => c.matchId == matchId && c.clipType == ClipType.highlight)
          .length;
      final highlightNumber = highlightCount + 1;

      // Generate the output filename.
      final fileName = _generateHighlightFileName(
        matchId,
        highlightNumber,
        triggerTime,
      );
      final outputPath = p.join(_clipsDir!.path, fileName);

      // Move the extracted file to the clips directory.
      final clipFile = await extractedFile.rename(outputPath);
      final fileSize = await clipFile.length();

      // Validate the clip.
      if (fileSize > maxClipSizeBytes) {
        _log('warn', 'Highlight exceeds max size: '
            '${fileSize ~/ (1024 * 1024)} MB');
      }

      // Get clip duration using FFmpeg probe.
      final duration = await _probeClipDuration(outputPath);
      if (duration != null && duration < minClipDurationSeconds) {
        _log('warn', 'Highlight too short: ${duration}s');
        await clipFile.delete();
        return null;
      }

      // Generate thumbnail from middle frame.
      final thumbnailPath = await _generateThumbnail(outputPath);

      // Create the clip model.
      final clip = Clip(
        id: _uuid.v4(),
        matchId: matchId,
        rallyNumber: highlightNumber,
        filePath: outputPath,
        thumbnailPath: thumbnailPath,
        durationSeconds: duration ??
            triggerTime.difference(startTime).inMilliseconds / 1000.0,
        fileSizeBytes: fileSize,
        createdAt: DateTime.now(),
        clipType: ClipType.highlight,
      );

      _clips.insert(0, clip);
      _log('info', 'Highlight saved: $fileName (${clip.durationFormatted})');

      return clip;
    } catch (e) {
      _log('error', 'Highlight extraction failed: $e');
      return null;
    } finally {
      _isProcessing = false;
      notifyListeners();
    }
  }

  /// Deletes a clip from disk and the local library.
  Future<bool> deleteClip(String clipId) async {
    final index = _clips.indexWhere((c) => c.id == clipId);
    if (index == -1) return false;

    final clip = _clips[index];

    try {
      final file = File(clip.filePath);
      if (await file.exists()) {
        await file.delete();
      }

      if (clip.thumbnailPath != null) {
        final thumb = File(clip.thumbnailPath!);
        if (await thumb.exists()) {
          await thumb.delete();
        }
      }

      _clips.removeAt(index);
      notifyListeners();
      _log('info', 'Deleted clip: ${clip.fileName}');
      return true;
    } catch (e) {
      _log('error', 'Failed to delete clip: $e');
      return false;
    }
  }

  /// Marks a clip as uploaded with its remote URL.
  void markAsUploaded(String clipId, String remoteUrl) {
    final index = _clips.indexWhere((c) => c.id == clipId);
    if (index == -1) return;

    _clips[index] = _clips[index].copyWith(
      isUploaded: true,
      remoteUrl: remoteUrl,
    );
    notifyListeners();
  }

  // --- Private helpers ---

  /// Generates a filename following the AGENT-03 naming convention.
  /// Format: rally_{matchId}_{number}_{date}_{time}.mp4
  String _generateFileName(int matchId, int rallyNumber, DateTime timestamp) {
    final dateFmt = DateFormat('yyyyMMdd').format(timestamp);
    final timeFmt = DateFormat('HHmmss').format(timestamp);
    return 'rally_${matchId}_${rallyNumber}_${dateFmt}_$timeFmt.mp4';
  }

  /// Generates a filename for highlight clips.
  /// Format: highlight_{matchId}_{number}_{date}_{time}.mp4
  String _generateHighlightFileName(
    int matchId,
    int highlightNumber,
    DateTime timestamp,
  ) {
    final dateFmt = DateFormat('yyyyMMdd').format(timestamp);
    final timeFmt = DateFormat('HHmmss').format(timestamp);
    return 'highlight_${matchId}_${highlightNumber}_${dateFmt}_$timeFmt.mp4';
  }

  /// Generates a thumbnail image from the middle frame of a clip.
  ///
  /// Uses FFmpeg to extract a single frame at the midpoint of the video.
  /// Returns the thumbnail file path, or null if generation fails.
  Future<String?> _generateThumbnail(String clipPath) async {
    final thumbnailPath = clipPath.replaceAll('.mp4', '_thumb.jpg');

    // First, probe the duration to find the midpoint.
    final duration = await _probeClipDuration(clipPath);
    final seekTo = duration != null ? (duration / 2).toStringAsFixed(3) : '1';

    final command = '-y '
        '-ss $seekTo '
        '-i "$clipPath" '
        '-vframes 1 '
        '-q:v 3 '
        '"$thumbnailPath"';

    try {
      final session = await FFmpegKit.execute(command);
      final returnCode = await session.getReturnCode();

      if (ReturnCode.isSuccess(returnCode)) {
        return thumbnailPath;
      }
    } catch (e) {
      _log('warn', 'Thumbnail generation failed: $e');
    }

    return null;
  }

  /// Probes a video file to get its duration in seconds.
  Future<double?> _probeClipDuration(String filePath) async {
    try {
      // Use FFmpeg to get duration via a quick analysis pass.
      final command = '-i "$filePath" -f null -';
      final session = await FFmpegKit.execute(command);
      final logs = await session.getLogsAsString();

      // Parse duration from FFmpeg output: "Duration: HH:MM:SS.ms"
      final durationRegex = RegExp(r'Duration:\s*(\d+):(\d+):(\d+)\.(\d+)');
      final match = durationRegex.firstMatch(logs ?? '');

      if (match != null) {
        final hours = int.parse(match.group(1)!);
        final minutes = int.parse(match.group(2)!);
        final seconds = int.parse(match.group(3)!);
        final centiseconds = int.parse(match.group(4)!);
        return hours * 3600.0 +
            minutes * 60.0 +
            seconds +
            centiseconds / 100.0;
      }
    } catch (e) {
      _log('warn', 'Duration probe failed: $e');
    }
    return null;
  }

  /// Scans the clips directory for existing clip files.
  Future<void> _loadExistingClips() async {
    if (_clipsDir == null) return;

    try {
      final entities = await _clipsDir!.list().toList();
      for (final entity in entities) {
        if (entity is File && entity.path.endsWith('.mp4')) {
          final fileName = p.basename(entity.path);
          final parsed = _parseFileName(fileName);
          if (parsed == null) continue;

          final fileSize = await entity.length();
          final thumbnailPath = entity.path.replaceAll('.mp4', '_thumb.jpg');
          final thumbExists = await File(thumbnailPath).exists();

          _clips.add(Clip(
            id: _uuid.v4(),
            matchId: parsed['matchId'] as int,
            rallyNumber: parsed['rallyNumber'] as int,
            filePath: entity.path,
            thumbnailPath: thumbExists ? thumbnailPath : null,
            durationSeconds: 0, // Will be probed lazily if needed.
            fileSizeBytes: fileSize,
            createdAt: entity.statSync().modified,
            clipType: parsed['clipType'] as ClipType? ?? ClipType.rally,
          ));
        }
      }

      // Sort newest first.
      _clips.sort((a, b) => b.createdAt.compareTo(a.createdAt));
    } catch (e) {
      _log('warn', 'Failed to load existing clips: $e');
    }
  }

  /// Parses a clip filename back into its component parts.
  /// Expected formats:
  ///   rally_{matchId}_{number}_{date}_{time}.mp4
  ///   highlight_{matchId}_{number}_{date}_{time}.mp4
  Map<String, dynamic>? _parseFileName(String fileName) {
    final regex = RegExp(r'(rally|highlight)_(\d+)_(\d+)_(\d{8})_(\d{6})\.mp4');
    final match = regex.firstMatch(fileName);
    if (match == null) return null;

    return {
      'clipType': match.group(1) == 'highlight'
          ? ClipType.highlight
          : ClipType.rally,
      'matchId': int.parse(match.group(2)!),
      'rallyNumber': int.parse(match.group(3)!),
      'date': match.group(4),
      'time': match.group(5),
    };
  }

  void _log(String level, String message) {
    developer.log(
      message,
      name: 'ClipService',
      level: level == 'error' ? 1000 : (level == 'warn' ? 900 : 800),
    );
  }

  @override
  void dispose() {
    super.dispose();
  }
}
