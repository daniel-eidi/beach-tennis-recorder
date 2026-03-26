import 'dart:io';
import 'dart:math';
import 'dart:developer' as developer;

import 'package:flutter/services.dart';

import '../models/clip.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

/// Service for trimming video clips around highlight markers using native
/// platform channels.
///
/// On iOS, uses AVFoundation via [VideoTrimPlugin] (AVAssetExportSession with
/// passthrough preset for fast, lossless trimming). Falls back gracefully if
/// the platform channel is unavailable (e.g., on Android where native trimming
/// is not yet implemented).
///
/// [AGENT-01] TASK-01-12: Real video trimming for highlight sharing.
class VideoTrimService {
  static const _channel =
      MethodChannel('com.beachtennis.recorder/video_trim');

  /// Trims a video around a highlight marker.
  ///
  /// Extracts [secondsBefore] seconds before and [secondsAfter] seconds after
  /// the [markerPosition].
  ///
  /// Returns the path to the trimmed video file, or null if trimming failed
  /// (in which case the caller should share the full video with timestamp
  /// context).
  static Future<String?> trimAroundMarker({
    required String videoPath,
    required Duration markerPosition,
    int secondsBefore = 15,
    int secondsAfter = 15,
  }) async {
    final startMs = max(0, markerPosition.inMilliseconds - secondsBefore * 1000);
    final endMs = markerPosition.inMilliseconds + secondsAfter * 1000;

    // Generate a unique output path in the temp directory.
    final tempDir = await getTemporaryDirectory();
    final timestamp = DateTime.now().millisecondsSinceEpoch;
    final inputName = p.basenameWithoutExtension(videoPath);
    final outputPath = p.join(
      tempDir.path,
      'trim_${inputName}_${timestamp}.mp4',
    );

    developer.log(
      'Trimming video: $videoPath '
      'range=${startMs}ms-${endMs}ms '
      'output=$outputPath',
      name: 'VideoTrimService',
    );

    try {
      final result = await _channel.invokeMethod<String>('trimVideo', {
        'inputPath': videoPath,
        'outputPath': outputPath,
        'startMs': startMs,
        'endMs': endMs,
      });

      if (result != null && File(result).existsSync()) {
        final fileSize = File(result).lengthSync();
        developer.log(
          'Trim succeeded: $result (${(fileSize / 1024 / 1024).toStringAsFixed(1)} MB)',
          name: 'VideoTrimService',
        );
        return result;
      }

      developer.log(
        'Trim returned null or file does not exist',
        name: 'VideoTrimService',
      );
      return null;
    } on PlatformException catch (e) {
      developer.log(
        'Platform trim failed: ${e.code} - ${e.message}',
        name: 'VideoTrimService',
      );
      return null;
    } on MissingPluginException {
      // Platform channel not available (e.g., Android, or plugin not registered).
      developer.log(
        'Video trim platform channel not available. '
        'Marker at ${markerPosition.inSeconds}s, '
        'window: -${secondsBefore}s / +${secondsAfter}s',
        name: 'VideoTrimService',
      );
      return null;
    }
  }

  /// Trims a video using a [HighlightMarker] which contains the position
  /// and the user-configured trim window (secondsBefore/secondsAfter).
  static Future<String?> trimHighlight({
    required String videoPath,
    required HighlightMarker marker,
  }) {
    return trimAroundMarker(
      videoPath: videoPath,
      markerPosition: marker.position,
      secondsBefore: marker.secondsBefore,
      secondsAfter: marker.secondsAfter,
    );
  }

  /// Cleans up old trimmed files from the temp directory.
  ///
  /// Call periodically (e.g., on app start) to reclaim disk space.
  static Future<void> cleanupTrimmedFiles() async {
    try {
      final tempDir = await getTemporaryDirectory();
      final dir = Directory(tempDir.path);
      if (!dir.existsSync()) return;

      final trimFiles = dir
          .listSync()
          .whereType<File>()
          .where((f) => p.basename(f.path).startsWith('trim_'))
          .toList();

      for (final file in trimFiles) {
        try {
          final stat = file.statSync();
          // Delete trimmed files older than 1 hour.
          if (DateTime.now().difference(stat.modified).inHours >= 1) {
            file.deleteSync();
            developer.log(
              'Cleaned up old trim file: ${file.path}',
              name: 'VideoTrimService',
            );
          }
        } catch (_) {
          // Ignore individual file errors.
        }
      }
    } catch (e) {
      developer.log(
        'Trim cleanup error: $e',
        name: 'VideoTrimService',
      );
    }
  }

  /// Builds a share message for a highlight at a given position.
  ///
  /// Used when actual trimming is not available or as supplemental text
  /// alongside a trimmed clip. The message includes the timestamp so the
  /// recipient knows where to look in the video.
  static String buildShareMessage({
    required int rallyNumber,
    required Duration markerPosition,
    int? highlightIndex,
  }) {
    final minutes = markerPosition.inMinutes.remainder(60);
    final seconds = markerPosition.inSeconds.remainder(60);
    final timestamp = '${minutes.toString().padLeft(2, '0')}:'
        '${seconds.toString().padLeft(2, '0')}';

    final highlightLabel = highlightIndex != null
        ? 'Highlight $highlightIndex'
        : 'Highlight';

    return 'Beach Tennis $highlightLabel at $timestamp '
        '(Rally #$rallyNumber)';
  }
}
