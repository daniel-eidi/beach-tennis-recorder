import 'dart:developer' as developer;

/// Service for trimming video clips around highlight markers.
///
/// For the MVP, this service provides a share message with the timestamp
/// rather than actual video trimming. Future versions can use FFmpeg or
/// platform-native trimming (AVFoundation on iOS, MediaCodec on Android).
class VideoTrimService {
  /// Trims a video around a highlight marker.
  ///
  /// Extracts [secondsBefore] seconds before and [secondsAfter] seconds after
  /// the marker position.
  ///
  /// Returns the path to the trimmed video file, or null if trimming is not
  /// available (in which case the caller should share the full video with
  /// timestamp context).
  static Future<String?> trimAroundMarker({
    required String videoPath,
    required Duration markerPosition,
    int secondsBefore = 15,
    int secondsAfter = 15,
  }) async {
    // MVP: Video trimming is not yet implemented.
    // Future implementations:
    //   1. Use video_compress package trimmer
    //   2. Use platform MethodChannel for native AVFoundation / MediaCodec
    //   3. Use FFmpeg kit if added to the project
    //
    // For now, return null to signal the caller should share the full video
    // with a timestamp annotation in the share message.
    developer.log(
      'Video trimming not available (MVP). '
      'Marker at ${markerPosition.inSeconds}s, '
      'window: -${secondsBefore}s / +${secondsAfter}s',
      name: 'VideoTrimService',
    );
    return null;
  }

  /// Builds a share message for a highlight at a given position.
  ///
  /// Used when actual trimming is not available. The message includes the
  /// timestamp so the recipient knows where to look in the video.
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
