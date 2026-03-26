import 'dart:io';

import 'package:flutter/material.dart';
import 'package:intl/intl.dart';
import 'package:provider/provider.dart';
import 'package:share_plus/share_plus.dart';
import 'package:video_player/video_player.dart';

import '../models/clip.dart';
import '../services/clip_service.dart';
import '../services/video_trim_service.dart';

/// Screen that displays all saved rally clips grouped by match.
///
/// Features:
/// - Clips listed newest first, grouped by match ID
/// - Each clip shows thumbnail, rally number, duration, file size, timestamp
/// - Tap to play with enhanced inline player
/// - Playback speed control (0.5x, 1x, 1.5x, 2x)
/// - Frame-by-frame step (forward/back)
/// - Share via system share sheet
/// - Export all clips for a match
/// - Swipe to delete with confirmation
///
/// Implements TASK-01-10, TASK-01-11, TASK-01-12.
class LibraryScreen extends StatefulWidget {
  const LibraryScreen({super.key});

  @override
  State<LibraryScreen> createState() => _LibraryScreenState();
}

class _LibraryScreenState extends State<LibraryScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<ClipService>().initialize();
    });
  }

  @override
  Widget build(BuildContext context) {
    final clipService = context.watch<ClipService>();
    final clips = clipService.clips;

    // Group clips by match ID.
    final grouped = <int, List<Clip>>{};
    for (final clip in clips) {
      grouped.putIfAbsent(clip.matchId, () => []).add(clip);
    }

    // Sort match IDs descending (newest first).
    final matchIds = grouped.keys.toList()..sort((a, b) => b.compareTo(a));

    return Scaffold(
      appBar: AppBar(
        title: const Text('Clip Library'),
        actions: [
          if (clips.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(right: 16),
              child: Center(
                child: Text(
                  '${clips.length} clips',
                  style: const TextStyle(color: Colors.white54),
                ),
              ),
            ),
        ],
      ),
      body: clips.isEmpty
          ? _buildEmptyState()
          : ListView.builder(
              padding: const EdgeInsets.symmetric(vertical: 8),
              itemCount: matchIds.length,
              itemBuilder: (context, index) {
                final matchId = matchIds[index];
                final matchClips = grouped[matchId]!;
                return _MatchClipGroup(
                  matchId: matchId,
                  clips: matchClips,
                );
              },
            ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            Icons.video_library_outlined,
            size: 64,
            color: Colors.white.withOpacity(0.3),
          ),
          const SizedBox(height: 16),
          Text(
            'No clips yet',
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
                  color: Colors.white54,
                ),
          ),
          const SizedBox(height: 8),
          const Text(
            'Start recording a match to capture rallies',
            style: TextStyle(color: Colors.white38),
          ),
        ],
      ),
    );
  }
}

/// A group of clips belonging to the same match.
class _MatchClipGroup extends StatelessWidget {
  final int matchId;
  final List<Clip> clips;

  const _MatchClipGroup({
    required this.matchId,
    required this.clips,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
          child: Row(
            children: [
              const Icon(Icons.sports_tennis, size: 18, color: Colors.white54),
              const SizedBox(width: 8),
              Text(
                'Match #$matchId',
                style: Theme.of(context).textTheme.titleSmall?.copyWith(
                      color: Colors.white54,
                      letterSpacing: 0.5,
                    ),
              ),
              const Spacer(),
              Text(
                '${clips.length} rallies',
                style: const TextStyle(
                  color: Colors.white38,
                  fontSize: 12,
                ),
              ),
              const SizedBox(width: 8),
              // Export all clips for this match.
              IconButton(
                icon: const Icon(Icons.ios_share, size: 18),
                tooltip: 'Export all clips',
                color: Colors.white54,
                onPressed: () => _exportAllClips(context),
              ),
            ],
          ),
        ),
        ...clips.map((clip) => _ClipTile(clip: clip)),
        const Divider(height: 1, indent: 16, endIndent: 16),
      ],
    );
  }

  void _exportAllClips(BuildContext context) {
    final files = clips
        .where((c) => File(c.filePath).existsSync())
        .map((c) => XFile(c.filePath))
        .toList();

    if (files.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No clip files found on disk')),
      );
      return;
    }

    Share.shareXFiles(
      files,
      text: 'Beach Tennis Match #$matchId - ${files.length} rallies',
    );
  }
}

/// A single clip tile showing thumbnail, metadata, and actions.
class _ClipTile extends StatelessWidget {
  final Clip clip;

  const _ClipTile({required this.clip});

  @override
  Widget build(BuildContext context) {
    final dateFormat = DateFormat('MMM dd, HH:mm');

    return Dismissible(
      key: Key(clip.id),
      direction: DismissDirection.endToStart,
      background: Container(
        color: Colors.red.shade900,
        alignment: Alignment.centerRight,
        padding: const EdgeInsets.only(right: 24),
        child: const Icon(Icons.delete, color: Colors.white),
      ),
      confirmDismiss: (_) => _confirmDelete(context),
      onDismissed: (_) {
        context.read<ClipService>().deleteClip(clip.id);
      },
      child: ListTile(
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
        leading: _buildThumbnail(),
        title: Text(
          'Rally #${clip.rallyNumber}',
          style: const TextStyle(fontWeight: FontWeight.w600),
        ),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              '${clip.durationFormatted} '
              '${clip.fileSizeMB != null ? "- ${clip.fileSizeMB!.toStringAsFixed(1)} MB" : ""}',
              style: const TextStyle(color: Colors.white54, fontSize: 12),
            ),
            Text(
              dateFormat.format(clip.createdAt),
              style: const TextStyle(color: Colors.white38, fontSize: 11),
            ),
          ],
        ),
        trailing: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (clip.isUploaded)
              const Icon(Icons.cloud_done, color: Colors.green, size: 18),
            const SizedBox(width: 4),
            IconButton(
              icon: const Icon(Icons.share, size: 20),
              onPressed: () => _shareClip(context),
              tooltip: 'Share clip',
            ),
          ],
        ),
        onTap: () => _playClip(context),
      ),
    );
  }

  Widget _buildThumbnail() {
    return ClipRRect(
      borderRadius: BorderRadius.circular(6),
      child: SizedBox(
        width: 80,
        height: 45,
        child: clip.thumbnailPath != null &&
                File(clip.thumbnailPath!).existsSync()
            ? Image.file(
                File(clip.thumbnailPath!),
                fit: BoxFit.cover,
              )
            : Container(
                color: Colors.grey.shade800,
                child: const Center(
                  child: Icon(
                    Icons.play_circle_outline,
                    color: Colors.white38,
                    size: 24,
                  ),
                ),
              ),
      ),
    );
  }

  Future<bool> _confirmDelete(BuildContext context) async {
    return await showDialog<bool>(
          context: context,
          builder: (ctx) => AlertDialog(
            title: const Text('Delete clip?'),
            content: Text(
              'Rally #${clip.rallyNumber} will be permanently deleted.',
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(ctx, false),
                child: const Text('CANCEL'),
              ),
              TextButton(
                onPressed: () => Navigator.pop(ctx, true),
                style: TextButton.styleFrom(foregroundColor: Colors.red),
                child: const Text('DELETE'),
              ),
            ],
          ),
        ) ??
        false;
  }

  void _shareClip(BuildContext context) {
    Share.shareXFiles(
      [XFile(clip.filePath)],
      text: 'Rally #${clip.rallyNumber} - Beach Tennis',
    );
  }

  void _playClip(BuildContext context) {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => _ClipPlayerScreen(clip: clip),
      ),
    );
  }
}

/// Full-screen inline video player with enhanced controls and highlight
/// navigation.
///
/// Features:
/// - Play/pause
/// - Seekable progress bar with highlight markers
/// - Previous/Next highlight navigation buttons
/// - Highlight list below the player with tap-to-seek
/// - Share individual highlights (full video + timestamp message for MVP)
/// - Full video share
/// - Playback speed control (0.5x, 1x, 1.5x, 2x) for rally review
/// - Frame-by-frame stepping (forward/back) using ~33ms steps
/// - Clip metadata display
///
/// Implements TASK-01-11, TASK-01-12.
class _ClipPlayerScreen extends StatefulWidget {
  final Clip clip;

  const _ClipPlayerScreen({required this.clip});

  @override
  State<_ClipPlayerScreen> createState() => _ClipPlayerScreenState();
}

class _ClipPlayerScreenState extends State<_ClipPlayerScreen> {
  late VideoPlayerController _videoController;
  bool _isInitialized = false;
  double _playbackSpeed = 1.0;
  bool _showMetadata = false;

  /// Approximate frame duration at 30fps for frame stepping.
  static const _frameDuration = Duration(milliseconds: 33);

  static const _speedOptions = [0.5, 1.0, 1.5, 2.0];

  /// Sorted list of highlight markers for navigation.
  late final List<Duration> _sortedMarkers;

  @override
  void initState() {
    super.initState();
    _sortedMarkers = List<Duration>.from(widget.clip.highlightMarkers)
      ..sort((a, b) => a.compareTo(b));
    _videoController = VideoPlayerController.file(File(widget.clip.filePath))
      ..initialize().then((_) {
        if (mounted) {
          setState(() => _isInitialized = true);
          _videoController.play();
        }
      });
  }

  @override
  void dispose() {
    _videoController.dispose();
    super.dispose();
  }

  void _stepForward() {
    if (!_isInitialized) return;
    _videoController.pause();
    final current = _videoController.value.position;
    final next = current + _frameDuration;
    final max = _videoController.value.duration;
    _videoController.seekTo(next > max ? max : next);
  }

  void _stepBackward() {
    if (!_isInitialized) return;
    _videoController.pause();
    final current = _videoController.value.position;
    final prev = current - _frameDuration;
    _videoController.seekTo(prev < Duration.zero ? Duration.zero : prev);
  }

  void _cycleSpeed() {
    final currentIndex = _speedOptions.indexOf(_playbackSpeed);
    final nextIndex = (currentIndex + 1) % _speedOptions.length;
    setState(() {
      _playbackSpeed = _speedOptions[nextIndex];
    });
    _videoController.setPlaybackSpeed(_playbackSpeed);
  }

  void _shareClip() {
    Share.shareXFiles(
      [XFile(widget.clip.filePath)],
      text: 'Rally #${widget.clip.rallyNumber} - Beach Tennis',
    );
  }

  void _jumpToNextHighlight() {
    if (!_isInitialized || _sortedMarkers.isEmpty) return;
    final current = _videoController.value.position;
    final next = widget.clip.nextHighlight(current);
    if (next != null) {
      _videoController.seekTo(next);
    }
  }

  void _jumpToPreviousHighlight() {
    if (!_isInitialized || _sortedMarkers.isEmpty) return;
    final current = _videoController.value.position;
    final prev = widget.clip.previousHighlight(current);
    if (prev != null) {
      _videoController.seekTo(prev);
    }
  }

  void _seekToMarker(Duration marker) {
    if (!_isInitialized) return;
    _videoController.seekTo(marker);
  }

  void _shareHighlight(Duration marker, int index) {
    final message = VideoTrimService.buildShareMessage(
      rallyNumber: widget.clip.rallyNumber,
      markerPosition: marker,
      highlightIndex: index + 1,
    );
    Share.shareXFiles(
      [XFile(widget.clip.filePath)],
      text: message,
    );
  }

  @override
  Widget build(BuildContext context) {
    final dateFormat = DateFormat('dd MMM yyyy, HH:mm:ss');
    final hasMarkers = _sortedMarkers.isNotEmpty;

    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        title: Text('Rally #${widget.clip.rallyNumber}'),
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline),
            tooltip: 'Toggle metadata',
            onPressed: () => setState(() => _showMetadata = !_showMetadata),
          ),
          IconButton(
            icon: const Icon(Icons.share),
            tooltip: 'Share full video',
            onPressed: _shareClip,
          ),
        ],
      ),
      body: Column(
        children: [
          // Metadata panel.
          if (_showMetadata)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              color: const Color(0xFF1A1A2E),
              child: Wrap(
                spacing: 24,
                runSpacing: 8,
                children: [
                  _MetadataItem(
                    label: 'Rally',
                    value: '#${widget.clip.rallyNumber}',
                  ),
                  _MetadataItem(
                    label: 'Duration',
                    value: widget.clip.durationFormatted,
                  ),
                  _MetadataItem(
                    label: 'Size',
                    value: widget.clip.fileSizeMB != null
                        ? '${widget.clip.fileSizeMB!.toStringAsFixed(1)} MB'
                        : 'N/A',
                  ),
                  _MetadataItem(
                    label: 'Created',
                    value: dateFormat.format(widget.clip.createdAt),
                  ),
                  _MetadataItem(
                    label: 'Match',
                    value: '#${widget.clip.matchId}',
                  ),
                  _MetadataItem(
                    label: 'Status',
                    value:
                        widget.clip.isUploaded ? 'Uploaded' : 'Local only',
                  ),
                  if (hasMarkers)
                    _MetadataItem(
                      label: 'Highlights',
                      value: '${_sortedMarkers.length}',
                    ),
                ],
              ),
            ),

          // Video player.
          Expanded(
            flex: hasMarkers ? 3 : 1,
            child: Center(
              child: _isInitialized
                  ? AspectRatio(
                      aspectRatio: _videoController.value.aspectRatio,
                      child: Stack(
                        alignment: Alignment.bottomCenter,
                        children: [
                          VideoPlayer(_videoController),
                          _buildControls(),
                        ],
                      ),
                    )
                  : const CircularProgressIndicator(),
            ),
          ),

          // Highlight list below the player.
          if (hasMarkers && _isInitialized)
            Expanded(
              flex: 2,
              child: _buildHighlightList(),
            ),
        ],
      ),
    );
  }

  Widget _buildControls() {
    return ValueListenableBuilder<VideoPlayerValue>(
      valueListenable: _videoController,
      builder: (context, value, child) {
        final hasMarkers = _sortedMarkers.isNotEmpty;

        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.bottomCenter,
              end: Alignment.topCenter,
              colors: [Colors.black87, Colors.transparent],
            ),
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // Progress bar with highlight markers overlay.
              if (hasMarkers && value.duration.inMilliseconds > 0)
                _HighlightProgressBar(
                  controller: _videoController,
                  markers: _sortedMarkers,
                )
              else
                VideoProgressIndicator(
                  _videoController,
                  allowScrubbing: true,
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  colors: const VideoProgressColors(
                    playedColor: Color(0xFF1E88E5),
                    bufferedColor: Colors.white24,
                    backgroundColor: Colors.white12,
                  ),
                ),
              const SizedBox(height: 4),
              // Controls row.
              Row(
                children: [
                  // Previous highlight button.
                  if (hasMarkers)
                    IconButton(
                      icon: const Icon(Icons.star, color: Colors.amber, size: 16),
                      iconSize: 22,
                      tooltip: 'Previous highlight',
                      padding: const EdgeInsets.all(4),
                      constraints: const BoxConstraints(),
                      onPressed: widget.clip.previousHighlight(value.position) != null
                          ? _jumpToPreviousHighlight
                          : null,
                    ),

                  // Frame step backward.
                  IconButton(
                    icon: const Icon(Icons.skip_previous, color: Colors.white),
                    iconSize: 22,
                    tooltip: 'Step back 1 frame',
                    onPressed: _stepBackward,
                  ),
                  // Play/pause.
                  IconButton(
                    icon: Icon(
                      value.isPlaying ? Icons.pause : Icons.play_arrow,
                      color: Colors.white,
                    ),
                    iconSize: 30,
                    onPressed: () {
                      value.isPlaying
                          ? _videoController.pause()
                          : _videoController.play();
                    },
                  ),
                  // Frame step forward.
                  IconButton(
                    icon: const Icon(Icons.skip_next, color: Colors.white),
                    iconSize: 22,
                    tooltip: 'Step forward 1 frame',
                    onPressed: _stepForward,
                  ),

                  // Next highlight button.
                  if (hasMarkers)
                    IconButton(
                      icon: const Icon(Icons.star, color: Colors.amber, size: 16),
                      iconSize: 22,
                      tooltip: 'Next highlight',
                      padding: const EdgeInsets.all(4),
                      constraints: const BoxConstraints(),
                      onPressed: widget.clip.nextHighlight(value.position) != null
                          ? _jumpToNextHighlight
                          : null,
                    ),

                  const SizedBox(width: 8),

                  // Playback speed button.
                  GestureDetector(
                    onTap: _cycleSpeed,
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 4,
                      ),
                      decoration: BoxDecoration(
                        color: _playbackSpeed != 1.0
                            ? const Color(0xFF1E88E5).withOpacity(0.3)
                            : Colors.white12,
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(
                          color: _playbackSpeed != 1.0
                              ? const Color(0xFF1E88E5)
                              : Colors.white24,
                        ),
                      ),
                      child: Text(
                        '${_playbackSpeed}x',
                        style: TextStyle(
                          color: _playbackSpeed != 1.0
                              ? const Color(0xFF1E88E5)
                              : Colors.white70,
                          fontSize: 13,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ),

                  const Spacer(),

                  // Position / duration.
                  Text(
                    '${_formatPosition(value.position)} / '
                    '${_formatPosition(value.duration)}',
                    style: const TextStyle(
                      color: Colors.white70,
                      fontSize: 12,
                      fontFeatures: [FontFeature.tabularFigures()],
                    ),
                  ),
                ],
              ),
            ],
          ),
        );
      },
    );
  }

  /// Builds the scrollable highlight list below the video player.
  Widget _buildHighlightList() {
    return ValueListenableBuilder<VideoPlayerValue>(
      valueListenable: _videoController,
      builder: (context, value, child) {
        final currentPos = value.position;

        return Container(
          color: const Color(0xFF0D0D1A),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
                child: Row(
                  children: [
                    const Icon(Icons.star, color: Colors.amber, size: 16),
                    const SizedBox(width: 6),
                    Text(
                      'Highlights (${_sortedMarkers.length})',
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
              const Divider(height: 1, color: Colors.white12),
              Expanded(
                child: ListView.builder(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  itemCount: _sortedMarkers.length,
                  itemBuilder: (context, index) {
                    final marker = _sortedMarkers[index];
                    // Consider a marker "active" if the playhead is within
                    // 2 seconds of it.
                    final isActive =
                        (currentPos.inMilliseconds - marker.inMilliseconds)
                                .abs() <
                            2000;

                    return _HighlightListItem(
                      index: index,
                      marker: marker,
                      isActive: isActive,
                      onTap: () => _seekToMarker(marker),
                      onShare: () => _shareHighlight(marker, index),
                    );
                  },
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  String _formatPosition(Duration d) {
    final minutes = d.inMinutes.remainder(60);
    final seconds = d.inSeconds.remainder(60);
    return '${minutes.toString().padLeft(2, '0')}:'
        '${seconds.toString().padLeft(2, '0')}';
  }
}

/// A single item in the highlight list below the player.
class _HighlightListItem extends StatelessWidget {
  final int index;
  final Duration marker;
  final bool isActive;
  final VoidCallback onTap;
  final VoidCallback onShare;

  const _HighlightListItem({
    required this.index,
    required this.marker,
    required this.isActive,
    required this.onTap,
    required this.onShare,
  });

  @override
  Widget build(BuildContext context) {
    final minutes = marker.inMinutes.remainder(60);
    final seconds = marker.inSeconds.remainder(60);
    final timestamp = '${minutes.toString().padLeft(2, '0')}:'
        '${seconds.toString().padLeft(2, '0')}';

    return Material(
      color: isActive ? Colors.amber.withOpacity(0.1) : Colors.transparent,
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
          child: Row(
            children: [
              Icon(
                Icons.star,
                size: 18,
                color: isActive ? Colors.amber : Colors.white38,
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  'Highlight ${index + 1}',
                  style: TextStyle(
                    color: isActive ? Colors.amber : Colors.white70,
                    fontSize: 14,
                    fontWeight: isActive ? FontWeight.w600 : FontWeight.normal,
                  ),
                ),
              ),
              Text(
                timestamp,
                style: TextStyle(
                  color: isActive ? Colors.amber : Colors.white54,
                  fontSize: 13,
                  fontFeatures: const [FontFeature.tabularFigures()],
                ),
              ),
              const SizedBox(width: 8),
              IconButton(
                icon: Icon(
                  Icons.share,
                  size: 18,
                  color: isActive ? Colors.amber : Colors.white38,
                ),
                tooltip: 'Share this highlight',
                padding: EdgeInsets.zero,
                constraints: const BoxConstraints(),
                onPressed: onShare,
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// Custom progress bar that overlays highlight markers as amber dots.
///
/// Wraps the standard [VideoProgressIndicator] and paints markers on top.
class _HighlightProgressBar extends StatelessWidget {
  final VideoPlayerController controller;
  final List<Duration> markers;

  const _HighlightProgressBar({
    required this.controller,
    required this.markers,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Stack(
        alignment: Alignment.center,
        children: [
          VideoProgressIndicator(
            controller,
            allowScrubbing: true,
            padding: EdgeInsets.zero,
            colors: const VideoProgressColors(
              playedColor: Color(0xFF1E88E5),
              bufferedColor: Colors.white24,
              backgroundColor: Colors.white12,
            ),
          ),
          // Paint highlight markers on top.
          if (controller.value.duration.inMilliseconds > 0)
            Positioned.fill(
              child: LayoutBuilder(
                builder: (context, constraints) {
                  final totalMs =
                      controller.value.duration.inMilliseconds.toDouble();
                  final barWidth = constraints.maxWidth;

                  return CustomPaint(
                    painter: _MarkerPainter(
                      markers: markers,
                      totalDurationMs: totalMs,
                      barWidth: barWidth,
                    ),
                  );
                },
              ),
            ),
        ],
      ),
    );
  }
}

/// Custom painter that draws amber dots at marker positions on the progress bar.
class _MarkerPainter extends CustomPainter {
  final List<Duration> markers;
  final double totalDurationMs;
  final double barWidth;

  _MarkerPainter({
    required this.markers,
    required this.totalDurationMs,
    required this.barWidth,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (totalDurationMs <= 0) return;

    final paint = Paint()
      ..color = Colors.amber
      ..style = PaintingStyle.fill;

    final centerY = size.height / 2;

    for (final marker in markers) {
      final fraction = marker.inMilliseconds / totalDurationMs;
      if (fraction < 0 || fraction > 1) continue;

      final x = fraction * barWidth;
      // Draw a small amber circle as the marker.
      canvas.drawCircle(Offset(x, centerY), 4.0, paint);
      // Draw a thin vertical line through the marker for visibility.
      final linePaint = Paint()
        ..color = Colors.amber.withOpacity(0.6)
        ..strokeWidth = 1.5;
      canvas.drawLine(
        Offset(x, centerY - 6),
        Offset(x, centerY + 6),
        linePaint,
      );
    }
  }

  @override
  bool shouldRepaint(_MarkerPainter oldDelegate) {
    return markers != oldDelegate.markers ||
        totalDurationMs != oldDelegate.totalDurationMs ||
        barWidth != oldDelegate.barWidth;
  }
}

class _MetadataItem extends StatelessWidget {
  final String label;
  final String value;

  const _MetadataItem({required this.label, required this.value});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          label.toUpperCase(),
          style: const TextStyle(
            color: Colors.white38,
            fontSize: 10,
            letterSpacing: 1,
          ),
        ),
        Text(
          value,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 13,
            fontWeight: FontWeight.w500,
          ),
        ),
      ],
    );
  }
}
