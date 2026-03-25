import 'dart:async';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/gesture.dart';
import '../models/rally.dart';
import '../services/pipeline_controller.dart';
import '../services/settings_service.dart';

/// Screen that shows the live camera preview during match recording.
///
/// Displays:
/// - Full-screen camera preview
/// - Rally state indicator (color-coded)
/// - Rally counter (number of rallies detected)
/// - Real-time inference FPS and latency
/// - Detection count overlay
/// - Start/stop recording button
///
/// The full pipeline (camera -> detection isolate -> rally controller ->
/// clip extraction) is managed via [PipelineController].
///
/// Implements TASK-01-09.
class RecordingScreen extends StatefulWidget {
  const RecordingScreen({super.key});

  @override
  State<RecordingScreen> createState() => _RecordingScreenState();
}

class _RecordingScreenState extends State<RecordingScreen>
    with WidgetsBindingObserver {
  Duration _elapsed = Duration.zero;
  Timer? _elapsedTimer;
  StreamSubscription? _gestureEventSubscription;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initializeCamera();
    _listenForGestureEvents();
  }

  void _listenForGestureEvents() {
    final pipeline = context.read<PipelineController>();
    _gestureEventSubscription =
        pipeline.gestureDetectorService.gestureEvents.listen((_) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: const Row(
              children: [
                Icon(Icons.star, color: Colors.yellow),
                SizedBox(width: 8),
                Text('HIGHLIGHT SAVED!'),
              ],
            ),
            backgroundColor: Colors.green.shade700,
            duration: const Duration(seconds: 2),
            behavior: SnackBarBehavior.floating,
          ),
        );
      }
    });
  }

  Future<void> _initializeCamera() async {
    final pipeline = context.read<PipelineController>();
    if (!pipeline.cameraService.isInitialized) {
      await pipeline.cameraService.initialize();
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final pipeline = context.read<PipelineController>();

    // Handle app lifecycle to properly manage camera resources.
    if (state == AppLifecycleState.inactive) {
      pipeline.cameraService.controller?.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initializeCamera();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _elapsedTimer?.cancel();
    _gestureEventSubscription?.cancel();
    super.dispose();
  }

  Future<void> _startSession() async {
    final pipeline = context.read<PipelineController>();
    await pipeline.startRecording();

    // Start elapsed time counter.
    _elapsed = Duration.zero;
    _elapsedTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (mounted) {
        setState(() {
          _elapsed += const Duration(seconds: 1);
        });
      }
    });
  }

  Future<void> _stopSession() async {
    _elapsedTimer?.cancel();
    final pipeline = context.read<PipelineController>();
    await pipeline.stopRecording();
  }

  @override
  Widget build(BuildContext context) {
    final pipeline = context.watch<PipelineController>();
    final cameraService = pipeline.cameraService;
    final rallyController = pipeline.rallyController;

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Camera preview (full screen).
          _buildCameraPreview(cameraService),

          // Top bar: back button, elapsed time, rally state.
          Positioned(
            top: 0,
            left: 0,
            right: 0,
            child: _buildTopBar(pipeline, rallyController),
          ),

          // Detection stats overlay (top-right area).
          if (pipeline.isRecording)
            Positioned(
              top: 80,
              right: 16,
              child: _buildStatsOverlay(pipeline),
            ),

          // Bottom bar: rally counter, record button.
          Positioned(
            bottom: 0,
            left: 0,
            right: 0,
            child: _buildBottomBar(pipeline, rallyController),
          ),
        ],
      ),
    );
  }

  Widget _buildCameraPreview(dynamic cameraService) {
    if (!cameraService.isInitialized || cameraService.controller == null) {
      return Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (cameraService.error != null) ...[
              const Icon(Icons.error_outline, color: Colors.red, size: 48),
              const SizedBox(height: 16),
              Text(
                cameraService.error!,
                style: const TextStyle(color: Colors.red),
                textAlign: TextAlign.center,
              ),
            ] else ...[
              const CircularProgressIndicator(),
              const SizedBox(height: 16),
              const Text(
                'Initializing camera...',
                style: TextStyle(color: Colors.white70),
              ),
            ],
          ],
        ),
      );
    }

    return CameraPreview(cameraService.controller!);
  }

  Widget _buildTopBar(
    PipelineController pipeline,
    dynamic rallyController,
  ) {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [Colors.black54, Colors.transparent],
        ),
      ),
      padding: const EdgeInsets.fromLTRB(8, 8, 16, 24),
      child: SafeArea(
        child: Row(
          children: [
            // Back button.
            IconButton(
              icon: const Icon(Icons.arrow_back, color: Colors.white),
              onPressed: () async {
                if (pipeline.isRecording) {
                  await _stopSession();
                }
                if (context.mounted) {
                  Navigator.pop(context);
                }
              },
            ),
            const Spacer(),

            // Elapsed time.
            if (pipeline.isRecording)
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 6,
                ),
                decoration: BoxDecoration(
                  color: Colors.black45,
                  borderRadius: BorderRadius.circular(16),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(Icons.circle, color: Colors.red, size: 10),
                    const SizedBox(width: 8),
                    Text(
                      _formatDuration(_elapsed),
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 16,
                        fontFeatures: [FontFeature.tabularFigures()],
                      ),
                    ),
                  ],
                ),
              ),

            const Spacer(),

            // Model status indicator.
            if (pipeline.isRecording)
              Padding(
                padding: const EdgeInsets.only(right: 8),
                child: Icon(
                  pipeline.modelAvailable
                      ? Icons.psychology
                      : Icons.psychology_outlined,
                  color: pipeline.modelAvailable
                      ? Colors.green
                      : Colors.orange,
                  size: 20,
                ),
              ),

            // Gesture detection indicator.
            if (pipeline.isRecording)
              _buildGestureIndicator(pipeline),

            const SizedBox(width: 8),

            // Rally state indicator.
            _buildRallyStateChip(rallyController.state),
          ],
        ),
      ),
    );
  }

  /// Builds the gesture detection indicator icon.
  ///
  /// Colors:
  ///   Grey: gesture detection disabled
  ///   White: idle (enabled, watching)
  ///   Yellow: racket raised detected
  ///   Green flash: gesture detected, saving clip!
  Widget _buildGestureIndicator(PipelineController pipeline) {
    final gestureState = pipeline.gestureState;
    final isEnabled = pipeline.isGestureEnabled;

    final (color, icon) = switch (gestureState) {
      GestureState.idle when !isEnabled => (Colors.grey, Icons.front_hand_outlined),
      GestureState.idle => (Colors.white, Icons.front_hand_outlined),
      GestureState.racketRaised => (Colors.yellow, Icons.front_hand),
      GestureState.gestureDetected => (Colors.greenAccent, Icons.front_hand),
      GestureState.cooldown => (Colors.white38, Icons.front_hand_outlined),
    };

    return GestureDetector(
      onTap: () => _toggleGestureDetection(pipeline),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.all(4),
        decoration: BoxDecoration(
          color: gestureState == GestureState.gestureDetected
              ? Colors.greenAccent.withOpacity(0.3)
              : Colors.transparent,
          borderRadius: BorderRadius.circular(8),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, color: color, size: 20),
            if (pipeline.gestureCount > 0) ...[
              const SizedBox(width: 4),
              Text(
                '${pipeline.gestureCount}',
                style: TextStyle(
                  color: color,
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  void _toggleGestureDetection(PipelineController pipeline) {
    final settings = context.read<SettingsService>();
    settings.gestureEnabled = !settings.gestureEnabled;

    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            settings.gestureEnabled
                ? 'Gesture detection enabled'
                : 'Gesture detection disabled',
          ),
          duration: const Duration(seconds: 1),
        ),
      );
    }
  }

  Widget _buildRallyStateChip(RallyState state) {
    final (label, color) = switch (state) {
      RallyState.idle => ('IDLE', Colors.grey),
      RallyState.emJogo => ('RALLY', Colors.green),
      RallyState.fimRally => ('SAVING...', Colors.orange),
    };

    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.3),
        border: Border.all(color: color),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: color,
          fontWeight: FontWeight.bold,
          fontSize: 14,
        ),
      ),
    );
  }

  /// Overlay showing real-time detection performance stats.
  Widget _buildStatsOverlay(PipelineController pipeline) {
    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.black54,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.end,
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(
            '${pipeline.fps.toStringAsFixed(1)} FPS',
            style: TextStyle(
              color: pipeline.fps > 10
                  ? Colors.greenAccent
                  : (pipeline.fps > 5 ? Colors.orange : Colors.redAccent),
              fontSize: 12,
              fontWeight: FontWeight.bold,
              fontFeatures: const [FontFeature.tabularFigures()],
            ),
          ),
          const SizedBox(height: 2),
          Text(
            '${pipeline.inferenceTimeMs}ms',
            style: const TextStyle(
              color: Colors.white70,
              fontSize: 11,
              fontFeatures: [FontFeature.tabularFigures()],
            ),
          ),
          const SizedBox(height: 2),
          Text(
            '${pipeline.detectionsFound} det',
            style: const TextStyle(
              color: Colors.white54,
              fontSize: 11,
              fontFeatures: [FontFeature.tabularFigures()],
            ),
          ),
          if (!pipeline.modelAvailable)
            const Text(
              'NO MODEL',
              style: TextStyle(
                color: Colors.orange,
                fontSize: 10,
                fontWeight: FontWeight.bold,
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildBottomBar(
    PipelineController pipeline,
    dynamic rallyController,
  ) {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.bottomCenter,
          end: Alignment.topCenter,
          colors: [Colors.black54, Colors.transparent],
        ),
      ),
      padding: const EdgeInsets.fromLTRB(24, 24, 24, 16),
      child: SafeArea(
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            // Rally counter.
            Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  '${pipeline.rallyCount}',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 32,
                    fontWeight: FontWeight.bold,
                    fontFeatures: [FontFeature.tabularFigures()],
                  ),
                ),
                const Text(
                  'RALLIES',
                  style: TextStyle(
                    color: Colors.white54,
                    fontSize: 12,
                    letterSpacing: 1.2,
                  ),
                ),
              ],
            ),

            // Record button.
            GestureDetector(
              onTap: pipeline.isRecording ? _stopSession : _startSession,
              child: Container(
                width: 72,
                height: 72,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  border: Border.all(color: Colors.white, width: 4),
                ),
                child: Center(
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 200),
                    width: pipeline.isRecording ? 28 : 56,
                    height: pipeline.isRecording ? 28 : 56,
                    decoration: BoxDecoration(
                      color: Colors.red,
                      borderRadius: BorderRadius.circular(
                        pipeline.isRecording ? 6 : 28,
                      ),
                    ),
                  ),
                ),
              ),
            ),

            // Frames processed counter.
            Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  '${pipeline.framesProcessed}',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    fontFeatures: [FontFeature.tabularFigures()],
                  ),
                ),
                const Text(
                  'FRAMES',
                  style: TextStyle(
                    color: Colors.white54,
                    fontSize: 12,
                    letterSpacing: 1.2,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  String _formatDuration(Duration d) {
    final hours = d.inHours;
    final minutes = d.inMinutes.remainder(60);
    final seconds = d.inSeconds.remainder(60);
    if (hours > 0) {
      return '${hours.toString().padLeft(2, '0')}:'
          '${minutes.toString().padLeft(2, '0')}:'
          '${seconds.toString().padLeft(2, '0')}';
    }
    return '${minutes.toString().padLeft(2, '0')}:'
        '${seconds.toString().padLeft(2, '0')}';
  }
}
