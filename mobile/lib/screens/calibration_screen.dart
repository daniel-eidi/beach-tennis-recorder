import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:uuid/uuid.dart';

import '../models/court_calibration.dart';
import '../services/calibration_service.dart';
import '../services/camera_service.dart';

/// Screen for calibrating the court by tapping 4 corners on the camera preview.
///
/// The user places numbered markers (1-4) on the court corners:
///   1 = top-left, 2 = top-right, 3 = bottom-right, 4 = bottom-left.
///
/// Once all 4 corners are placed, a semi-transparent overlay shows the
/// detected court area. The user names the location and saves the
/// calibration, which persists locally for reuse.
///
/// Implements TASK-01-13.
class CalibrationScreen extends StatefulWidget {
  const CalibrationScreen({super.key});

  @override
  State<CalibrationScreen> createState() => _CalibrationScreenState();
}

class _CalibrationScreenState extends State<CalibrationScreen> {
  final List<Offset> _corners = [];
  String _locationName = '';
  bool _isSaving = false;

  static const _cornerLabels = ['Top-Left', 'Top-Right', 'Bottom-Right', 'Bottom-Left'];
  static const _uuid = Uuid();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _initCamera();
    });
  }

  Future<void> _initCamera() async {
    final cameraService = context.read<CameraService>();
    if (!cameraService.isInitialized) {
      await cameraService.initialize();
    }
  }

  void _onTapDown(TapDownDetails details) {
    if (_corners.length >= 4) return;
    setState(() {
      _corners.add(details.localPosition);
    });
  }

  void _resetCorners() {
    setState(() {
      _corners.clear();
    });
  }

  Future<void> _saveCalibration() async {
    if (_corners.length != 4 || _locationName.trim().isEmpty) return;

    setState(() => _isSaving = true);

    try {
      final points = _corners
          .map((o) => Point2D(o.dx, o.dy))
          .toList();

      final calibration = CourtCalibration.fromCorners(
        id: _uuid.v4(),
        name: _locationName.trim(),
        corners: points,
      );

      final calibrationService = context.read<CalibrationService>();
      await calibrationService.saveCalibration(calibration);
      calibrationService.setActiveCalibration(calibration.id);

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Calibration "${calibration.name}" saved and activated'),
            backgroundColor: Colors.green.shade700,
          ),
        );
        Navigator.pop(context);
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to save calibration: $e'),
            backgroundColor: Colors.red.shade700,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _isSaving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final cameraService = context.watch<CameraService>();

    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        title: const Text('Court Calibration'),
        actions: [
          if (_corners.isNotEmpty)
            IconButton(
              icon: const Icon(Icons.refresh),
              tooltip: 'Reset corners',
              onPressed: _resetCorners,
            ),
        ],
      ),
      body: Column(
        children: [
          // Instructions bar.
          _buildInstructionBar(),

          // Camera preview with overlay.
          Expanded(
            child: _buildCameraArea(cameraService),
          ),

          // Bottom controls.
          _buildBottomControls(),
        ],
      ),
    );
  }

  Widget _buildInstructionBar() {
    final String instruction;
    if (_corners.length < 4) {
      final nextCorner = _cornerLabels[_corners.length];
      instruction = 'Tap corner ${_corners.length + 1}/4: $nextCorner';
    } else {
      instruction = 'All corners placed. Enter a name and save.';
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      color: const Color(0xFF1A1A2E),
      child: Row(
        children: [
          Icon(
            _corners.length < 4
                ? Icons.touch_app
                : Icons.check_circle,
            color: _corners.length < 4
                ? const Color(0xFF1E88E5)
                : Colors.green,
            size: 20,
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              instruction,
              style: const TextStyle(color: Colors.white, fontSize: 14),
            ),
          ),
          // Corner count badge.
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(
              color: _corners.length == 4
                  ? Colors.green.withOpacity(0.2)
                  : const Color(0xFF1E88E5).withOpacity(0.2),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Text(
              '${_corners.length}/4',
              style: TextStyle(
                color: _corners.length == 4
                    ? Colors.green
                    : const Color(0xFF1E88E5),
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCameraArea(CameraService cameraService) {
    if (!cameraService.isInitialized || cameraService.controller == null) {
      return Center(
        child: cameraService.error != null
            ? Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(Icons.error_outline, color: Colors.red, size: 48),
                  const SizedBox(height: 16),
                  Text(
                    cameraService.error!,
                    style: const TextStyle(color: Colors.red),
                    textAlign: TextAlign.center,
                  ),
                ],
              )
            : const CircularProgressIndicator(),
      );
    }

    return GestureDetector(
      onTapDown: _onTapDown,
      child: Stack(
        fit: StackFit.expand,
        children: [
          // Camera preview.
          CameraPreview(cameraService.controller!),

          // Court overlay painter.
          CustomPaint(
            painter: _CourtOverlayPainter(
              corners: _corners,
              isComplete: _corners.length == 4,
            ),
          ),

          // Corner markers.
          ..._buildCornerMarkers(),
        ],
      ),
    );
  }

  List<Widget> _buildCornerMarkers() {
    return List.generate(_corners.length, (i) {
      final offset = _corners[i];
      return Positioned(
        left: offset.dx - 18,
        top: offset.dy - 18,
        child: _CornerMarker(
          number: i + 1,
          label: _cornerLabels[i],
        ),
      );
    });
  }

  Widget _buildBottomControls() {
    return Container(
      padding: const EdgeInsets.all(16),
      color: const Color(0xFF1A1A2E),
      child: SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // Location name input.
            if (_corners.length == 4) ...[
              TextField(
                decoration: InputDecoration(
                  labelText: 'Location name',
                  hintText: 'e.g., Ipanema Court 3',
                  prefixIcon: const Icon(Icons.location_on),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  filled: true,
                  fillColor: Colors.white.withOpacity(0.05),
                ),
                style: const TextStyle(color: Colors.white),
                onChanged: (value) => _locationName = value,
              ),
              const SizedBox(height: 12),
            ],
            Row(
              children: [
                // Reset button.
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: _corners.isEmpty ? null : _resetCorners,
                    icon: const Icon(Icons.refresh),
                    label: const Text('Reset'),
                    style: OutlinedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      side: const BorderSide(color: Colors.white24),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                // Save button.
                Expanded(
                  flex: 2,
                  child: ElevatedButton.icon(
                    onPressed: _corners.length == 4 &&
                            _locationName.trim().isNotEmpty &&
                            !_isSaving
                        ? _saveCalibration
                        : null,
                    icon: _isSaving
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Colors.white,
                            ),
                          )
                        : const Icon(Icons.save),
                    label: Text(_isSaving ? 'Saving...' : 'Save Calibration'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 14),
                      backgroundColor: const Color(0xFF1E88E5),
                      foregroundColor: Colors.white,
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

/// Corner marker widget showing a numbered circle with label.
class _CornerMarker extends StatelessWidget {
  final int number;
  final String label;

  const _CornerMarker({required this.number, required this.label});

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 36,
          height: 36,
          decoration: BoxDecoration(
            color: const Color(0xFF1E88E5),
            shape: BoxShape.circle,
            border: Border.all(color: Colors.white, width: 2),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.5),
                blurRadius: 4,
                offset: const Offset(0, 2),
              ),
            ],
          ),
          child: Center(
            child: Text(
              '$number',
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
                fontSize: 16,
              ),
            ),
          ),
        ),
        Container(
          margin: const EdgeInsets.only(top: 2),
          padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
          decoration: BoxDecoration(
            color: Colors.black54,
            borderRadius: BorderRadius.circular(4),
          ),
          child: Text(
            label,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 9,
              fontWeight: FontWeight.w500,
            ),
          ),
        ),
      ],
    );
  }
}

/// Custom painter that draws the court overlay on the camera preview.
///
/// Draws lines connecting the placed corners and fills the court area
/// with a semi-transparent blue overlay when all 4 corners are placed.
class _CourtOverlayPainter extends CustomPainter {
  final List<Offset> corners;
  final bool isComplete;

  _CourtOverlayPainter({
    required this.corners,
    required this.isComplete,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (corners.isEmpty) return;

    final linePaint = Paint()
      ..color = const Color(0xFF1E88E5)
      ..strokeWidth = 2.5
      ..style = PaintingStyle.stroke;

    // Draw lines between placed corners.
    for (int i = 0; i < corners.length - 1; i++) {
      canvas.drawLine(corners[i], corners[i + 1], linePaint);
    }

    // Close the polygon if all 4 corners are placed.
    if (isComplete) {
      canvas.drawLine(corners[3], corners[0], linePaint);

      // Fill with semi-transparent overlay.
      final fillPaint = Paint()
        ..color = const Color(0xFF1E88E5).withOpacity(0.15)
        ..style = PaintingStyle.fill;

      final path = Path()
        ..moveTo(corners[0].dx, corners[0].dy)
        ..lineTo(corners[1].dx, corners[1].dy)
        ..lineTo(corners[2].dx, corners[2].dy)
        ..lineTo(corners[3].dx, corners[3].dy)
        ..close();

      canvas.drawPath(path, fillPaint);

      // Draw net line (midpoint of left edge to midpoint of right edge).
      final netLeft = Offset(
        (corners[0].dx + corners[3].dx) / 2,
        (corners[0].dy + corners[3].dy) / 2,
      );
      final netRight = Offset(
        (corners[1].dx + corners[2].dx) / 2,
        (corners[1].dy + corners[2].dy) / 2,
      );

      final netPaint = Paint()
        ..color = Colors.white.withOpacity(0.6)
        ..strokeWidth = 2
        ..style = PaintingStyle.stroke;

      canvas.drawLine(netLeft, netRight, netPaint);

      // Label the net.
      final textPainter = TextPainter(
        text: const TextSpan(
          text: 'NET',
          style: TextStyle(
            color: Colors.white70,
            fontSize: 12,
            fontWeight: FontWeight.bold,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      final netCenter = Offset(
        (netLeft.dx + netRight.dx) / 2 - textPainter.width / 2,
        (netLeft.dy + netRight.dy) / 2 - textPainter.height - 4,
      );
      textPainter.paint(canvas, netCenter);
    }
  }

  @override
  bool shouldRepaint(_CourtOverlayPainter old) {
    return old.corners.length != corners.length || old.isComplete != isComplete;
  }
}
