import 'dart:io';

import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../services/calibration_service.dart';
import '../services/clip_service.dart';
import '../services/settings_service.dart';

/// Settings screen for configuring camera, detection, rally, and storage
/// options.
///
/// All changes are persisted immediately via [SettingsService] and take
/// effect on the next recording session (or immediately for settings that
/// the rally controller reads dynamically).
///
/// Implements TASK-01-14.
class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Settings'),
        actions: [
          IconButton(
            icon: const Icon(Icons.restore),
            tooltip: 'Reset to defaults',
            onPressed: () => _confirmResetDefaults(context),
          ),
        ],
      ),
      body: Consumer<SettingsService>(
        builder: (context, settings, _) {
          return ListView(
            padding: const EdgeInsets.symmetric(vertical: 8),
            children: [
              _buildSectionHeader(context, 'Camera'),
              _buildResolutionTile(context, settings),
              _buildFpsTile(context, settings),
              const Divider(),

              _buildSectionHeader(context, 'Buffer'),
              _buildBufferDurationTile(context, settings),
              const Divider(),

              _buildSectionHeader(context, 'Detection'),
              _buildConfidenceSlider(context, settings),
              _buildVelocitySlider(context, settings),
              const Divider(),

              _buildSectionHeader(context, 'Rally'),
              _buildPreRallyTile(context, settings),
              _buildPostRallyTile(context, settings),
              _buildTimeoutTile(context, settings),
              _buildNetCrossTile(context, settings),
              const Divider(),

              _buildSectionHeader(context, 'Court Calibration'),
              _buildCalibrationTile(context),
              const Divider(),

              _buildSectionHeader(context, 'Storage'),
              _buildStorageTile(context),
              _buildClearClipsTile(context),
            ],
          );
        },
      ),
    );
  }

  // ─── Section header ─────────────────────────────────────────────────────

  Widget _buildSectionHeader(BuildContext context, String title) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 4),
      child: Text(
        title.toUpperCase(),
        style: TextStyle(
          color: const Color(0xFF1E88E5),
          fontSize: 12,
          fontWeight: FontWeight.bold,
          letterSpacing: 1.2,
        ),
      ),
    );
  }

  // ─── Camera settings ────────────────────────────────────────────────────

  Widget _buildResolutionTile(BuildContext context, SettingsService settings) {
    return ListTile(
      leading: const Icon(Icons.high_quality),
      title: const Text('Resolution'),
      subtitle: Text(settings.resolution),
      trailing: const Icon(Icons.chevron_right),
      onTap: () {
        _showOptionDialog<String>(
          context,
          title: 'Resolution',
          options: ['720p', '1080p', '4K'],
          current: settings.resolution,
          onSelect: (value) => settings.resolution = value,
        );
      },
    );
  }

  Widget _buildFpsTile(BuildContext context, SettingsService settings) {
    return ListTile(
      leading: const Icon(Icons.speed),
      title: const Text('Frame Rate'),
      subtitle: Text('${settings.fps} FPS'),
      trailing: const Icon(Icons.chevron_right),
      onTap: () {
        _showOptionDialog<int>(
          context,
          title: 'Frame Rate',
          options: [24, 30, 60],
          labels: ['24 FPS', '30 FPS', '60 FPS'],
          current: settings.fps,
          onSelect: (value) => settings.fps = value,
        );
      },
    );
  }

  // ─── Buffer settings ────────────────────────────────────────────────────

  Widget _buildBufferDurationTile(
    BuildContext context,
    SettingsService settings,
  ) {
    return ListTile(
      leading: const Icon(Icons.timelapse),
      title: const Text('Buffer Duration'),
      subtitle: Text('${settings.bufferDuration} seconds'),
      trailing: const Icon(Icons.chevron_right),
      onTap: () {
        _showOptionDialog<int>(
          context,
          title: 'Buffer Duration',
          options: [30, 60, 90],
          labels: ['30 seconds', '60 seconds', '90 seconds'],
          current: settings.bufferDuration,
          onSelect: (value) => settings.bufferDuration = value,
        );
      },
    );
  }

  // ─── Detection settings ─────────────────────────────────────────────────

  Widget _buildConfidenceSlider(
    BuildContext context,
    SettingsService settings,
  ) {
    return ListTile(
      leading: const Icon(Icons.tune),
      title: const Text('Confidence Threshold'),
      subtitle: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Slider(
            value: settings.confidenceThreshold,
            min: 0.1,
            max: 0.9,
            divisions: 16,
            label: settings.confidenceThreshold.toStringAsFixed(2),
            onChanged: (value) {
              settings.confidenceThreshold =
                  double.parse(value.toStringAsFixed(2));
            },
          ),
          Text(
            'Current: ${settings.confidenceThreshold.toStringAsFixed(2)} '
            '(default: ${SettingsService.defaultConfidenceThreshold})',
            style: const TextStyle(fontSize: 12, color: Colors.white38),
          ),
        ],
      ),
    );
  }

  Widget _buildVelocitySlider(
    BuildContext context,
    SettingsService settings,
  ) {
    return ListTile(
      leading: const Icon(Icons.speed),
      title: const Text('Velocity Threshold'),
      subtitle: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Slider(
            value: settings.velocityThreshold,
            min: 5.0,
            max: 30.0,
            divisions: 25,
            label: settings.velocityThreshold.toStringAsFixed(1),
            onChanged: (value) {
              settings.velocityThreshold =
                  double.parse(value.toStringAsFixed(1));
            },
          ),
          Text(
            'Current: ${settings.velocityThreshold.toStringAsFixed(1)} px/frame '
            '(default: ${SettingsService.defaultVelocityThreshold})',
            style: const TextStyle(fontSize: 12, color: Colors.white38),
          ),
        ],
      ),
    );
  }

  // ─── Rally settings ─────────────────────────────────────────────────────

  Widget _buildPreRallyTile(BuildContext context, SettingsService settings) {
    return ListTile(
      leading: const Icon(Icons.skip_previous),
      title: const Text('Pre-Rally Capture'),
      subtitle: Text('${settings.preRallySeconds} seconds before serve'),
      trailing: const Icon(Icons.chevron_right),
      onTap: () {
        _showOptionDialog<int>(
          context,
          title: 'Pre-Rally Seconds',
          options: [1, 2, 3, 4, 5],
          labels: ['1s', '2s', '3s', '4s', '5s'],
          current: settings.preRallySeconds,
          onSelect: (value) => settings.preRallySeconds = value,
        );
      },
    );
  }

  Widget _buildPostRallyTile(BuildContext context, SettingsService settings) {
    return ListTile(
      leading: const Icon(Icons.skip_next),
      title: const Text('Post-Rally Capture'),
      subtitle: Text('${settings.postRallySeconds} seconds after rally ends'),
      trailing: const Icon(Icons.chevron_right),
      onTap: () {
        _showOptionDialog<int>(
          context,
          title: 'Post-Rally Seconds',
          options: [1, 2, 3, 4, 5],
          labels: ['1s', '2s', '3s', '4s', '5s'],
          current: settings.postRallySeconds,
          onSelect: (value) => settings.postRallySeconds = value,
        );
      },
    );
  }

  Widget _buildTimeoutTile(BuildContext context, SettingsService settings) {
    return ListTile(
      leading: const Icon(Icons.timer_off),
      title: const Text('Rally Timeout'),
      subtitle: Text(
        '${settings.timeoutSeconds}s without detection ends rally',
      ),
      trailing: const Icon(Icons.chevron_right),
      onTap: () {
        _showOptionDialog<int>(
          context,
          title: 'Timeout Seconds',
          options: [5, 6, 7, 8, 10, 12, 15],
          labels: ['5s', '6s', '7s', '8s', '10s', '12s', '15s'],
          current: settings.timeoutSeconds,
          onSelect: (value) => settings.timeoutSeconds = value,
        );
      },
    );
  }

  Widget _buildNetCrossTile(BuildContext context, SettingsService settings) {
    return SwitchListTile(
      secondary: const Icon(Icons.swap_horiz),
      title: const Text('Require Net Crossing'),
      subtitle: const Text('Ball must cross the net to confirm rally'),
      value: settings.netCrossRequired,
      onChanged: (value) => settings.netCrossRequired = value,
    );
  }

  // ─── Calibration ────────────────────────────────────────────────────────

  Widget _buildCalibrationTile(BuildContext context) {
    final calibrationService = context.watch<CalibrationService>();
    final active = calibrationService.activeCalibration;

    return ListTile(
      leading: Icon(
        Icons.crop_free,
        color: active != null ? Colors.green : Colors.white54,
      ),
      title: const Text('Active Calibration'),
      subtitle: Text(
        active != null ? active.name : 'No calibration set',
        style: TextStyle(
          color: active != null ? Colors.white : Colors.white38,
        ),
      ),
      trailing: const Icon(Icons.chevron_right),
      onTap: () => Navigator.pushNamed(context, '/calibration'),
    );
  }

  // ─── Storage ────────────────────────────────────────────────────────────

  Widget _buildStorageTile(BuildContext context) {
    return FutureBuilder<_StorageInfo>(
      future: _getStorageInfo(context),
      builder: (context, snapshot) {
        final info = snapshot.data;
        return ListTile(
          leading: const Icon(Icons.storage),
          title: const Text('Storage Used'),
          subtitle: Text(
            info != null
                ? '${info.clipCount} clips, ${info.sizeMB.toStringAsFixed(1)} MB'
                : 'Calculating...',
          ),
        );
      },
    );
  }

  Widget _buildClearClipsTile(BuildContext context) {
    return ListTile(
      leading: const Icon(Icons.delete_sweep, color: Colors.red),
      title: const Text('Clear All Clips'),
      subtitle: const Text('Permanently delete all saved clips'),
      onTap: () => _confirmClearClips(context),
    );
  }

  // ─── Dialogs ────────────────────────────────────────────────────────────

  void _showOptionDialog<T>({
    required BuildContext context,
    required String title,
    required List<T> options,
    List<String>? labels,
    required T current,
    required void Function(T) onSelect,
  }) {
    showDialog(
      context: context,
      builder: (ctx) => SimpleDialog(
        title: Text(title),
        children: List.generate(options.length, (i) {
          final option = options[i];
          final label = labels != null ? labels[i] : option.toString();
          return RadioListTile<T>(
            title: Text(label),
            value: option,
            groupValue: current,
            onChanged: (value) {
              if (value != null) onSelect(value);
              Navigator.pop(ctx);
            },
          );
        }),
      ),
    );
  }

  void _confirmResetDefaults(BuildContext context) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Reset Settings?'),
        content: const Text(
          'All settings will be restored to their default values.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('CANCEL'),
          ),
          TextButton(
            onPressed: () {
              context.read<SettingsService>().resetToDefaults();
              Navigator.pop(ctx);
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Settings reset to defaults')),
              );
            },
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: const Text('RESET'),
          ),
        ],
      ),
    );
  }

  void _confirmClearClips(BuildContext context) {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Delete All Clips?'),
        content: const Text(
          'This will permanently delete all saved rally clips from this device. '
          'This action cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('CANCEL'),
          ),
          TextButton(
            onPressed: () async {
              Navigator.pop(ctx);
              final clipService = context.read<ClipService>();
              // Delete all clips.
              final clipIds = clipService.clips.map((c) => c.id).toList();
              for (final id in clipIds) {
                await clipService.deleteClip(id);
              }
              if (context.mounted) {
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(
                    content: Text('Deleted ${clipIds.length} clips'),
                  ),
                );
              }
            },
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: const Text('DELETE ALL'),
          ),
        ],
      ),
    );
  }

  Future<_StorageInfo> _getStorageInfo(BuildContext context) async {
    final clipService = context.read<ClipService>();
    final clips = clipService.clips;
    int totalBytes = 0;
    for (final clip in clips) {
      if (clip.fileSizeBytes != null) {
        totalBytes += clip.fileSizeBytes!;
      } else {
        try {
          final file = File(clip.filePath);
          if (await file.exists()) {
            totalBytes += await file.length();
          }
        } catch (_) {}
      }
    }
    return _StorageInfo(
      clipCount: clips.length,
      sizeMB: totalBytes / (1024 * 1024),
    );
  }
}

class _StorageInfo {
  final int clipCount;
  final double sizeMB;

  const _StorageInfo({required this.clipCount, required this.sizeMB});
}
