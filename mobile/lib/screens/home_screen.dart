import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../services/calibration_service.dart';
import '../services/clip_service.dart';
import '../services/match_service.dart';

/// Main landing screen for the Beach Tennis Recorder app.
///
/// Provides navigation to:
/// - [RecordingScreen] for starting a new recording session
/// - [LibraryScreen] for reviewing saved rally clips
/// - [SettingsScreen] for configuring the app
/// - [CalibrationScreen] for court calibration
///
/// Shows quick stats: total matches, total clips, calibration status.
///
/// Implements TASK-01-08.
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _initializeServices();
    });
  }

  Future<void> _initializeServices() async {
    try {
      final matchService = context.read<MatchService>();
      final clipService = context.read<ClipService>();
      final calibrationService = context.read<CalibrationService>();

      await Future.wait([
        matchService.initialize(),
        clipService.initialize(),
        calibrationService.initialize(),
      ]);
    } catch (e) {
      debugPrint('Service initialization error: $e');
    }
  }

  @override
  Widget build(BuildContext context) {
    final clipService = context.watch<ClipService>();
    final matchService = context.watch<MatchService>();
    final calibrationService = context.watch<CalibrationService>();
    final clipCount = clipService.clips.length;
    final matchCount = matchService.matches.length;
    final isCalibrated = calibrationService.hasActiveCalibration;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Beach Tennis Recorder'),
        actions: [
          // Calibration status indicator.
          Tooltip(
            message: isCalibrated
                ? 'Court calibrated: ${calibrationService.activeCalibration!.name}'
                : 'No court calibration',
            child: IconButton(
              icon: Icon(
                Icons.crop_free,
                color: isCalibrated ? Colors.green : Colors.orange,
              ),
              onPressed: () =>
                  Navigator.pushNamed(context, '/calibration'),
            ),
          ),
          // Settings icon.
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () => Navigator.pushNamed(context, '/settings'),
          ),
        ],
      ),
      body: SafeArea(
        child: Center(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 32),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Icon(
                  Icons.sports_tennis,
                  size: 80,
                  color: Color(0xFF1E88E5),
                ),
                const SizedBox(height: 16),
                Text(
                  'Beach Tennis Recorder',
                  style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                ),
                const SizedBox(height: 8),
                Text(
                  'Record matches and automatically capture every rally',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: Colors.white70,
                      ),
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 24),

                // Quick stats row.
                _buildStatsRow(matchCount, clipCount, isCalibrated),
                const SizedBox(height: 32),

                // Record button.
                SizedBox(
                  width: double.infinity,
                  height: 56,
                  child: ElevatedButton.icon(
                    onPressed: () {
                      Navigator.pushNamed(context, '/recording');
                    },
                    icon: const Icon(Icons.videocam, size: 28),
                    label: const Text(
                      'Start Recording',
                      style: TextStyle(fontSize: 18),
                    ),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: const Color(0xFF1E88E5),
                      foregroundColor: Colors.white,
                    ),
                  ),
                ),
                const SizedBox(height: 16),

                // Library button.
                SizedBox(
                  width: double.infinity,
                  height: 56,
                  child: OutlinedButton.icon(
                    onPressed: () {
                      Navigator.pushNamed(context, '/library');
                    },
                    icon: const Icon(Icons.video_library, size: 28),
                    label: Text(
                      'Clip Library ($clipCount)',
                      style: const TextStyle(fontSize: 18),
                    ),
                    style: OutlinedButton.styleFrom(
                      side: const BorderSide(color: Color(0xFF1E88E5)),
                    ),
                  ),
                ),
                const SizedBox(height: 24),

                // Recent matches.
                if (matchService.matches.isNotEmpty) ...[
                  Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      'RECENT MATCHES',
                      style: TextStyle(
                        color: Colors.white38,
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                        letterSpacing: 1.2,
                      ),
                    ),
                  ),
                  const SizedBox(height: 8),
                  ...matchService.matches.take(3).map((match) {
                    final clips = clipService.clipCountForMatch(match.id);
                    return Card(
                      child: ListTile(
                        leading: const Icon(Icons.sports_tennis,
                            color: Color(0xFF1E88E5)),
                        title: Text(match.name),
                        subtitle: Text(
                          '$clips rallies - ${match.totalDurationFormatted}',
                          style: const TextStyle(
                            color: Colors.white54,
                            fontSize: 12,
                          ),
                        ),
                        trailing: const Icon(
                          Icons.chevron_right,
                          color: Colors.white38,
                        ),
                        onTap: () {
                          Navigator.pushNamed(context, '/library');
                        },
                      ),
                    );
                  }),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildStatsRow(int matches, int clips, bool calibrated) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        _StatChip(
          icon: Icons.sports_tennis,
          label: 'Matches',
          value: '$matches',
        ),
        _StatChip(
          icon: Icons.movie,
          label: 'Clips',
          value: '$clips',
        ),
        _StatChip(
          icon: Icons.crop_free,
          label: 'Court',
          value: calibrated ? 'OK' : '--',
          valueColor: calibrated ? Colors.green : Colors.orange,
        ),
      ],
    );
  }
}

class _StatChip extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;
  final Color? valueColor;

  const _StatChip({
    required this.icon,
    required this.label,
    required this.value,
    this.valueColor,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.05),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 20, color: Colors.white54),
          const SizedBox(height: 4),
          Text(
            value,
            style: TextStyle(
              color: valueColor ?? Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.bold,
              fontFeatures: const [FontFeature.tabularFigures()],
            ),
          ),
          Text(
            label,
            style: const TextStyle(
              color: Colors.white38,
              fontSize: 11,
            ),
          ),
        ],
      ),
    );
  }
}
