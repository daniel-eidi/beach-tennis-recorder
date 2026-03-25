import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import 'services/camera_service.dart';
import 'services/buffer_service.dart';
import 'services/calibration_service.dart';
import 'services/clip_service.dart';
import 'services/gesture_detector_service.dart';
import 'services/match_service.dart';
import 'services/pipeline_controller.dart';
import 'services/rally_controller.dart';
import 'services/settings_service.dart';
import 'screens/calibration_screen.dart';
import 'screens/home_screen.dart';
import 'screens/library_screen.dart';
import 'screens/recording_screen.dart';
import 'screens/settings_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Lock orientation to landscape for optimal court recording.
  await SystemChrome.setPreferredOrientations([
    DeviceOrientation.landscapeLeft,
    DeviceOrientation.landscapeRight,
  ]);

  // Set system UI overlay style.
  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.light,
    ),
  );

  // Initialize SettingsService before the widget tree so that other
  // services can read settings immediately.
  final settingsService = SettingsService();
  await settingsService.initialize();

  runApp(BeachTennisRecorderApp(settingsService: settingsService));
}

class BeachTennisRecorderApp extends StatelessWidget {
  final SettingsService settingsService;

  const BeachTennisRecorderApp({super.key, required this.settingsService});

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        // Settings — initialized first, no dependencies.
        ChangeNotifierProvider.value(value: settingsService),

        // Core services.
        ChangeNotifierProvider(create: (_) => CameraService()),
        ChangeNotifierProvider(create: (_) => BufferService()),
        ChangeNotifierProvider(create: (_) => MatchService()),

        // Calibration depends on Settings.
        ChangeNotifierProxyProvider<SettingsService, CalibrationService>(
          create: (context) => CalibrationService(
            settingsService: context.read<SettingsService>(),
          ),
          update: (_, settings, previous) =>
              previous ?? CalibrationService(settingsService: settings),
        ),

        // Clip service depends on Buffer.
        ChangeNotifierProxyProvider<BufferService, ClipService>(
          create: (context) => ClipService(
            bufferService: context.read<BufferService>(),
          ),
          update: (_, bufferService, previous) =>
              previous ?? ClipService(bufferService: bufferService),
        ),

        // Gesture detector depends on Settings.
        ChangeNotifierProxyProvider<SettingsService, GestureDetectorService>(
          create: (context) => GestureDetectorService(
            settingsService: context.read<SettingsService>(),
          ),
          update: (_, settings, previous) {
            if (previous != null) {
              previous.updateSettings(settings);
              return previous;
            }
            return GestureDetectorService(settingsService: settings);
          },
        ),

        // Rally controller depends on Clip + Settings.
        ChangeNotifierProxyProvider2<ClipService, SettingsService,
            RallyController>(
          create: (context) => RallyController(
            clipService: context.read<ClipService>(),
            settingsService: context.read<SettingsService>(),
          ),
          update: (_, clipService, settings, previous) {
            if (previous != null) {
              previous.updateSettings(settings);
              return previous;
            }
            return RallyController(
              clipService: clipService,
              settingsService: settings,
            );
          },
        ),

        // Pipeline orchestrates everything.
        ChangeNotifierProxyProvider6<CameraService, BufferService, ClipService,
            RallyController, MatchService, GestureDetectorService,
            PipelineController>(
          create: (context) => PipelineController(
            cameraService: context.read<CameraService>(),
            bufferService: context.read<BufferService>(),
            clipService: context.read<ClipService>(),
            rallyController: context.read<RallyController>(),
            matchService: context.read<MatchService>(),
            gestureDetectorService: context.read<GestureDetectorService>(),
          ),
          update: (_, camera, buffer, clip, rally, match, gesture, previous) =>
              previous ??
              PipelineController(
                cameraService: camera,
                bufferService: buffer,
                clipService: clip,
                rallyController: rally,
                matchService: match,
                gestureDetectorService: gesture,
              ),
        ),
      ],
      child: MaterialApp(
        title: 'Beach Tennis Recorder',
        debugShowCheckedModeBanner: false,
        theme: _buildTheme(),
        initialRoute: '/',
        routes: {
          '/': (context) => const HomeScreen(),
          '/recording': (context) => const RecordingScreen(),
          '/library': (context) => const LibraryScreen(),
          '/calibration': (context) => const CalibrationScreen(),
          '/settings': (context) => const SettingsScreen(),
        },
      ),
    );
  }

  ThemeData _buildTheme() {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.dark,
      colorSchemeSeed: const Color(0xFF1E88E5),
      scaffoldBackgroundColor: const Color(0xFF121212),
      appBarTheme: const AppBarTheme(
        backgroundColor: Color(0xFF1A1A2E),
        elevation: 0,
        centerTitle: true,
      ),
      cardTheme: CardTheme(
        color: const Color(0xFF1A1A2E),
        elevation: 2,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(12),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
      ),
      floatingActionButtonTheme: const FloatingActionButtonThemeData(
        backgroundColor: Color(0xFF1E88E5),
        foregroundColor: Colors.white,
      ),
    );
  }
}
