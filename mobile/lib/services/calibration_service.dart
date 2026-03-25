import 'dart:convert';
import 'dart:developer' as developer;
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

import '../models/court_calibration.dart';
import 'settings_service.dart';

/// Manages court calibration persistence and selection.
///
/// Calibrations are stored as individual JSON files in the app documents
/// directory under `bt_calibrations/`. Each calibration represents a
/// physical court location and is reused across multiple matches.
///
/// The active calibration (selected for detection) is tracked via
/// [SettingsService.activeCalibrationId].
class CalibrationService extends ChangeNotifier {
  final SettingsService _settingsService;

  Directory? _calibrationsDir;
  final List<CourtCalibration> _calibrations = [];
  CourtCalibration? _activeCalibration;
  bool _isInitialized = false;

  CalibrationService({required SettingsService settingsService})
      : _settingsService = settingsService;

  /// All saved calibrations, newest first.
  List<CourtCalibration> get calibrations =>
      List.unmodifiable(_calibrations);

  /// The currently active calibration used for detection.
  CourtCalibration? get activeCalibration => _activeCalibration;

  /// Whether any calibration is active.
  bool get hasActiveCalibration => _activeCalibration != null;

  bool get isInitialized => _isInitialized;

  /// Initializes storage and loads existing calibrations from disk.
  Future<void> initialize() async {
    final appDir = await getApplicationDocumentsDirectory();
    _calibrationsDir = Directory(p.join(appDir.path, 'bt_calibrations'));

    if (!await _calibrationsDir!.exists()) {
      await _calibrationsDir!.create(recursive: true);
    }

    await _loadCalibrations();

    // Restore active calibration from settings.
    final activeId = _settingsService.activeCalibrationId;
    if (activeId != null) {
      _activeCalibration = _calibrations
          .cast<CourtCalibration?>()
          .firstWhere(
            (c) => c!.id == activeId,
            orElse: () => null,
          );
    }

    _isInitialized = true;
    _log('info', 'CalibrationService initialized: '
        '${_calibrations.length} calibrations, '
        'active=${_activeCalibration?.name ?? "none"}');
    notifyListeners();
  }

  /// Saves a new calibration to disk.
  Future<void> saveCalibration(CourtCalibration calibration) async {
    if (_calibrationsDir == null) await initialize();

    try {
      final file = File(_calibrationFilePath(calibration.id));
      await file.writeAsString(jsonEncode(calibration.toJson()));

      // Add to the in-memory list (or update if already exists).
      final existingIndex =
          _calibrations.indexWhere((c) => c.id == calibration.id);
      if (existingIndex >= 0) {
        _calibrations[existingIndex] = calibration;
      } else {
        _calibrations.insert(0, calibration);
      }

      _log('info', 'Saved calibration: "${calibration.name}"');
      notifyListeners();
    } catch (e) {
      _log('error', 'Failed to save calibration: $e');
    }
  }

  /// Loads all calibrations from disk.
  Future<List<CourtCalibration>> loadCalibrations() async {
    if (_calibrationsDir == null) await initialize();
    return List.unmodifiable(_calibrations);
  }

  /// Returns the active calibration, or null.
  CourtCalibration? getActiveCalibration() => _activeCalibration;

  /// Sets the active calibration by ID.
  void setActiveCalibration(String id) {
    _activeCalibration = _calibrations
        .cast<CourtCalibration?>()
        .firstWhere(
          (c) => c!.id == id,
          orElse: () => null,
        );

    _settingsService.activeCalibrationId =
        _activeCalibration?.id;

    _log('info', 'Active calibration set: '
        '${_activeCalibration?.name ?? "none"}');
    notifyListeners();
  }

  /// Clears the active calibration selection.
  void clearActiveCalibration() {
    _activeCalibration = null;
    _settingsService.activeCalibrationId = null;
    _log('info', 'Active calibration cleared');
    notifyListeners();
  }

  /// Deletes a calibration from disk and memory.
  Future<bool> deleteCalibration(String id) async {
    final index = _calibrations.indexWhere((c) => c.id == id);
    if (index == -1) return false;

    try {
      final file = File(_calibrationFilePath(id));
      if (await file.exists()) {
        await file.delete();
      }

      _calibrations.removeAt(index);

      // Clear active if deleted.
      if (_activeCalibration?.id == id) {
        _activeCalibration = null;
        _settingsService.activeCalibrationId = null;
      }

      _log('info', 'Deleted calibration: $id');
      notifyListeners();
      return true;
    } catch (e) {
      _log('error', 'Failed to delete calibration: $e');
      return false;
    }
  }

  // ─── Private ────────────────────────────────────────────────────────────

  String _calibrationFilePath(String id) {
    return p.join(_calibrationsDir!.path, '$id.json');
  }

  Future<void> _loadCalibrations() async {
    if (_calibrationsDir == null) return;

    try {
      final entities = await _calibrationsDir!.list().toList();
      _calibrations.clear();

      for (final entity in entities) {
        if (entity is File && entity.path.endsWith('.json')) {
          try {
            final content = await entity.readAsString();
            final json = jsonDecode(content) as Map<String, dynamic>;
            _calibrations.add(CourtCalibration.fromJson(json));
          } catch (e) {
            _log('warn', 'Failed to parse calibration file: ${entity.path}');
          }
        }
      }

      // Sort newest first.
      _calibrations.sort((a, b) => b.createdAt.compareTo(a.createdAt));
    } catch (e) {
      _log('error', 'Failed to load calibrations: $e');
    }
  }

  void _log(String level, String message) {
    developer.log(
      message,
      name: 'CalibrationService',
      level: level == 'error' ? 1000 : (level == 'warn' ? 900 : 800),
    );
  }

  @override
  void dispose() {
    super.dispose();
  }
}
