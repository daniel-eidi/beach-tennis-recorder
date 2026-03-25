import 'dart:developer' as developer;

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Persists and exposes all user-configurable settings for the app.
///
/// Defaults match the constants defined in CLAUDE.md. Changes are saved
/// immediately to SharedPreferences and trigger UI rebuilds via
/// ChangeNotifier.
///
/// Implements TASK-01-14 (settings persistence).
class SettingsService extends ChangeNotifier {
  SharedPreferences? _prefs;
  bool _isInitialized = false;

  bool get isInitialized => _isInitialized;

  // ─── Keys ───────────────────────────────────────────────────────────────

  static const _kResolution = 'settings_resolution';
  static const _kFps = 'settings_fps';
  static const _kBufferDuration = 'settings_buffer_duration';
  static const _kConfidenceThreshold = 'settings_confidence_threshold';
  static const _kVelocityThreshold = 'settings_velocity_threshold';
  static const _kPreRallySeconds = 'settings_pre_rally_seconds';
  static const _kPostRallySeconds = 'settings_post_rally_seconds';
  static const _kTimeoutSeconds = 'settings_timeout_seconds';
  static const _kNetCrossRequired = 'settings_net_cross_required';
  static const _kActiveCalibrationId = 'settings_active_calibration_id';

  // ─── Defaults (from CLAUDE.md) ──────────────────────────────────────────

  static const String defaultResolution = '1080p';
  static const int defaultFps = 30;
  static const int defaultBufferDuration = 60;
  static const double defaultConfidenceThreshold = 0.45;
  static const double defaultVelocityThreshold = 15.0;
  static const int defaultPreRallySeconds = 3;
  static const int defaultPostRallySeconds = 2;
  static const int defaultTimeoutSeconds = 8;
  static const bool defaultNetCrossRequired = true;

  // ─── Initialization ─────────────────────────────────────────────────────

  /// Loads settings from disk. Must be called before accessing any setting.
  Future<void> initialize() async {
    _prefs = await SharedPreferences.getInstance();
    _isInitialized = true;
    _log('info', 'SettingsService initialized');
    notifyListeners();
  }

  // ─── Camera settings ────────────────────────────────────────────────────

  /// Camera resolution preset: '720p', '1080p', or '4K'.
  String get resolution =>
      _prefs?.getString(_kResolution) ?? defaultResolution;

  set resolution(String value) {
    _prefs?.setString(_kResolution, value);
    _log('info', 'Resolution set to $value');
    notifyListeners();
  }

  /// Camera frames per second: 24, 30, or 60.
  int get fps => _prefs?.getInt(_kFps) ?? defaultFps;

  set fps(int value) {
    _prefs?.setInt(_kFps, value);
    _log('info', 'FPS set to $value');
    notifyListeners();
  }

  // ─── Buffer settings ────────────────────────────────────────────────────

  /// Circular buffer duration in seconds: 30, 60, or 90.
  int get bufferDuration =>
      _prefs?.getInt(_kBufferDuration) ?? defaultBufferDuration;

  set bufferDuration(int value) {
    _prefs?.setInt(_kBufferDuration, value);
    _log('info', 'Buffer duration set to ${value}s');
    notifyListeners();
  }

  // ─── Detection settings ─────────────────────────────────────────────────

  /// Minimum model confidence to accept a detection (0.1 – 0.9).
  double get confidenceThreshold =>
      _prefs?.getDouble(_kConfidenceThreshold) ?? defaultConfidenceThreshold;

  set confidenceThreshold(double value) {
    _prefs?.setDouble(_kConfidenceThreshold, value);
    _log('info', 'Confidence threshold set to $value');
    notifyListeners();
  }

  /// Minimum ball velocity in px/frame to consider it in motion (5 – 30).
  double get velocityThreshold =>
      _prefs?.getDouble(_kVelocityThreshold) ?? defaultVelocityThreshold;

  set velocityThreshold(double value) {
    _prefs?.setDouble(_kVelocityThreshold, value);
    _log('info', 'Velocity threshold set to $value');
    notifyListeners();
  }

  // ─── Rally settings ─────────────────────────────────────────────────────

  /// Seconds of video to capture before the serve (1 – 5).
  int get preRallySeconds =>
      _prefs?.getInt(_kPreRallySeconds) ?? defaultPreRallySeconds;

  set preRallySeconds(int value) {
    _prefs?.setInt(_kPreRallySeconds, value);
    _log('info', 'Pre-rally seconds set to $value');
    notifyListeners();
  }

  /// Seconds of video to capture after rally ends (1 – 5).
  int get postRallySeconds =>
      _prefs?.getInt(_kPostRallySeconds) ?? defaultPostRallySeconds;

  set postRallySeconds(int value) {
    _prefs?.setInt(_kPostRallySeconds, value);
    _log('info', 'Post-rally seconds set to $value');
    notifyListeners();
  }

  /// Maximum seconds without a ball detection before ending the rally (5 – 15).
  int get timeoutSeconds =>
      _prefs?.getInt(_kTimeoutSeconds) ?? defaultTimeoutSeconds;

  set timeoutSeconds(int value) {
    _prefs?.setInt(_kTimeoutSeconds, value);
    _log('info', 'Timeout seconds set to $value');
    notifyListeners();
  }

  /// Whether the ball must cross the net to confirm a rally.
  bool get netCrossRequired =>
      _prefs?.getBool(_kNetCrossRequired) ?? defaultNetCrossRequired;

  set netCrossRequired(bool value) {
    _prefs?.setBool(_kNetCrossRequired, value);
    _log('info', 'Net cross required set to $value');
    notifyListeners();
  }

  // ─── Calibration ────────────────────────────────────────────────────────

  /// ID of the active court calibration, or null if none is set.
  String? get activeCalibrationId =>
      _prefs?.getString(_kActiveCalibrationId);

  set activeCalibrationId(String? value) {
    if (value == null) {
      _prefs?.remove(_kActiveCalibrationId);
    } else {
      _prefs?.setString(_kActiveCalibrationId, value);
    }
    _log('info', 'Active calibration set to $value');
    notifyListeners();
  }

  // ─── Reset ──────────────────────────────────────────────────────────────

  /// Resets all settings to their default values.
  Future<void> resetToDefaults() async {
    await _prefs?.remove(_kResolution);
    await _prefs?.remove(_kFps);
    await _prefs?.remove(_kBufferDuration);
    await _prefs?.remove(_kConfidenceThreshold);
    await _prefs?.remove(_kVelocityThreshold);
    await _prefs?.remove(_kPreRallySeconds);
    await _prefs?.remove(_kPostRallySeconds);
    await _prefs?.remove(_kTimeoutSeconds);
    await _prefs?.remove(_kNetCrossRequired);
    // Note: active calibration is NOT reset — it is location-specific.
    _log('info', 'All settings reset to defaults');
    notifyListeners();
  }

  // ─── Logging ────────────────────────────────────────────────────────────

  void _log(String level, String message) {
    developer.log(
      message,
      name: 'SettingsService',
      level: level == 'error' ? 1000 : (level == 'warn' ? 900 : 800),
    );
  }
}
