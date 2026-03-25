import 'dart:convert';
import 'dart:developer' as developer;
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

import '../models/match.dart';

/// Manages match lifecycle and persistence.
///
/// Matches are stored as individual JSON files on disk for offline-first
/// operation. No backend is required for core functionality.
///
/// File layout:
///   {appDocuments}/bt_matches/{matchId}.json
class MatchService extends ChangeNotifier {
  Directory? _matchesDir;
  final List<Match> _matches = [];
  Match? _currentMatch;

  /// All stored matches, newest first.
  List<Match> get matches => List.unmodifiable(_matches);

  /// The currently active match, or null if no match is in progress.
  Match? get currentMatch => _currentMatch;

  /// Initializes the match storage directory and loads existing matches.
  Future<void> initialize() async {
    final appDir = await getApplicationDocumentsDirectory();
    _matchesDir = Directory(p.join(appDir.path, 'bt_matches'));

    if (!await _matchesDir!.exists()) {
      await _matchesDir!.create(recursive: true);
    }

    await _loadMatches();
    _log('info', 'MatchService initialized: ${_matches.length} matches found');
  }

  /// Creates a new match and sets it as the current active match.
  ///
  /// Returns the newly created [Match].
  Future<Match> createMatch() async {
    if (_matchesDir == null) {
      await initialize();
    }

    final match = Match.create();
    _matches.insert(0, match);
    _currentMatch = match;

    await _saveMatch(match);
    _log('info', 'Created match: ${match.id} "${match.name}"');
    notifyListeners();
    return match;
  }

  /// Returns the current active match, or null.
  Match? getCurrentMatch() => _currentMatch;

  /// Returns all matches stored locally, newest first.
  List<Match> listMatches() => List.unmodifiable(_matches);

  /// Updates match metadata (e.g., after a new clip is saved).
  Future<void> updateMatch(Match updatedMatch) async {
    final index = _matches.indexWhere((m) => m.id == updatedMatch.id);
    if (index == -1) return;

    _matches[index] = updatedMatch;
    if (_currentMatch?.id == updatedMatch.id) {
      _currentMatch = updatedMatch;
    }

    await _saveMatch(updatedMatch);
    notifyListeners();
  }

  /// Ends the current match session.
  void endCurrentMatch() {
    if (_currentMatch != null) {
      _log('info', 'Ended match: ${_currentMatch!.id}');
      _currentMatch = null;
      notifyListeners();
    }
  }

  /// Deletes a match and its JSON file from disk.
  Future<bool> deleteMatch(int matchId) async {
    final index = _matches.indexWhere((m) => m.id == matchId);
    if (index == -1) return false;

    final match = _matches.removeAt(index);
    if (_currentMatch?.id == matchId) {
      _currentMatch = null;
    }

    try {
      final file = File(_matchFilePath(match.id));
      if (await file.exists()) {
        await file.delete();
      }
      _log('info', 'Deleted match: ${match.id}');
      notifyListeners();
      return true;
    } catch (e) {
      _log('error', 'Failed to delete match file: $e');
      return false;
    }
  }

  // --- Private helpers ---

  String _matchFilePath(int matchId) {
    return p.join(_matchesDir!.path, '$matchId.json');
  }

  Future<void> _saveMatch(Match match) async {
    try {
      final file = File(_matchFilePath(match.id));
      await file.writeAsString(jsonEncode(match.toJson()));
    } catch (e) {
      _log('error', 'Failed to save match ${match.id}: $e');
    }
  }

  Future<void> _loadMatches() async {
    if (_matchesDir == null) return;

    try {
      final entities = await _matchesDir!.list().toList();
      _matches.clear();

      for (final entity in entities) {
        if (entity is File && entity.path.endsWith('.json')) {
          try {
            final content = await entity.readAsString();
            final json = jsonDecode(content) as Map<String, dynamic>;
            _matches.add(Match.fromJson(json));
          } catch (e) {
            _log('warn', 'Failed to parse match file: ${entity.path}');
          }
        }
      }

      // Sort newest first.
      _matches.sort((a, b) => b.createdAt.compareTo(a.createdAt));
    } catch (e) {
      _log('error', 'Failed to load matches: $e');
    }
  }

  void _log(String level, String message) {
    developer.log(
      message,
      name: 'MatchService',
      level: level == 'error' ? 1000 : (level == 'warn' ? 900 : 800),
    );
  }

  @override
  void dispose() {
    super.dispose();
  }
}
