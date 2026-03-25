"""
Beach Tennis Recorder - Ball Physics Unit Tests
AGENT-02 | Sprint 2

Tests for tracking/ball_physics.py:
- Speed estimation (px/frame, m/s, km/h)
- Trajectory direction
- Shot type classification (serve, lob, smash, drive, drop)
- Full track analysis
"""

import math
import sys
from pathlib import Path
from typing import List

import pytest

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from vision.tracking.ball_physics import (
    BallPhysicsAnalyzer,
    CourtCalibration,
    ShotType,
)
from vision.tracking.byte_tracker import Detection, Track


def _make_track(
    positions: List[tuple],
    class_id: int = 0,
    track_id: int = 1,
    confidence: float = 0.85,
) -> Track:
    """Helper: create a Track with a sequence of positions."""
    track = Track(track_id=track_id, class_id=class_id)
    for x, y in positions:
        det = Detection(float(x), float(y), 15.0, 15.0, confidence, class_id)
        track.update(det)
    return track


class TestCourtCalibration:
    """Tests for CourtCalibration."""

    def test_default_values(self) -> None:
        cal = CourtCalibration()
        assert cal.court_length_m == 16.0
        assert cal.court_width_m == 8.0

    def test_px_per_meter(self) -> None:
        cal = CourtCalibration(
            frame_court_x_min=0, frame_court_x_max=800,
            frame_court_y_min=0, frame_court_y_max=1600,
            court_width_m=8.0, court_length_m=16.0,
        )
        assert cal.px_per_meter_x == pytest.approx(100.0)
        assert cal.px_per_meter_y == pytest.approx(100.0)


class TestBallPhysicsAnalyzer:
    """Tests for BallPhysicsAnalyzer."""

    def test_speed_px_per_frame(self) -> None:
        analyzer = BallPhysicsAnalyzer(fps=30.0)
        track = _make_track([(100, 200), (120, 200), (140, 200)])
        speed = analyzer.estimate_speed_px_per_frame(track)
        assert speed > 0

    def test_speed_ms(self) -> None:
        cal = CourtCalibration(
            frame_court_x_min=0, frame_court_x_max=540,
            court_width_m=8.0,
        )
        analyzer = BallPhysicsAnalyzer(calibration=cal, fps=30.0)
        track = _make_track([(100, 200), (120, 200), (140, 200)])
        speed_ms = analyzer.estimate_speed_ms(track)
        assert speed_ms > 0

    def test_speed_kmh(self) -> None:
        analyzer = BallPhysicsAnalyzer(fps=30.0)
        track = _make_track([(100, 200), (120, 200), (140, 200)])
        speed_kmh = analyzer.estimate_speed_kmh(track)
        # km/h should be 3.6x m/s
        speed_ms = analyzer.estimate_speed_ms(track)
        assert speed_kmh == pytest.approx(speed_ms * 3.6, rel=0.01)

    def test_stationary_ball_speed_zero(self) -> None:
        analyzer = BallPhysicsAnalyzer(fps=30.0)
        track = _make_track([(100, 200)])
        assert analyzer.estimate_speed_px_per_frame(track) == 0.0

    def test_trajectory_direction_right(self) -> None:
        analyzer = BallPhysicsAnalyzer()
        track = _make_track([(100, 200), (120, 200), (140, 200)])
        direction = analyzer.get_trajectory_direction(track)
        assert direction is not None
        assert abs(direction) < 15  # Moving right ~ 0 degrees

    def test_trajectory_direction_down(self) -> None:
        analyzer = BallPhysicsAnalyzer()
        track = _make_track([(200, 100), (200, 120), (200, 140)])
        direction = analyzer.get_trajectory_direction(track)
        assert direction is not None
        assert abs(direction - 90) < 15  # Moving down ~ 90 degrees

    def test_trajectory_direction_up(self) -> None:
        analyzer = BallPhysicsAnalyzer()
        track = _make_track([(200, 140), (200, 120), (200, 100)])
        direction = analyzer.get_trajectory_direction(track)
        assert direction is not None
        assert abs(direction + 90) < 15  # Moving up ~ -90 degrees

    def test_trajectory_direction_stationary(self) -> None:
        analyzer = BallPhysicsAnalyzer()
        track = _make_track([(200, 200)])
        direction = analyzer.get_trajectory_direction(track)
        assert direction is None

    def test_is_moving_up(self) -> None:
        analyzer = BallPhysicsAnalyzer()
        track = _make_track([(200, 200), (200, 180), (200, 160)])
        assert analyzer.is_moving_up(track) is True
        assert analyzer.is_moving_down(track) is False

    def test_is_moving_down(self) -> None:
        analyzer = BallPhysicsAnalyzer()
        track = _make_track([(200, 100), (200, 120), (200, 140)])
        assert analyzer.is_moving_down(track) is True
        assert analyzer.is_moving_up(track) is False

    def test_detect_serve(self) -> None:
        cal = CourtCalibration(
            frame_court_y_min=80, frame_court_y_max=560,
        )
        analyzer = BallPhysicsAnalyzer(calibration=cal, fps=30.0)

        # Serve: fast horizontal from near baseline
        track = _make_track([
            (100, 500),   # Near bottom baseline
            (130, 495),
            (160, 490),
            (190, 485),
        ])
        assert analyzer.detect_serve(track, min_speed_px=10.0) is True

    def test_not_serve_from_center(self) -> None:
        cal = CourtCalibration(
            frame_court_y_min=80, frame_court_y_max=560,
        )
        analyzer = BallPhysicsAnalyzer(calibration=cal, fps=30.0)

        # Fast but from center court - not a serve
        track = _make_track([
            (100, 320),   # Center of court
            (130, 320),
            (160, 320),
            (190, 320),
        ])
        assert analyzer.detect_serve(track, min_speed_px=10.0) is False

    def test_detect_lob(self) -> None:
        analyzer = BallPhysicsAnalyzer(fps=30.0)

        # Lob: ball going up with horizontal component
        track = _make_track([
            (200, 400),
            (210, 380),
            (220, 360),
            (230, 340),
            (240, 320),
        ])
        assert analyzer.detect_lob(track, min_upward_speed=5.0) is True

    def test_not_lob_going_down(self) -> None:
        analyzer = BallPhysicsAnalyzer(fps=30.0)

        # Ball going down is not a lob
        track = _make_track([
            (200, 200),
            (210, 220),
            (220, 240),
            (230, 260),
        ])
        assert analyzer.detect_lob(track) is False

    def test_detect_smash(self) -> None:
        cal = CourtCalibration(
            frame_court_y_min=80, frame_court_y_max=560,
        )
        analyzer = BallPhysicsAnalyzer(calibration=cal, fps=30.0)

        # Smash: fast downward from upper court area
        track = _make_track([
            (300, 150),
            (310, 185),
            (320, 220),
        ])
        # Need high speed, set velocities manually
        track._vy = 30.0  # Fast downward
        track._vx = 5.0

        assert analyzer.detect_smash(
            track, min_downward_speed=15.0, min_total_speed=20.0
        ) is True

    def test_not_smash_from_lower_court(self) -> None:
        cal = CourtCalibration(
            frame_court_y_min=80, frame_court_y_max=560,
        )
        analyzer = BallPhysicsAnalyzer(calibration=cal, fps=30.0)

        # Fast downward but from lower half of court
        track = _make_track([
            (300, 400),
            (310, 430),
            (320, 460),
        ])
        track._vy = 30.0
        track._vx = 5.0

        assert analyzer.detect_smash(track) is False

    def test_classify_shot_unknown_short_trajectory(self) -> None:
        analyzer = BallPhysicsAnalyzer()
        track = _make_track([(100, 200)])
        assert analyzer.classify_shot(track) == ShotType.UNKNOWN

    def test_classify_shot_drive(self) -> None:
        analyzer = BallPhysicsAnalyzer()
        # Fast horizontal ball
        track = _make_track([
            (100, 300),
            (120, 302),
            (140, 304),
            (160, 306),
        ])
        shot = analyzer.classify_shot(track)
        assert shot == ShotType.DRIVE

    def test_analyze_track(self) -> None:
        analyzer = BallPhysicsAnalyzer(fps=30.0)
        track = _make_track([
            (100, 200),
            (120, 210),
            (140, 220),
            (160, 230),
        ])

        result = analyzer.analyze_track(track)

        assert result["track_id"] == 1
        assert result["trajectory_length"] == 4
        assert result["speed_px_per_frame"] > 0
        assert result["speed_ms"] > 0
        assert result["speed_kmh"] > 0
        assert result["direction_degrees"] is not None
        assert result["shot_type"] in [s.value for s in ShotType]
        assert "velocity" in result
        assert "trajectory_bounds" in result

    def test_analyze_track_bounds(self) -> None:
        analyzer = BallPhysicsAnalyzer()
        track = _make_track([(100, 200), (200, 300), (150, 250)])
        result = analyzer.analyze_track(track)

        bounds = result["trajectory_bounds"]
        assert bounds["x_min"] == pytest.approx(100.0, abs=0.5)
        assert bounds["x_max"] == pytest.approx(200.0, abs=0.5)
        assert bounds["y_min"] == pytest.approx(200.0, abs=0.5)
        assert bounds["y_max"] == pytest.approx(300.0, abs=0.5)
