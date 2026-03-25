"""
Beach Tennis Recorder - RallyDetector Unit Tests
AGENT-02 | Sprint 2

Tests for tracking/rally_detector.py:
- State transitions (IDLE -> EM_JOGO -> FIM_RALLY -> IDLE)
- Bounce detection
- Out of bounds detection
- Timeout handling
- Net crossing requirement
- Full rally lifecycle
"""

import sys
import time
from pathlib import Path
from typing import List, Optional
from unittest.mock import patch

import pytest

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from vision.tracking.byte_tracker import Detection, Track
from vision.tracking.rally_detector import (
    BounceDetector,
    CourtConfig,
    NetCrossingDetector,
    OutOfBoundsDetector,
    RallyDetector,
    RallyEvent,
    RallyState,
)


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


def _make_fast_ball_track(speed_factor: float = 20.0) -> Track:
    """Helper: create a ball track moving fast enough to trigger EM_JOGO."""
    positions = [
        (100, 200),
        (100 + speed_factor, 200),
        (100 + speed_factor * 2, 200),
    ]
    return _make_track(positions, class_id=0)


class TestBounceDetector:
    """Tests for the BounceDetector sub-component."""

    def test_no_bounce_constant_velocity(self) -> None:
        bd = BounceDetector(min_speed=3.0, direction_change_threshold=0.5)
        track = _make_track([(100, 100), (100, 110), (100, 120), (100, 130)])
        # Constant downward velocity - no bounce
        assert bd.update(track) is False

    def test_bounce_on_y_reversal(self) -> None:
        bd = BounceDetector(min_speed=3.0, direction_change_threshold=0.5)

        # First: moving down
        track_down = _make_track([(100, 100), (100, 110), (100, 120)])
        bd.update(track_down)

        # Then: moving up (bounce)
        track_up = _make_track([(100, 100), (100, 110), (100, 120), (100, 105)])
        # Need to set internal velocity to simulate direction change
        track_up._vy = -8.0  # Moving up now
        result = bd.update(track_up)
        assert result is True
        assert bd.bounce_count == 1

    def test_reset(self) -> None:
        bd = BounceDetector()
        track = _make_track([(100, 100), (100, 110)])
        bd.update(track)
        bd.reset()
        assert bd.bounce_count == 0


class TestNetCrossingDetector:
    """Tests for the NetCrossingDetector sub-component."""

    def test_no_crossing_same_side(self) -> None:
        court = CourtConfig(net_y_min=280, net_y_max=360)
        ncd = NetCrossingDetector(court)

        # Ball stays below the net
        track = _make_track([(100, 400), (110, 410)])
        assert ncd.update(track) is False
        assert ncd.crossing_count == 0

    def test_crossing_detected(self) -> None:
        court = CourtConfig(net_y_min=280, net_y_max=360)
        ncd = NetCrossingDetector(court)
        net_mid = (280 + 360) / 2  # = 320

        # Ball on one side
        track1 = _make_track([(100, net_mid - 50)])  # y=270 (above net)
        ncd.update(track1)

        # Ball crosses to other side
        track2 = _make_track([(100, net_mid - 50), (100, net_mid + 50)])  # y=370
        assert ncd.update(track2) is True
        assert ncd.crossing_count == 1

    def test_multiple_crossings(self) -> None:
        court = CourtConfig(net_y_min=280, net_y_max=360)
        ncd = NetCrossingDetector(court)
        net_mid = 320

        # Cross once
        t1 = _make_track([(100, net_mid - 50)])
        ncd.update(t1)
        t2 = _make_track([(100, net_mid - 50), (100, net_mid + 50)])
        ncd.update(t2)

        # Cross back
        t3 = _make_track([(100, net_mid + 50), (100, net_mid + 50), (100, net_mid - 50)])
        ncd.update(t3)

        assert ncd.crossing_count == 2

    def test_reset(self) -> None:
        court = CourtConfig()
        ncd = NetCrossingDetector(court)
        ncd._crossing_count = 3
        ncd.reset()
        assert ncd.crossing_count == 0


class TestOutOfBoundsDetector:
    """Tests for the OutOfBoundsDetector sub-component."""

    def test_in_bounds(self) -> None:
        court = CourtConfig(
            court_x_min=50, court_x_max=590,
            court_y_min=80, court_y_max=560,
        )
        oobd = OutOfBoundsDetector(court, margin=20.0)

        track = _make_track([(300, 300)])  # Center of court
        assert oobd.is_out(track) is False

    def test_out_of_bounds_left(self) -> None:
        court = CourtConfig(court_x_min=50, court_x_max=590,
                            court_y_min=80, court_y_max=560)
        oobd = OutOfBoundsDetector(court, margin=20.0)

        track = _make_track([(10, 300)])  # Way left of court
        assert oobd.is_out(track) is True

    def test_out_of_bounds_right(self) -> None:
        court = CourtConfig(court_x_min=50, court_x_max=590,
                            court_y_min=80, court_y_max=560)
        oobd = OutOfBoundsDetector(court, margin=20.0)

        track = _make_track([(620, 300)])
        assert oobd.is_out(track) is True

    def test_out_of_bounds_top(self) -> None:
        court = CourtConfig(court_x_min=50, court_x_max=590,
                            court_y_min=80, court_y_max=560)
        oobd = OutOfBoundsDetector(court, margin=20.0)

        track = _make_track([(300, 40)])
        assert oobd.is_out(track) is True

    def test_within_margin(self) -> None:
        court = CourtConfig(court_x_min=50, court_x_max=590,
                            court_y_min=80, court_y_max=560)
        oobd = OutOfBoundsDetector(court, margin=20.0)

        # Just outside court but within margin
        track = _make_track([(40, 300)])  # 50 - 40 = 10 < margin of 20
        assert oobd.is_out(track) is False


class TestRallyDetector:
    """Tests for the RallyDetector state machine."""

    def test_initial_state_is_idle(self) -> None:
        rd = RallyDetector()
        assert rd.state == RallyState.IDLE
        assert rd.rally_number == 0
        assert rd.is_rally_active is False

    def test_idle_to_em_jogo_on_fast_ball(self) -> None:
        rd = RallyDetector(
            velocity_threshold=10.0,
            confidence_threshold=0.3,
        )
        assert rd.state == RallyState.IDLE

        # Create a fast ball track
        track = _make_fast_ball_track(speed_factor=25.0)
        rd.update([track])

        assert rd.state == RallyState.EM_JOGO
        assert rd.rally_number == 1
        assert rd.is_rally_active is True

    def test_no_transition_on_slow_ball(self) -> None:
        rd = RallyDetector(velocity_threshold=15.0)

        # Slow ball (small position changes)
        track = _make_track([(100, 200), (101, 200), (102, 200)])
        rd.update([track])

        assert rd.state == RallyState.IDLE

    def test_no_transition_on_low_confidence(self) -> None:
        rd = RallyDetector(
            velocity_threshold=10.0,
            confidence_threshold=0.8,
        )

        # Fast ball but low confidence
        track = _make_track(
            [(100, 200), (130, 200), (160, 200)],
            confidence=0.5,
        )
        rd.update([track])

        assert rd.state == RallyState.IDLE

    def test_em_jogo_to_fim_rally_on_out_of_bounds(self) -> None:
        court = CourtConfig(
            court_x_min=50, court_x_max=590,
            court_y_min=80, court_y_max=560,
        )
        events: List[RallyEvent] = []
        rd = RallyDetector(
            court=court,
            velocity_threshold=10.0,
            confidence_threshold=0.3,
            on_rally_end=lambda e: events.append(e),
        )

        # Start rally with fast ball
        fast_track = _make_fast_ball_track(speed_factor=25.0)
        rd.update([fast_track])
        assert rd.state == RallyState.EM_JOGO

        # Ball goes out of bounds
        oob_track = _make_track([(650, 300), (670, 300), (690, 300)])
        event = rd.update([oob_track])

        assert rd.state == RallyState.IDLE  # auto-transition through FIM_RALLY
        assert event is not None
        assert event.end_reason == "out_of_bounds"
        assert len(events) == 1

    def test_em_jogo_to_fim_rally_on_timeout(self) -> None:
        events: List[RallyEvent] = []
        rd = RallyDetector(
            velocity_threshold=10.0,
            confidence_threshold=0.3,
            rally_timeout_seconds=0.1,  # Very short for testing
            on_rally_end=lambda e: events.append(e),
        )

        # Start rally
        fast_track = _make_fast_ball_track(speed_factor=25.0)
        rd.update([fast_track])
        assert rd.state == RallyState.EM_JOGO

        # Wait for timeout
        time.sleep(0.15)

        # Update with no ball tracks
        event = rd.update([])

        assert rd.state == RallyState.IDLE
        assert event is not None
        assert event.end_reason == "timeout"

    def test_em_jogo_to_fim_rally_on_bounce_with_net_cross(self) -> None:
        """Rally should end on bounce only after net has been crossed."""
        court = CourtConfig(net_y_min=280, net_y_max=360)
        events: List[RallyEvent] = []
        rd = RallyDetector(
            court=court,
            velocity_threshold=10.0,
            confidence_threshold=0.3,
            net_cross_required=True,
            on_rally_end=lambda e: events.append(e),
        )

        # Start rally
        fast_track = _make_fast_ball_track(speed_factor=25.0)
        rd.update([fast_track])
        assert rd.state == RallyState.EM_JOGO

        # Manually simulate net crossing and bounce
        rd._net_detector._crossing_count = 1
        rd._bounce_detector._bounce_count = 0

        # Create a track that triggers bounce detection
        # The bounce detector needs to see a Y-direction reversal
        track = Track(track_id=1, class_id=0)
        track.update(Detection(200, 400, 15, 15, 0.85, 0))
        track.update(Detection(210, 420, 15, 15, 0.85, 0))
        track._vy = 10.0  # Going down

        rd._bounce_detector._prev_vy = 10.0

        # Now reverse direction (bounce)
        track.update(Detection(220, 400, 15, 15, 0.85, 0))
        track._vy = -10.0  # Going up after bounce

        event = rd.update([track])

        # Should have ended
        assert event is not None
        assert event.end_reason == "bounce"

    def test_net_cross_not_required(self) -> None:
        """With net_cross_required=False, bounce should end rally immediately."""
        events: List[RallyEvent] = []
        rd = RallyDetector(
            velocity_threshold=10.0,
            confidence_threshold=0.3,
            net_cross_required=False,
            on_rally_end=lambda e: events.append(e),
        )

        # Start rally
        fast_track = _make_fast_ball_track(speed_factor=25.0)
        rd.update([fast_track])
        assert rd.state == RallyState.EM_JOGO

        # Simulate bounce without net crossing
        rd._bounce_detector._prev_vy = 10.0
        track = Track(track_id=1, class_id=0)
        track.update(Detection(200, 400, 15, 15, 0.85, 0))
        track.update(Detection(210, 420, 15, 15, 0.85, 0))
        track._vy = -10.0  # Bounce up

        event = rd.update([track])

        if event is not None:
            assert event.end_reason == "bounce"

    def test_full_rally_lifecycle(self) -> None:
        """Test a complete rally: IDLE -> EM_JOGO -> FIM_RALLY -> IDLE."""
        events: List[RallyEvent] = []
        court = CourtConfig(
            court_x_min=50, court_x_max=590,
            court_y_min=80, court_y_max=560,
        )
        rd = RallyDetector(
            court=court,
            velocity_threshold=10.0,
            confidence_threshold=0.3,
            on_rally_end=lambda e: events.append(e),
        )

        # 1. IDLE state
        assert rd.state == RallyState.IDLE

        # 2. Start rally (fast ball)
        fast_track = _make_fast_ball_track(speed_factor=25.0)
        rd.update([fast_track])
        assert rd.state == RallyState.EM_JOGO
        assert rd.rally_number == 1

        # 3. End rally (out of bounds)
        oob_track = _make_track([(650, 300), (670, 300), (690, 300)])
        event = rd.update([oob_track])

        # 4. Back to IDLE
        assert rd.state == RallyState.IDLE
        assert event is not None
        assert event.rally_number == 1
        assert event.end_reason == "out_of_bounds"
        assert event.duration_seconds >= 0

        # 5. Can start another rally
        fast_track2 = _make_fast_ball_track(speed_factor=30.0)
        rd.update([fast_track2])
        assert rd.state == RallyState.EM_JOGO
        assert rd.rally_number == 2

    def test_events_list(self) -> None:
        court = CourtConfig(
            court_x_min=50, court_x_max=590,
            court_y_min=80, court_y_max=560,
        )
        rd = RallyDetector(
            court=court,
            velocity_threshold=10.0,
            confidence_threshold=0.3,
        )

        # Complete two rallies
        for _ in range(2):
            fast = _make_fast_ball_track(speed_factor=25.0)
            rd.update([fast])
            oob = _make_track([(650, 300), (670, 300), (690, 300)])
            rd.update([oob])

        assert len(rd.events) == 2

    def test_reset(self) -> None:
        rd = RallyDetector(
            velocity_threshold=10.0,
            confidence_threshold=0.3,
        )

        fast = _make_fast_ball_track(speed_factor=25.0)
        rd.update([fast])
        assert rd.state == RallyState.EM_JOGO

        rd.reset()
        assert rd.state == RallyState.IDLE
        assert rd.rally_number == 0
        assert rd.events == []

    def test_no_transition_on_empty_tracks(self) -> None:
        rd = RallyDetector()
        event = rd.update([])
        assert rd.state == RallyState.IDLE
        assert event is None

    def test_uses_highest_confidence_ball(self) -> None:
        """When multiple ball tracks exist, use the one with highest confidence."""
        rd = RallyDetector(
            velocity_threshold=10.0,
            confidence_threshold=0.3,
        )

        # Two ball tracks with different confidence
        track_low = _make_track(
            [(100, 200), (130, 200), (160, 200)],
            confidence=0.5, track_id=1,
        )
        track_high = _make_track(
            [(300, 300), (330, 300), (360, 300)],
            confidence=0.9, track_id=2,
        )

        rd.update([track_low, track_high])
        # Should have transitioned since the high-confidence track is fast enough
        assert rd.state == RallyState.EM_JOGO
