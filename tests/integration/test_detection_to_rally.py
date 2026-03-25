"""
Integration tests: Detection -> Rally pipeline.

AGENT-05 | TASK-05-02

Tests the rally state machine transitions driven by detection sequences.
"""

from __future__ import annotations

import sys
import unittest.mock as mock
from pathlib import Path
from typing import List

import pytest

# Ensure project root is importable
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vision.tracking.byte_tracker import ByteTracker, Detection
from vision.tracking.rally_detector import (
    CourtConfig,
    RallyDetector,
    RallyEvent,
    RallyState,
    VELOCITY_THRESHOLD,
    RALLY_TIMEOUT_SECONDS,
)
from tests.conftest import (
    make_ball_detection,
    run_scenario_through_pipeline,
    scenario_detections_to_objects,
)
from tests.mock_data.scenarios import (
    Scenario,
    get_scenario,
)


# ---------------------------------------------------------------------------
# State machine transition tests
# ---------------------------------------------------------------------------

class TestRallyStateTransitions:
    """Tests for rally state machine transition logic."""

    def test_initial_state_is_idle(self):
        detector = RallyDetector(fps=30.0)
        assert detector.state == RallyState.IDLE

    def test_fast_ball_triggers_em_jogo(self):
        """A sequence of detections showing fast ball movement should
        transition from IDLE to EM_JOGO."""
        tracker = ByteTracker()
        detector = RallyDetector(fps=30.0, velocity_threshold=15.0)

        # Feed fast-moving ball detections (>15 px/frame displacement)
        positions = [
            (100.0, 400.0), (120.0, 380.0), (140.0, 360.0),
            (160.0, 340.0), (180.0, 320.0),
        ]
        for x, y in positions:
            dets = [Detection(x, y, 15.0, 15.0, 0.85, class_id=0)]
            active = tracker.update(dets)
            ball_tracks = [t for t in active if t.class_id == 0]
            detector.update(ball_tracks)

        assert detector.state == RallyState.EM_JOGO

    def test_slow_ball_stays_idle(self):
        """Movement below velocity threshold should not trigger EM_JOGO."""
        tracker = ByteTracker()
        detector = RallyDetector(fps=30.0, velocity_threshold=15.0)

        # Slow movement: ~2 px/frame
        positions = [
            (300.0, 400.0), (302.0, 400.0), (304.0, 400.0),
            (306.0, 400.0), (308.0, 400.0), (310.0, 400.0),
        ]
        for x, y in positions:
            dets = [Detection(x, y, 15.0, 15.0, 0.85, class_id=0)]
            active = tracker.update(dets)
            ball_tracks = [t for t in active if t.class_id == 0]
            detector.update(ball_tracks)

        assert detector.state == RallyState.IDLE

    def test_low_confidence_stays_idle(self):
        """Detections below confidence threshold should not trigger rally."""
        tracker = ByteTracker()
        detector = RallyDetector(fps=30.0, confidence_threshold=0.45)

        # Fast movement but low confidence (0.30 < 0.45)
        positions = [
            (100.0, 400.0), (130.0, 370.0), (160.0, 340.0),
            (190.0, 310.0), (220.0, 280.0),
        ]
        for x, y in positions:
            dets = [Detection(x, y, 15.0, 15.0, 0.30, class_id=0)]
            active = tracker.update(dets)
            ball_tracks = [t for t in active if t.class_id == 0]
            detector.update(ball_tracks)

        assert detector.state == RallyState.IDLE


class TestBounceDetection:
    """Tests for bounce (Y-direction reversal) ending a rally."""

    def test_bounce_after_net_cross_ends_rally(self, normal_rally_scenario):
        """Ball crosses net then bounces -> rally should end with 'bounce'."""
        events, detector, _ = run_scenario_through_pipeline(
            normal_rally_scenario, net_cross_required=True
        )
        assert len(events) >= 1
        assert events[0].end_reason == "bounce"
        assert events[0].net_crossings >= 1
        assert events[0].ball_bounces >= 1

    def test_bounce_without_net_cross_no_end_when_required(self):
        """When NET_CROSS_REQUIRED=True, bounce alone should NOT end rally
        if net has not been crossed."""
        tracker = ByteTracker()
        events: List[RallyEvent] = []
        detector = RallyDetector(
            fps=30.0,
            on_rally_end=lambda e: events.append(e),
            net_cross_required=True,
        )

        # Fast ball on one side only (no net crossing), then bounce
        # Ball moves fast on the bottom side (y > 320) only
        positions = [
            (100.0, 500.0), (120.0, 480.0), (140.0, 460.0),
            (160.0, 440.0), (180.0, 420.0), (200.0, 400.0),
            # Bounce: Y reversal, but still on same side of net
            (210.0, 420.0), (220.0, 440.0), (225.0, 450.0),
            (228.0, 440.0), (230.0, 425.0),
        ]
        for x, y in positions:
            dets = [Detection(x, y, 15.0, 15.0, 0.85, class_id=0)]
            active = tracker.update(dets)
            ball_tracks = [t for t in active if t.class_id == 0]
            detector.update(ball_tracks)

        # Rally should still be active (no net crossing -> bounce doesn't end it)
        assert detector.state == RallyState.EM_JOGO
        assert len(events) == 0

    def test_bounce_ends_rally_when_net_cross_not_required(self):
        """When NET_CROSS_REQUIRED=False, bounce alone should end rally."""
        tracker = ByteTracker()
        events: List[RallyEvent] = []
        detector = RallyDetector(
            fps=30.0,
            on_rally_end=lambda e: events.append(e),
            net_cross_required=False,
        )

        # Fast ball then bounce, no net crossing
        positions = [
            (100.0, 500.0), (120.0, 475.0), (140.0, 450.0),
            (160.0, 425.0), (180.0, 400.0),
            # Bounce
            (190.0, 420.0), (200.0, 445.0), (205.0, 460.0),
            (208.0, 445.0), (210.0, 420.0),
        ]
        for x, y in positions:
            dets = [Detection(x, y, 15.0, 15.0, 0.85, class_id=0)]
            active = tracker.update(dets)
            ball_tracks = [t for t in active if t.class_id == 0]
            detector.update(ball_tracks)

        assert len(events) >= 1
        assert events[0].end_reason == "bounce"


class TestOutOfBounds:
    """Tests for ball going out of bounds ending a rally."""

    def test_ball_out_of_bounds_ends_rally(self, out_of_bounds_scenario):
        events, detector, _ = run_scenario_through_pipeline(
            out_of_bounds_scenario, net_cross_required=True
        )
        # Ball should go OOB and end rally
        assert len(events) >= 1
        assert events[0].end_reason == "out_of_bounds"

    def test_serve_ace_out_of_bounds(self, serve_ace_scenario):
        events, detector, _ = run_scenario_through_pipeline(
            serve_ace_scenario, net_cross_required=True
        )
        assert len(events) >= 1
        assert events[0].end_reason == "out_of_bounds"


class TestTimeout:
    """Tests for timeout ending a rally."""

    def test_timeout_ends_rally(self, timeout_scenario):
        """Ball disappears for >RALLY_TIMEOUT_SECONDS -> rally ends."""
        events, detector, _ = run_scenario_through_pipeline(
            timeout_scenario,
            simulate_timeout=True,
            timeout_after_frame=6,  # after the moving frames
            timeout_seconds=9.0,
        )
        assert len(events) >= 1
        assert events[0].end_reason == "timeout"


class TestFullLifecycle:
    """Tests for complete rally lifecycle: IDLE -> EM_JOGO -> FIM_RALLY -> IDLE."""

    def test_normal_rally_lifecycle(self, normal_rally_scenario):
        events, detector, _ = run_scenario_through_pipeline(
            normal_rally_scenario, net_cross_required=True
        )
        # At least one rally should have completed the full cycle
        assert len(events) >= 1
        assert events[0].rally_number == 1
        assert events[0].duration_seconds >= 0
        # The first rally should have transitioned through all states:
        # IDLE -> EM_JOGO -> (end_rally) -> IDLE
        # (Note: remaining frames may start another rally, so detector
        # may not be IDLE at the very end of the scenario.)

    def test_multiple_consecutive_rallies(self, multiple_rallies_scenario):
        """Two rallies in sequence should both be detected."""
        events, detector, _ = run_scenario_through_pipeline(
            multiple_rallies_scenario, net_cross_required=True
        )
        # Should detect at least 2 rallies (may detect more if remaining
        # frames continue to generate events)
        assert len(events) >= 2, (
            f"Expected >= 2 rallies, got {len(events)}: "
            f"{[(e.rally_number, e.end_reason) for e in events]}"
        )
        assert events[0].rally_number == 1
        assert events[1].rally_number == 2

    def test_rally_number_increments(self, multiple_rallies_scenario):
        events, detector, _ = run_scenario_through_pipeline(
            multiple_rallies_scenario, net_cross_required=True
        )
        numbers = [e.rally_number for e in events]
        assert numbers == sorted(numbers)
        # Numbers should be sequential
        for i in range(1, len(numbers)):
            assert numbers[i] == numbers[i - 1] + 1


class TestFalsePositiveScenarios:
    """Tests for scenarios that should NOT trigger rallies."""

    def test_warmup_does_not_trigger_rally(self, warmup_scenario):
        """Slow warmup movement should not start a rally."""
        events, detector, _ = run_scenario_through_pipeline(
            warmup_scenario, velocity_threshold=15.0
        )
        assert len(events) == 0
        assert detector.state == RallyState.IDLE

    def test_no_detections_stays_idle(self):
        """Frames with no detections should keep the detector idle."""
        detector = RallyDetector(fps=30.0)
        for _ in range(100):
            event = detector.update([])
            assert event is None
        assert detector.state == RallyState.IDLE

    def test_net_only_detections_stay_idle(self):
        """Only net detections (no ball) should not trigger a rally."""
        tracker = ByteTracker()
        detector = RallyDetector(fps=30.0)

        for _ in range(50):
            dets = [Detection(320.0, 320.0, 540.0, 80.0, 0.90, class_id=1)]
            active = tracker.update(dets)
            ball_tracks = [t for t in active if t.class_id == 0]
            detector.update(ball_tracks)

        assert detector.state == RallyState.IDLE


class TestResetBehavior:
    """Tests for detector reset between sessions."""

    def test_reset_clears_state(self, normal_rally_scenario):
        events, detector, _ = run_scenario_through_pipeline(
            normal_rally_scenario, net_cross_required=True
        )
        assert len(events) >= 1

        detector.reset()
        assert detector.state == RallyState.IDLE
        assert detector.rally_number == 0
        assert len(detector.events) == 0
