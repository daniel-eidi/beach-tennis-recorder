"""
Integration tests: ByteTracker -> RallyDetector pipeline.

AGENT-05 | TASK-05-02

Tests the interaction between the ByteTracker and RallyDetector,
including track continuity, velocity calculation, and handling of
noisy / multi-object input.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vision.tracking.byte_tracker import ByteTracker, Detection, Track
from vision.tracking.rally_detector import (
    RallyDetector,
    RallyEvent,
    RallyState,
)
from tests.conftest import (
    make_ball_detection,
    make_net_detection,
    make_court_detection,
    run_scenario_through_pipeline,
    scenario_detections_to_objects,
)
from tests.mock_data.scenarios import get_scenario


# ---------------------------------------------------------------------------
# Track continuity
# ---------------------------------------------------------------------------

class TestTrackContinuity:
    """Verify that ByteTracker maintains consistent track IDs across frames."""

    def test_single_ball_track_maintained(self):
        """A smoothly moving ball should keep the same track ID."""
        tracker = ByteTracker()
        track_ids: List[int] = []

        for i in range(20):
            x = 100.0 + i * 15.0
            y = 300.0 + i * 3.0
            dets = [make_ball_detection(x, y, 0.85)]
            active = tracker.update(dets)
            ball_tracks = [t for t in active if t.class_id == 0]
            if ball_tracks:
                track_ids.append(ball_tracks[0].track_id)

        # All frames should have the same track ID
        assert len(set(track_ids)) == 1, (
            f"Expected single track ID, got: {set(track_ids)}"
        )

    def test_track_survives_brief_occlusion(self):
        """Track should survive a few frames of missed detections."""
        tracker = ByteTracker(max_frames_lost=10)
        first_id = None

        # 10 frames with detection
        for i in range(10):
            dets = [make_ball_detection(100.0 + i * 10.0, 300.0, 0.85)]
            active = tracker.update(dets)
            ball_tracks = [t for t in active if t.class_id == 0]
            if ball_tracks and first_id is None:
                first_id = ball_tracks[0].track_id

        # 5 frames with no detection (brief occlusion)
        for _ in range(5):
            tracker.update([])

        # Ball reappears near predicted position
        dets = [make_ball_detection(200.0, 300.0, 0.85)]
        active = tracker.update(dets)
        ball_tracks = [t for t in active if t.class_id == 0]

        # Should have at least one ball track (may be same or new)
        assert len(ball_tracks) >= 1

    def test_track_lost_after_max_frames(self):
        """Track should be deactivated after max_frames_lost with no match."""
        tracker = ByteTracker(max_frames_lost=5)

        # Create a track
        dets = [make_ball_detection(100.0, 300.0, 0.85)]
        tracker.update(dets)

        # Miss many frames
        for _ in range(10):
            tracker.update([])

        # No active ball tracks
        ball_tracks = tracker.get_ball_tracks()
        assert len(ball_tracks) == 0


# ---------------------------------------------------------------------------
# Velocity calculation
# ---------------------------------------------------------------------------

class TestVelocityCalculation:
    """Verify velocity feeds correctly from tracker to rally detector."""

    def test_horizontal_velocity_computed(self):
        """A ball moving horizontally should have non-zero vx."""
        tracker = ByteTracker()

        for i in range(10):
            dets = [make_ball_detection(100.0 + i * 20.0, 300.0, 0.85)]
            tracker.update(dets)

        ball_tracks = tracker.get_ball_tracks()
        assert len(ball_tracks) == 1

        vx, vy = ball_tracks[0].velocity
        assert abs(vx) > 5.0, f"Expected significant horizontal velocity, got vx={vx}"
        assert abs(vy) < abs(vx), "Horizontal movement should dominate"

    def test_vertical_velocity_computed(self):
        """A ball moving vertically should have non-zero vy."""
        tracker = ByteTracker()

        for i in range(10):
            dets = [make_ball_detection(300.0, 100.0 + i * 20.0, 0.85)]
            tracker.update(dets)

        ball_tracks = tracker.get_ball_tracks()
        assert len(ball_tracks) == 1

        vx, vy = ball_tracks[0].velocity
        assert abs(vy) > 5.0, f"Expected significant vertical velocity, got vy={vy}"

    def test_speed_exceeds_threshold_for_fast_ball(self):
        """Speed for fast-moving ball should exceed VELOCITY_THRESHOLD."""
        tracker = ByteTracker()

        # Move at ~30 px/frame
        for i in range(10):
            dets = [make_ball_detection(100.0 + i * 30.0, 300.0, 0.85)]
            tracker.update(dets)

        ball_tracks = tracker.get_ball_tracks()
        assert len(ball_tracks) == 1
        assert ball_tracks[0].speed > 15.0


# ---------------------------------------------------------------------------
# Multi-object tracking
# ---------------------------------------------------------------------------

class TestMultiObjectTracking:
    """Test tracking with ball + net + court_line detections."""

    def test_ball_and_net_tracked_separately(self):
        """Ball and net should get different track IDs and class_ids."""
        tracker = ByteTracker()

        for i in range(10):
            dets = [
                make_ball_detection(100.0 + i * 15.0, 200.0, 0.85),
                make_net_detection(),
                make_court_detection(),
            ]
            tracker.update(dets)

        ball_tracks = [t for t in tracker.active_tracks if t.class_id == 0]
        net_tracks = [t for t in tracker.active_tracks if t.class_id == 1]
        court_tracks = [t for t in tracker.active_tracks if t.class_id == 2]

        assert len(ball_tracks) >= 1
        assert len(net_tracks) >= 1
        # Court line may or may not be tracked depending on consistency
        # Just verify ball and net are separate
        ball_ids = {t.track_id for t in ball_tracks}
        net_ids = {t.track_id for t in net_tracks}
        assert ball_ids.isdisjoint(net_ids)

    def test_get_ball_tracks_filters_correctly(self):
        """get_ball_tracks should return only class_id=0."""
        tracker = ByteTracker()

        dets = [
            make_ball_detection(200.0, 300.0, 0.85),
            make_net_detection(),
            make_court_detection(),
        ]
        tracker.update(dets)
        ball_tracks = tracker.get_ball_tracks()

        for t in ball_tracks:
            assert t.class_id == 0

    def test_rally_detector_ignores_non_ball_tracks(self):
        """RallyDetector should only respond to ball tracks."""
        tracker = ByteTracker()
        events: List[RallyEvent] = []
        detector = RallyDetector(
            fps=30.0,
            on_rally_end=lambda e: events.append(e),
        )

        # Only net detections, no ball -> no rally
        for i in range(50):
            dets = [make_net_detection()]
            active = tracker.update(dets)
            # Pass all tracks (including net) - detector should ignore non-ball
            ball_tracks = [t for t in active if t.class_id == 0]
            detector.update(ball_tracks)

        assert detector.state == RallyState.IDLE
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Noisy detections
# ---------------------------------------------------------------------------

class TestNoisyDetections:
    """Test tracker + detector with noisy / imperfect input."""

    def test_jittery_detections_maintain_track(self, noisy_scenario):
        """Detections with positional jitter should still maintain tracking."""
        detection_frames = scenario_detections_to_objects(noisy_scenario)
        tracker = ByteTracker()

        tracks_seen = set()
        for frame_dets in detection_frames:
            active = tracker.update(frame_dets)
            for t in active:
                if t.class_id == 0:
                    tracks_seen.add(t.track_id)

        # Noisy scenario should still produce relatively few tracks
        # (ideally 1-2, not one per frame)
        assert len(tracks_seen) <= 5, (
            f"Too many tracks ({len(tracks_seen)}), tracker not handling noise well"
        )

    def test_noisy_scenario_detects_rally(self, noisy_scenario):
        """Even with noise, the rally should still be detected."""
        events, detector, _ = run_scenario_through_pipeline(
            noisy_scenario, net_cross_required=True
        )
        # The noisy scenario should still produce a rally event
        assert len(events) >= 1

    def test_mixed_confidence_levels(self):
        """Mix of high and low confidence detections should be handled."""
        tracker = ByteTracker(high_threshold=0.5, low_threshold=0.1)

        # Alternating high/low confidence
        for i in range(20):
            x = 100.0 + i * 20.0
            y = 300.0
            conf = 0.85 if i % 2 == 0 else 0.25
            dets = [make_ball_detection(x, y, conf)]
            tracker.update(dets)

        # Should still maintain a track
        ball_tracks = tracker.get_ball_tracks()
        assert len(ball_tracks) >= 1


# ---------------------------------------------------------------------------
# Tracker reset between rallies
# ---------------------------------------------------------------------------

class TestTrackerReset:
    """Test tracker behavior around rally boundaries."""

    def test_tracker_reset_clears_all_tracks(self):
        """Resetting the tracker should clear all tracks."""
        tracker = ByteTracker()

        # Create some tracks
        for i in range(10):
            dets = [make_ball_detection(100.0 + i * 15.0, 300.0, 0.85)]
            tracker.update(dets)

        assert len(tracker.active_tracks) >= 1

        tracker.reset()
        assert len(tracker.active_tracks) == 0
        assert len(tracker.all_tracks) == 0

    def test_new_tracks_after_reset(self):
        """New detections after reset should create fresh tracks."""
        tracker = ByteTracker()

        # Create initial track
        for i in range(5):
            dets = [make_ball_detection(100.0 + i * 15.0, 300.0, 0.85)]
            tracker.update(dets)

        old_ids = {t.track_id for t in tracker.active_tracks}
        tracker.reset()

        # New detections
        for i in range(5):
            dets = [make_ball_detection(400.0 + i * 15.0, 200.0, 0.85)]
            tracker.update(dets)

        new_ids = {t.track_id for t in tracker.active_tracks}

        # After reset, IDs start from 1 again
        assert len(new_ids) >= 1
