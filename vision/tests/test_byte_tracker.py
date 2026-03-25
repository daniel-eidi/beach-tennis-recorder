"""
Beach Tennis Recorder - ByteTracker Unit Tests
AGENT-02 | Sprint 2

Tests for tracking/byte_tracker.py:
- Track creation
- Track association across frames
- Velocity calculation
- Track loss and removal
- Multiple simultaneous tracks
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is on path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from vision.tracking.byte_tracker import ByteTracker, Detection, Track


class TestDetection:
    """Tests for the Detection dataclass."""

    def test_center_property(self) -> None:
        det = Detection(100.0, 200.0, 30.0, 30.0, 0.9, class_id=0)
        assert det.center == (100.0, 200.0)

    def test_tlbr_property(self) -> None:
        det = Detection(100.0, 200.0, 30.0, 40.0, 0.9, class_id=0)
        top, left, bottom, right = det.tlbr
        assert left == pytest.approx(85.0)
        assert top == pytest.approx(180.0)
        assert right == pytest.approx(115.0)
        assert bottom == pytest.approx(220.0)


class TestTrack:
    """Tests for the Track dataclass."""

    def test_initial_state(self) -> None:
        track = Track(track_id=1, class_id=0)
        assert track.track_id == 1
        assert track.class_id == 0
        assert track.is_active is True
        assert track.last_detection is None
        assert track.position is None
        assert track.speed == 0.0
        assert track.velocity == (0.0, 0.0)
        assert track.trajectory == []

    def test_update_single_detection(self) -> None:
        track = Track(track_id=1, class_id=0)
        det = Detection(100.0, 200.0, 15.0, 15.0, 0.85, class_id=0)
        track.update(det)

        assert track.last_detection is det
        assert track.position == (100.0, 200.0)
        assert track.frames_since_update == 0
        assert len(track.detections) == 1

    def test_velocity_after_two_updates(self) -> None:
        track = Track(track_id=1, class_id=0)
        det1 = Detection(100.0, 200.0, 15.0, 15.0, 0.85, class_id=0)
        det2 = Detection(120.0, 210.0, 15.0, 15.0, 0.85, class_id=0)

        track.update(det1)
        track.update(det2)

        vx, vy = track.velocity
        # With alpha=0.6, first update: vx = 0.6 * 20 + 0.4 * 0 = 12
        assert vx == pytest.approx(12.0)
        assert vy == pytest.approx(6.0)
        assert track.speed > 0

    def test_trajectory(self) -> None:
        track = Track(track_id=1, class_id=0)
        positions = [(100, 200), (120, 210), (140, 220)]
        for x, y in positions:
            track.update(Detection(float(x), float(y), 15.0, 15.0, 0.85, 0))

        traj = track.trajectory
        assert len(traj) == 3
        assert traj[0] == (100.0, 200.0)
        assert traj[-1] == (140.0, 220.0)

    def test_predict(self) -> None:
        track = Track(track_id=1, class_id=0)
        track.update(Detection(100.0, 200.0, 15.0, 15.0, 0.85, 0))
        track.update(Detection(120.0, 210.0, 15.0, 15.0, 0.85, 0))

        pred_x, pred_y = track.predict()
        # Predicted = last position + velocity
        assert pred_x == pytest.approx(120.0 + 12.0)
        assert pred_y == pytest.approx(210.0 + 6.0)

    def test_mark_missed(self) -> None:
        track = Track(track_id=1, class_id=0)
        det = Detection(100.0, 200.0, 15.0, 15.0, 0.85, 0)
        track.update(det)

        track.mark_missed()
        assert track.frames_since_update == 1

        track.mark_missed()
        assert track.frames_since_update == 2


class TestByteTracker:
    """Tests for the ByteTracker class."""

    def test_single_detection_creates_track(self) -> None:
        tracker = ByteTracker()
        dets = [Detection(100.0, 200.0, 15.0, 15.0, 0.85, class_id=0)]
        active = tracker.update(dets)

        assert len(active) == 1
        assert active[0].track_id == 1
        assert active[0].class_id == 0
        assert active[0].position == (100.0, 200.0)

    def test_track_association_across_frames(self) -> None:
        tracker = ByteTracker(match_distance_threshold=100.0)

        # Frame 1
        dets1 = [Detection(100.0, 200.0, 15.0, 15.0, 0.85, class_id=0)]
        active1 = tracker.update(dets1)
        track_id = active1[0].track_id

        # Frame 2: ball moved slightly
        dets2 = [Detection(115.0, 205.0, 15.0, 15.0, 0.82, class_id=0)]
        active2 = tracker.update(dets2)

        # Should maintain same track ID
        assert len(active2) == 1
        assert active2[0].track_id == track_id
        assert len(active2[0].detections) == 2

    def test_velocity_calculation(self) -> None:
        tracker = ByteTracker()

        # Frame 1
        tracker.update([Detection(100.0, 200.0, 15.0, 15.0, 0.85, 0)])
        # Frame 2
        active = tracker.update([Detection(120.0, 210.0, 15.0, 15.0, 0.85, 0)])

        track = active[0]
        assert track.speed > 0
        vx, vy = track.velocity
        assert vx > 0  # Moving right
        assert vy > 0  # Moving down

    def test_track_loss_after_max_frames(self) -> None:
        max_lost = 5
        tracker = ByteTracker(max_frames_lost=max_lost)

        # Create a track
        tracker.update([Detection(100.0, 200.0, 15.0, 15.0, 0.85, 0)])

        # Send empty frames until track is lost
        for _ in range(max_lost + 1):
            active = tracker.update([])

        assert len(active) == 0

    def test_track_survives_below_max_lost(self) -> None:
        max_lost = 10
        tracker = ByteTracker(max_frames_lost=max_lost)

        tracker.update([Detection(100.0, 200.0, 15.0, 15.0, 0.85, 0)])

        # Miss a few frames but stay under threshold
        for _ in range(max_lost - 2):
            active = tracker.update([])

        assert len(active) == 1
        assert active[0].frames_since_update == max_lost - 2

    def test_multiple_simultaneous_tracks(self) -> None:
        tracker = ByteTracker(match_distance_threshold=50.0)

        # Two detections far apart
        dets = [
            Detection(100.0, 100.0, 15.0, 15.0, 0.9, class_id=0),
            Detection(400.0, 400.0, 15.0, 15.0, 0.85, class_id=1),
        ]
        active = tracker.update(dets)

        assert len(active) == 2
        ids = {t.track_id for t in active}
        assert len(ids) == 2  # Different IDs

    def test_multiple_tracks_maintained(self) -> None:
        tracker = ByteTracker(match_distance_threshold=80.0)

        # Frame 1: two objects
        tracker.update([
            Detection(100.0, 100.0, 15.0, 15.0, 0.9, 0),
            Detection(400.0, 400.0, 15.0, 15.0, 0.85, 0),
        ])

        # Frame 2: both objects move slightly
        active = tracker.update([
            Detection(110.0, 105.0, 15.0, 15.0, 0.88, 0),
            Detection(410.0, 405.0, 15.0, 15.0, 0.83, 0),
        ])

        assert len(active) == 2
        for t in active:
            assert len(t.detections) == 2

    def test_get_ball_tracks(self) -> None:
        tracker = ByteTracker()

        dets = [
            Detection(100.0, 200.0, 15.0, 15.0, 0.9, class_id=0),   # ball
            Detection(300.0, 300.0, 200.0, 50.0, 0.95, class_id=1),  # net
        ]
        tracker.update(dets)

        ball_tracks = tracker.get_ball_tracks()
        assert len(ball_tracks) == 1
        assert ball_tracks[0].class_id == 0

    def test_reset(self) -> None:
        tracker = ByteTracker()
        tracker.update([Detection(100.0, 200.0, 15.0, 15.0, 0.85, 0)])
        assert len(tracker.active_tracks) == 1

        tracker.reset()
        assert len(tracker.active_tracks) == 0
        assert len(tracker.all_tracks) == 0

    def test_low_confidence_detection_not_creating_new_track(self) -> None:
        tracker = ByteTracker(high_threshold=0.5, low_threshold=0.1)

        # Low confidence detection should not create a new track
        dets = [Detection(100.0, 200.0, 15.0, 15.0, 0.3, class_id=0)]
        active = tracker.update(dets)

        assert len(active) == 0

    def test_low_confidence_matches_existing_track(self) -> None:
        tracker = ByteTracker(
            high_threshold=0.5, low_threshold=0.1,
            match_distance_threshold=100.0,
        )

        # Frame 1: High confidence creates track
        tracker.update([Detection(100.0, 200.0, 15.0, 15.0, 0.85, 0)])

        # Frame 2: Low confidence should match existing track
        active = tracker.update([Detection(105.0, 202.0, 15.0, 15.0, 0.3, 0)])

        assert len(active) == 1
        assert len(active[0].detections) == 2
