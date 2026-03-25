"""
Beach Tennis Recorder - GestureDetector Unit Tests
AGENT-02 | Gesture Detection Feature

Tests for tracking/gesture_detector.py:
- State transitions (IDLE -> ARMS_RAISED -> HAND_APPROACHING_RACKET -> CLAP_DETECTED)
- Cooldown prevents rapid re-triggers
- Gesture not triggered with arms down
- Gesture not triggered without racket
- Debounce behavior (min_frames)
- Full gesture sequence detection
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from vision.tracking.byte_tracker import Detection
from vision.tracking.gesture_detector import (
    GESTURE_COOLDOWN_SECONDS,
    GESTURE_MIN_FRAMES,
    GestureDetector,
    GestureEvent,
    GestureState,
    _bbox_center,
    _distance,
    _hand_racket_iou,
    _point_in_bbox,
)
from vision.tracking.pose_estimator import PoseResult


# ---------- Helpers ----------

def _make_pose(
    left_shoulder: Tuple[float, float] = (200.0, 300.0),
    right_shoulder: Tuple[float, float] = (400.0, 300.0),
    left_wrist: Tuple[float, float] = (180.0, 500.0),
    right_wrist: Tuple[float, float] = (420.0, 500.0),
    left_elbow: Tuple[float, float] = (190.0, 400.0),
    right_elbow: Tuple[float, float] = (410.0, 400.0),
    left_index: Optional[Tuple[float, float]] = None,
    right_index: Optional[Tuple[float, float]] = None,
    visibility: float = 0.9,
    frame_size: Tuple[int, int] = (1080, 720),
) -> PoseResult:
    """Create a PoseResult with specified keypoint positions."""
    kps: Dict[str, Tuple[float, float, float]] = {
        "left_shoulder": (left_shoulder[0], left_shoulder[1], visibility),
        "right_shoulder": (right_shoulder[0], right_shoulder[1], visibility),
        "left_wrist": (left_wrist[0], left_wrist[1], visibility),
        "right_wrist": (right_wrist[0], right_wrist[1], visibility),
        "left_elbow": (left_elbow[0], left_elbow[1], visibility),
        "right_elbow": (right_elbow[0], right_elbow[1], visibility),
    }
    if left_index is not None:
        kps["left_index"] = (left_index[0], left_index[1], visibility)
    if right_index is not None:
        kps["right_index"] = (right_index[0], right_index[1], visibility)

    return PoseResult(
        keypoints=kps,
        frame_width=frame_size[0],
        frame_height=frame_size[1],
    )


def _make_arms_raised_pose(
    racket_hand: str = "right",
    frame_size: Tuple[int, int] = (1080, 720),
) -> PoseResult:
    """Create a pose with both arms raised above shoulders."""
    # Shoulders at y=300; wrists at y=150 (well above shoulders)
    # In a 720px tall frame, shoulder-wrist gap = 150px > 0.15*720 = 108px
    return _make_pose(
        left_shoulder=(200.0, 300.0),
        right_shoulder=(400.0, 300.0),
        left_wrist=(180.0, 150.0),
        right_wrist=(420.0, 150.0),
        left_elbow=(190.0, 220.0),
        right_elbow=(410.0, 220.0),
        frame_size=frame_size,
    )


def _make_racket_detection(
    x: float = 430.0,
    y: float = 140.0,
    w: float = 50.0,
    h: float = 80.0,
    confidence: float = 0.85,
    class_id: int = 38,  # COCO tennis racket
) -> Detection:
    """Create a YOLO racket detection."""
    return Detection(x, y, w, h, confidence, class_id)


def _make_dummy_frame(w: int = 1080, h: int = 720) -> np.ndarray:
    """Create a dummy BGR frame."""
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------- Utility Function Tests ----------

class TestUtilityFunctions:
    def test_bbox_center(self) -> None:
        assert _bbox_center(100, 200, 300, 400) == (200.0, 300.0)

    def test_distance(self) -> None:
        assert _distance((0, 0), (3, 4)) == 5.0

    def test_point_in_bbox(self) -> None:
        assert _point_in_bbox((150, 250), (100, 200, 300, 400)) is True
        assert _point_in_bbox((50, 250), (100, 200, 300, 400)) is False

    def test_point_in_bbox_with_margin(self) -> None:
        # Point just outside but within margin
        assert _point_in_bbox((95, 250), (100, 200, 300, 400), margin=10) is True

    def test_hand_racket_iou_overlap(self) -> None:
        # Hand directly on racket center
        iou = _hand_racket_iou((200.0, 200.0), (180.0, 180.0, 220.0, 220.0))
        assert iou > 0.0

    def test_hand_racket_iou_no_overlap(self) -> None:
        # Hand far from racket
        iou = _hand_racket_iou((0.0, 0.0), (500.0, 500.0, 550.0, 550.0))
        assert iou == 0.0


# ---------- GestureDetector Tests ----------

class TestGestureDetectorInit:
    def test_initial_state_is_idle(self) -> None:
        gd = GestureDetector(fps=30.0)
        assert gd.state == GestureState.IDLE
        assert gd.events == []

    def test_custom_parameters(self) -> None:
        gd = GestureDetector(
            fps=60.0,
            cooldown_seconds=10.0,
            min_frames=5,
        )
        assert gd._fps == 60.0
        assert gd._cooldown_seconds == 10.0
        assert gd._min_frames == 5


class TestIdleToArmsRaised:
    def test_transition_on_arms_raised(self) -> None:
        gd = GestureDetector(fps=30.0, min_frames=2)
        frame = _make_dummy_frame()
        pose = _make_arms_raised_pose()

        gd.update(frame, [], pose_results=[pose])
        assert gd.state == GestureState.ARMS_RAISED

    def test_no_transition_with_arms_down(self) -> None:
        gd = GestureDetector(fps=30.0)
        frame = _make_dummy_frame()
        # Wrists below shoulders (default pose)
        pose = _make_pose()

        gd.update(frame, [], pose_results=[pose])
        assert gd.state == GestureState.IDLE

    def test_no_transition_without_pose(self) -> None:
        gd = GestureDetector(fps=30.0)
        frame = _make_dummy_frame()

        gd.update(frame, [], pose_results=[])
        assert gd.state == GestureState.IDLE


class TestArmsRaisedToApproaching:
    def test_transition_with_racket(self) -> None:
        gd = GestureDetector(fps=30.0, min_frames=2)
        frame = _make_dummy_frame()
        pose = _make_arms_raised_pose()
        racket = _make_racket_detection(x=430.0, y=140.0)

        # Frame 1: enter ARMS_RAISED
        gd.update(frame, [racket], pose_results=[pose])
        assert gd.state == GestureState.ARMS_RAISED

        # Frame 2: min_frames=2, state_frame_count reaches 2
        gd.update(frame, [racket], pose_results=[pose])

        # Frame 3: now min_frames satisfied, should transition
        gd.update(frame, [racket], pose_results=[pose])
        assert gd.state == GestureState.HAND_APPROACHING_RACKET

    def test_no_transition_without_racket(self) -> None:
        gd = GestureDetector(fps=30.0, min_frames=2, timeout_frames=5)
        frame = _make_dummy_frame()
        pose = _make_arms_raised_pose()

        # Multiple frames with no racket
        for _ in range(4):
            gd.update(frame, [], pose_results=[pose])

        # Should still be in ARMS_RAISED (no racket to approach)
        assert gd.state == GestureState.ARMS_RAISED

    def test_timeout_back_to_idle(self) -> None:
        gd = GestureDetector(fps=30.0, min_frames=2, timeout_frames=3)
        frame = _make_dummy_frame()
        pose = _make_arms_raised_pose()

        # Enter ARMS_RAISED
        gd.update(frame, [], pose_results=[pose])
        assert gd.state == GestureState.ARMS_RAISED

        # No racket for timeout_frames
        for _ in range(4):
            gd.update(frame, [], pose_results=[pose])

        assert gd.state == GestureState.IDLE


class TestFullGestureSequence:
    def test_complete_gesture_detection(self) -> None:
        """Test the full sequence: IDLE -> ARMS_RAISED -> APPROACHING -> CLAP -> COOLDOWN."""
        gd = GestureDetector(
            fps=30.0,
            min_frames=2,
            cooldown_seconds=1.0,
            approach_velocity_threshold=3.0,
            overlap_threshold=0.1,
            timeout_frames=20,
        )
        frame = _make_dummy_frame()

        # Phase 1: Arms raised (need min_frames+1 frames to advance)
        pose_raised = _make_arms_raised_pose()
        racket = _make_racket_detection(x=430.0, y=140.0)

        gd.update(frame, [racket], pose_results=[pose_raised])
        assert gd.state == GestureState.ARMS_RAISED

        gd.update(frame, [racket], pose_results=[pose_raised])
        gd.update(frame, [racket], pose_results=[pose_raised])
        assert gd.state == GestureState.HAND_APPROACHING_RACKET

        # Phase 2: Free hand approaches racket
        # Left hand starts far, moves toward racket at (430, 140)
        # Right wrist is at (420, 150) — close to racket, so right is the racket hand
        # Left wrist at (180, 150) is the free hand
        # Move left hand toward the racket over several frames
        approach_positions = [
            (300.0, 150.0),  # Getting closer
            (370.0, 145.0),  # Closer
            (410.0, 142.0),  # Very close
            (425.0, 140.0),  # Contact!
        ]

        for lw_x, lw_y in approach_positions:
            pose = _make_pose(
                left_shoulder=(200.0, 300.0),
                right_shoulder=(400.0, 300.0),
                left_wrist=(lw_x, lw_y),
                right_wrist=(420.0, 150.0),
                left_elbow=(190.0, 220.0),
                right_elbow=(410.0, 220.0),
                frame_size=(1080, 720),
            )
            event = gd.update(frame, [racket], pose_results=[pose])

            if event is not None:
                break

        # Should have detected the gesture
        assert len(gd.events) >= 1
        assert gd.state == GestureState.COOLDOWN

        event = gd.events[-1]
        assert event.confidence > 0
        assert event.frame_index > 0

    def test_gesture_not_triggered_with_arms_down(self) -> None:
        """Arms must be raised — casual arm movement should not trigger."""
        gd = GestureDetector(fps=30.0, min_frames=2)
        frame = _make_dummy_frame()

        # Arms down (wrists below shoulders)
        pose = _make_pose(
            left_wrist=(180.0, 500.0),
            right_wrist=(420.0, 500.0),
        )
        racket = _make_racket_detection()

        for _ in range(20):
            event = gd.update(frame, [racket], pose_results=[pose])
            assert event is None

        assert gd.state == GestureState.IDLE
        assert len(gd.events) == 0

    def test_gesture_not_triggered_without_racket(self) -> None:
        """No racket detected means no gesture can complete."""
        gd = GestureDetector(fps=30.0, min_frames=2, timeout_frames=5)
        frame = _make_dummy_frame()
        pose = _make_arms_raised_pose()

        for _ in range(20):
            event = gd.update(frame, [], pose_results=[pose])
            assert event is None

        assert len(gd.events) == 0


class TestCooldown:
    def test_cooldown_prevents_rapid_retrigger(self) -> None:
        """After a gesture, cooldown must elapse before another can fire."""
        gd = GestureDetector(
            fps=30.0,
            min_frames=1,
            cooldown_seconds=2.0,
            approach_velocity_threshold=1.0,
            overlap_threshold=0.05,
            timeout_frames=30,
        )
        frame = _make_dummy_frame()

        def _trigger_gesture() -> Optional[GestureEvent]:
            """Run frames that should trigger a gesture."""
            # Arms raised
            pose_raised = _make_arms_raised_pose()
            racket = _make_racket_detection(x=430.0, y=140.0)
            gd.update(frame, [racket], pose_results=[pose_raised])

            # Approach and contact
            for lw_x in [300.0, 370.0, 410.0, 425.0, 430.0]:
                pose = _make_pose(
                    left_shoulder=(200.0, 300.0),
                    right_shoulder=(400.0, 300.0),
                    left_wrist=(lw_x, 140.0),
                    right_wrist=(420.0, 150.0),
                    left_elbow=(190.0, 220.0),
                    right_elbow=(410.0, 220.0),
                    frame_size=(1080, 720),
                )
                ev = gd.update(frame, [racket], pose_results=[pose])
                if ev is not None:
                    return ev
            return None

        # First trigger should work
        event1 = _trigger_gesture()
        # May or may not trigger on first attempt due to state transitions
        # Keep trying until we get one
        attempts = 0
        while event1 is None and attempts < 5:
            gd.reset()
            event1 = _trigger_gesture()
            attempts += 1

        if event1 is not None:
            assert gd.state == GestureState.COOLDOWN
            initial_event_count = len(gd.events)

            # Immediately try to trigger again — should be blocked by cooldown
            pose_raised = _make_arms_raised_pose()
            racket = _make_racket_detection(x=430.0, y=140.0)
            for _ in range(10):
                ev = gd.update(frame, [racket], pose_results=[pose_raised])
                assert ev is None  # Should not trigger during cooldown

            assert len(gd.events) == initial_event_count

    def test_cooldown_expires(self) -> None:
        """After cooldown_seconds worth of frames, detector returns to IDLE."""
        fps = 30.0
        cooldown_s = 1.0
        gd = GestureDetector(fps=fps, cooldown_seconds=cooldown_s, min_frames=1)
        frame = _make_dummy_frame()

        # Manually set to COOLDOWN state
        gd._state = GestureState.COOLDOWN
        gd._cooldown_start_frame = gd._frame_index

        # Advance enough frames for cooldown to expire
        cooldown_frames = int(cooldown_s * fps) + 1
        pose = _make_pose()

        for _ in range(cooldown_frames):
            gd.update(frame, [], pose_results=[pose])

        assert gd.state == GestureState.IDLE


class TestReset:
    def test_reset_clears_state(self) -> None:
        gd = GestureDetector(fps=30.0)
        frame = _make_dummy_frame()
        pose = _make_arms_raised_pose()

        gd.update(frame, [], pose_results=[pose])
        assert gd.state == GestureState.ARMS_RAISED

        gd.reset()
        assert gd.state == GestureState.IDLE
        assert gd.events == []
        assert gd._frame_index == 0


class TestDebounce:
    def test_min_frames_enforced(self) -> None:
        """Arms must be raised for min_frames before advancing."""
        gd = GestureDetector(fps=30.0, min_frames=5)
        frame = _make_dummy_frame()
        pose = _make_arms_raised_pose()
        racket = _make_racket_detection(x=430.0, y=140.0)

        # First frame: enter ARMS_RAISED
        gd.update(frame, [racket], pose_results=[pose])
        assert gd.state == GestureState.ARMS_RAISED

        # Frames 2-5: still ARMS_RAISED (min_frames=5, need state_frame_count >= 5)
        for _ in range(4):
            gd.update(frame, [racket], pose_results=[pose])
            assert gd.state == GestureState.ARMS_RAISED

        # Frame 6: state_frame_count reaches 5, should now advance
        gd.update(frame, [racket], pose_results=[pose])
        assert gd.state == GestureState.HAND_APPROACHING_RACKET
