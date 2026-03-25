"""
Beach Tennis Recorder - Gesture Detection (Palm-to-Racket Clap)
AGENT-02 | Gesture Detection Feature

Detects when a player raises their arms and claps their hand against the
racket face. This gesture acts as a manual trigger to save the last 30
seconds of video as a highlight clip.

State machine:
  IDLE -> ARMS_RAISED -> HAND_APPROACHING_RACKET -> CLAP_DETECTED -> COOLDOWN -> IDLE

Uses MediaPipe Pose (via PoseEstimator) for body keypoints and YOLO
detections for racket bounding boxes.
"""

import enum
import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .byte_tracker import Detection
from .pose_estimator import PoseEstimator, PoseResult


AGENT = "02"
TASK = "gesture-detect"

# ---------- Constants ----------

GESTURE_COOLDOWN_SECONDS: float = 5.0
GESTURE_MIN_FRAMES: int = 3
ARM_RAISED_THRESHOLD: float = 0.15   # wrist must be this fraction above shoulder Y
HAND_RACKET_OVERLAP_THRESHOLD: float = 0.3  # IoU threshold for hand-racket contact
APPROACH_VELOCITY_THRESHOLD: float = 5.0    # px/frame minimum approach speed
TIMEOUT_FRAMES: int = 15  # max frames to stay in a transient state without progress

# YOLO class IDs (from quick_youtube_test: 38 = tennis racket in COCO)
RACKET_CLASS_ID_COCO: int = 38
# Custom model class IDs if using a fine-tuned model — extend as needed
RACKET_CLASS_IDS: set = {RACKET_CLASS_ID_COCO}


def log(task: str, status: str, message: str = "", **kwargs: Any) -> None:
    """Emit structured JSON log line."""
    entry: Dict[str, Any] = {
        "agent": AGENT,
        "task": task,
        "status": status,
        "message": message,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    entry.update(kwargs)
    print(json.dumps(entry), flush=True)


# ---------- Data Classes ----------

class GestureState(enum.Enum):
    """States of the gesture state machine."""
    IDLE = "IDLE"
    ARMS_RAISED = "ARMS_RAISED"
    HAND_APPROACHING_RACKET = "HAND_APPROACHING_RACKET"
    CLAP_DETECTED = "CLAP_DETECTED"
    COOLDOWN = "COOLDOWN"


@dataclass
class GestureEvent:
    """Emitted when a palm-to-racket clap gesture is detected."""
    timestamp: float          # seconds from recording start
    player_bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    frame_index: int


# ---------- Utility Functions ----------

def _bbox_center(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
    """Return center point of a bounding box."""
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _point_in_bbox(
    point: Tuple[float, float],
    bbox: Tuple[float, float, float, float],
    margin: float = 0.0,
) -> bool:
    """Check if a point lies within a bounding box (with optional margin expansion)."""
    x, y = point
    x1, y1, x2, y2 = bbox
    return (x1 - margin) <= x <= (x2 + margin) and (y1 - margin) <= y <= (y2 + margin)


def _hand_racket_iou(
    hand_pos: Tuple[float, float],
    racket_bbox: Tuple[float, float, float, float],
    hand_radius: float = 20.0,
) -> float:
    """
    Compute a pseudo-IoU between a hand point (expanded to a small box)
    and a racket bounding box.
    """
    hx, hy = hand_pos
    h_x1 = hx - hand_radius
    h_y1 = hy - hand_radius
    h_x2 = hx + hand_radius
    h_y2 = hy + hand_radius

    r_x1, r_y1, r_x2, r_y2 = racket_bbox

    # Intersection
    ix1 = max(h_x1, r_x1)
    iy1 = max(h_y1, r_y1)
    ix2 = min(h_x2, r_x2)
    iy2 = min(h_y2, r_y2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0

    area_hand = (h_x2 - h_x1) * (h_y2 - h_y1)
    area_racket = (r_x2 - r_x1) * (r_y2 - r_y1)
    union = area_hand + area_racket - inter

    return inter / union if union > 0 else 0.0


def _extract_racket_detections(
    yolo_detections: List[Detection],
) -> List[Tuple[float, float, float, float]]:
    """
    Extract racket bounding boxes from YOLO detections.

    Returns list of (x1, y1, x2, y2) bounding boxes.
    """
    rackets: List[Tuple[float, float, float, float]] = []
    for det in yolo_detections:
        if det.class_id in RACKET_CLASS_IDS:
            x1 = det.x_center - det.width / 2
            y1 = det.y_center - det.height / 2
            x2 = det.x_center + det.width / 2
            y2 = det.y_center + det.height / 2
            rackets.append((x1, y1, x2, y2))
    return rackets


# ---------- GestureDetector ----------

class GestureDetector:
    """
    Detects palm-to-racket clap gestures using pose estimation and YOLO
    racket detections.

    The detection follows a strict sequential state machine to avoid
    false positives:
      IDLE -> ARMS_RAISED -> HAND_APPROACHING_RACKET -> CLAP_DETECTED -> COOLDOWN -> IDLE

    Args:
        fps: Video frame rate (used to compute timestamps).
        cooldown_seconds: Minimum seconds between gesture triggers.
        min_frames: Minimum consecutive frames required in each transient state.
        arm_raised_threshold: Fraction of frame height wrist must be above shoulder.
        overlap_threshold: Pseudo-IoU threshold for hand-racket contact.
        approach_velocity_threshold: Minimum approach speed (px/frame).
        timeout_frames: Max frames in a transient state without progress before reset.
    """

    def __init__(
        self,
        fps: float = 30.0,
        cooldown_seconds: float = GESTURE_COOLDOWN_SECONDS,
        min_frames: int = GESTURE_MIN_FRAMES,
        arm_raised_threshold: float = ARM_RAISED_THRESHOLD,
        overlap_threshold: float = HAND_RACKET_OVERLAP_THRESHOLD,
        approach_velocity_threshold: float = APPROACH_VELOCITY_THRESHOLD,
        timeout_frames: int = TIMEOUT_FRAMES,
    ) -> None:
        self._fps = fps
        self._cooldown_seconds = cooldown_seconds
        self._min_frames = min_frames
        self._arm_raised_threshold = arm_raised_threshold
        self._overlap_threshold = overlap_threshold
        self._approach_velocity_threshold = approach_velocity_threshold
        self._timeout_frames = timeout_frames

        # State
        self._state = GestureState.IDLE
        self._state_frame_count: int = 0
        self._frame_index: int = 0
        self._cooldown_start_frame: int = 0

        # Tracking hand-racket approach
        self._prev_hand_racket_distance: Optional[float] = None
        self._approaching_frames: int = 0

        # Which hand is the "free" hand (not holding the racket)
        self._free_hand_side: Optional[str] = None  # "left" or "right"
        self._racket_hand_side: Optional[str] = None

        # Last pose result for visualization
        self._last_pose: Optional[PoseResult] = None
        self._last_racket_bbox: Optional[Tuple[float, float, float, float]] = None
        self._last_free_hand_pos: Optional[Tuple[float, float]] = None

        # Events emitted
        self._events: List[GestureEvent] = []

        log(TASK, "ok", "GestureDetector initialized",
            cooldown_s=cooldown_seconds, min_frames=min_frames)

    @property
    def state(self) -> GestureState:
        return self._state

    @property
    def events(self) -> List[GestureEvent]:
        return list(self._events)

    @property
    def last_pose(self) -> Optional[PoseResult]:
        return self._last_pose

    @property
    def last_racket_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        return self._last_racket_bbox

    @property
    def last_free_hand_pos(self) -> Optional[Tuple[float, float]]:
        return self._last_free_hand_pos

    def _transition(self, new_state: GestureState, reason: str = "") -> None:
        old = self._state
        self._state = new_state
        self._state_frame_count = 0
        log(TASK, "info", f"Gesture state: {old.value} -> {new_state.value}",
            reason=reason, frame=self._frame_index)

    def _find_closest_racket_to_hand(
        self,
        hand_pos: Tuple[float, float],
        racket_bboxes: List[Tuple[float, float, float, float]],
    ) -> Optional[Tuple[Tuple[float, float, float, float], float]]:
        """Find the racket bbox closest to a hand position. Returns (bbox, distance)."""
        if not racket_bboxes:
            return None

        best_bbox = None
        best_dist = float("inf")
        for bbox in racket_bboxes:
            center = _bbox_center(*bbox)
            d = _distance(hand_pos, center)
            if d < best_dist:
                best_dist = d
                best_bbox = bbox

        if best_bbox is None:
            return None
        return best_bbox, best_dist

    def _determine_hand_roles(
        self,
        pose: PoseResult,
        racket_bboxes: List[Tuple[float, float, float, float]],
    ) -> Optional[Tuple[str, Tuple[float, float], Tuple[float, float, float, float]]]:
        """
        Determine which hand holds the racket and which is the free hand.

        Returns:
            (free_hand_side, free_hand_pos, closest_racket_bbox) or None.
        """
        left_hand, right_hand = pose.get_hand_positions()
        if left_hand is None or right_hand is None:
            return None

        if not racket_bboxes:
            return None

        # Find which hand is closer to a racket
        left_result = self._find_closest_racket_to_hand(left_hand, racket_bboxes)
        right_result = self._find_closest_racket_to_hand(right_hand, racket_bboxes)

        if left_result is None and right_result is None:
            return None

        left_dist = left_result[1] if left_result else float("inf")
        right_dist = right_result[1] if right_result else float("inf")

        if left_dist < right_dist:
            # Left hand holds racket, right hand is free
            assert left_result is not None
            return ("right", right_hand, left_result[0])
        else:
            # Right hand holds racket, left hand is free
            assert right_result is not None
            return ("left", left_hand, right_result[0])

    def update(
        self,
        frame: np.ndarray,
        yolo_detections: List[Detection],
        pose_results: Optional[List[PoseResult]] = None,
        pose_estimator: Optional[PoseEstimator] = None,
    ) -> Optional[GestureEvent]:
        """
        Process a single frame for gesture detection.

        Either provide pre-computed pose_results or a pose_estimator
        to run pose estimation internally.

        Args:
            frame: BGR image (H, W, 3).
            yolo_detections: All YOLO detections for this frame.
            pose_results: Pre-computed pose results (optional).
            pose_estimator: PoseEstimator instance to use if pose_results not given.

        Returns:
            GestureEvent if a clap gesture was detected, None otherwise.
        """
        t0 = time.perf_counter()
        self._frame_index += 1
        self._state_frame_count += 1

        # Get pose
        if pose_results is None and pose_estimator is not None:
            pose_results = pose_estimator.estimate(frame)

        if not pose_results:
            self._last_pose = None
            self._last_racket_bbox = None
            self._last_free_hand_pos = None
            # No pose detected — timeout transient states
            if self._state not in (GestureState.IDLE, GestureState.COOLDOWN):
                if self._state_frame_count > self._timeout_frames:
                    self._transition(GestureState.IDLE, "no_pose_timeout")
            return self._check_cooldown()

        pose = pose_results[0]
        self._last_pose = pose

        # Extract racket bounding boxes from YOLO detections
        racket_bboxes = _extract_racket_detections(yolo_detections)
        self._last_racket_bbox = racket_bboxes[0] if racket_bboxes else None

        # Run the state machine
        event = self._run_state_machine(pose, racket_bboxes)

        ms = (time.perf_counter() - t0) * 1000
        if event is not None:
            log(TASK, "ok", "Gesture detected!",
                ms=round(ms, 1), frame=self._frame_index,
                confidence=round(event.confidence, 3))

        return event

    def _check_cooldown(self) -> Optional[GestureEvent]:
        """Check if cooldown period has elapsed."""
        if self._state == GestureState.COOLDOWN:
            frames_elapsed = self._frame_index - self._cooldown_start_frame
            cooldown_frames = int(self._cooldown_seconds * self._fps)
            if frames_elapsed >= cooldown_frames:
                self._transition(GestureState.IDLE, "cooldown_expired")
        return None

    def _run_state_machine(
        self,
        pose: PoseResult,
        racket_bboxes: List[Tuple[float, float, float, float]],
    ) -> Optional[GestureEvent]:
        """Core state machine logic."""

        # --- COOLDOWN ---
        if self._state == GestureState.COOLDOWN:
            return self._check_cooldown()

        # --- IDLE ---
        if self._state == GestureState.IDLE:
            if pose.is_arms_raised(threshold=self._arm_raised_threshold):
                self._transition(GestureState.ARMS_RAISED, "arms_detected_raised")
            return None

        # --- ARMS_RAISED ---
        if self._state == GestureState.ARMS_RAISED:
            if not pose.is_arms_raised(threshold=self._arm_raised_threshold):
                # Lost the arms-raised posture
                if self._state_frame_count > self._timeout_frames:
                    self._transition(GestureState.IDLE, "arms_lowered_timeout")
                return None

            # Need arms raised for min_frames before progressing
            if self._state_frame_count < self._min_frames:
                return None

            # Check if we can identify hand roles with a racket
            roles = self._determine_hand_roles(pose, racket_bboxes)
            if roles is not None:
                free_side, free_pos, racket_bbox = roles
                self._free_hand_side = free_side
                self._last_free_hand_pos = free_pos
                self._last_racket_bbox = racket_bbox

                # Compute initial distance
                racket_center = _bbox_center(*racket_bbox)
                self._prev_hand_racket_distance = _distance(free_pos, racket_center)
                self._approaching_frames = 0

                self._transition(
                    GestureState.HAND_APPROACHING_RACKET,
                    f"free_hand={free_side}",
                )
            elif self._state_frame_count > self._timeout_frames:
                self._transition(GestureState.IDLE, "no_racket_timeout")

            return None

        # --- HAND_APPROACHING_RACKET ---
        if self._state == GestureState.HAND_APPROACHING_RACKET:
            # Must still have arms raised
            if not pose.is_arms_raised(threshold=self._arm_raised_threshold):
                if self._state_frame_count > self._timeout_frames:
                    self._transition(GestureState.IDLE, "arms_lowered")
                    return None

            # Find hand and racket positions
            roles = self._determine_hand_roles(pose, racket_bboxes)
            if roles is None:
                if self._state_frame_count > self._timeout_frames:
                    self._transition(GestureState.IDLE, "lost_hand_or_racket")
                return None

            free_side, free_pos, racket_bbox = roles
            self._last_free_hand_pos = free_pos
            self._last_racket_bbox = racket_bbox

            racket_center = _bbox_center(*racket_bbox)
            current_dist = _distance(free_pos, racket_center)

            # Check approach velocity
            if self._prev_hand_racket_distance is not None:
                approach_speed = self._prev_hand_racket_distance - current_dist
                if approach_speed > self._approach_velocity_threshold:
                    self._approaching_frames += 1
                else:
                    # Not approaching fast enough — allow some slack
                    self._approaching_frames = max(0, self._approaching_frames - 1)

            self._prev_hand_racket_distance = current_dist

            # Check overlap (clap contact)
            iou = _hand_racket_iou(free_pos, racket_bbox)
            in_bbox = _point_in_bbox(free_pos, racket_bbox, margin=15.0)

            if (iou >= self._overlap_threshold or in_bbox) and self._approaching_frames >= 1:
                self._transition(GestureState.CLAP_DETECTED, "hand_contact_racket")

                # Build the event
                player_bbox = pose.get_bounding_box()
                if player_bbox is None:
                    player_bbox = racket_bbox  # fallback

                event = GestureEvent(
                    timestamp=self._frame_index / self._fps,
                    player_bbox=player_bbox,
                    confidence=self._compute_confidence(pose, iou),
                    frame_index=self._frame_index,
                )
                self._events.append(event)

                # Enter cooldown
                self._cooldown_start_frame = self._frame_index
                self._transition(GestureState.COOLDOWN, "gesture_complete")

                return event

            # Timeout
            if self._state_frame_count > self._timeout_frames:
                self._transition(GestureState.IDLE, "approach_timeout")

            return None

        return None

    def _compute_confidence(self, pose: PoseResult, iou: float) -> float:
        """
        Compute a confidence score for the gesture detection.

        Combines pose keypoint visibility and hand-racket overlap.
        """
        # Average keypoint visibility for relevant joints
        vis_scores: List[float] = []
        for kp_name in ("left_wrist", "right_wrist", "left_shoulder", "right_shoulder"):
            kp = pose.keypoints.get(kp_name)
            if kp is not None:
                vis_scores.append(kp[2])

        avg_visibility = np.mean(vis_scores) if vis_scores else 0.5
        # Blend visibility and overlap
        return float(0.5 * avg_visibility + 0.5 * min(iou / self._overlap_threshold, 1.0))

    def reset(self) -> None:
        """Reset detector to initial state."""
        self._state = GestureState.IDLE
        self._state_frame_count = 0
        self._frame_index = 0
        self._cooldown_start_frame = 0
        self._prev_hand_racket_distance = None
        self._approaching_frames = 0
        self._free_hand_side = None
        self._racket_hand_side = None
        self._last_pose = None
        self._last_racket_bbox = None
        self._last_free_hand_pos = None
        self._events.clear()
        log(TASK, "ok", "GestureDetector reset")
