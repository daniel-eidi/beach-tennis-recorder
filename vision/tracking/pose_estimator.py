"""
Beach Tennis Recorder - MediaPipe Pose Wrapper
AGENT-02 | Gesture Detection Feature

Wraps MediaPipe Pose for body keypoint estimation.
Used by GestureDetector to identify arm-raised + racket-clap gestures.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None  # type: ignore[assignment]


AGENT = "02"
TASK = "gesture-pose"


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


# MediaPipe landmark indices we care about
LANDMARK_NAMES: Dict[int, str] = {
    0: "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    19: "left_index",
    20: "right_index",
}

# Fraction above shoulder Y that wrist must be to count as "raised"
ARM_RAISED_THRESHOLD: float = 0.15


@dataclass
class PoseResult:
    """Result from a single person's pose estimation."""

    keypoints: Dict[str, Tuple[float, float, float]]
    """Mapping of landmark name -> (x_px, y_px, confidence)."""

    frame_width: int = 0
    frame_height: int = 0

    @property
    def left_shoulder(self) -> Optional[Tuple[float, float, float]]:
        return self.keypoints.get("left_shoulder")

    @property
    def right_shoulder(self) -> Optional[Tuple[float, float, float]]:
        return self.keypoints.get("right_shoulder")

    @property
    def left_wrist(self) -> Optional[Tuple[float, float, float]]:
        return self.keypoints.get("left_wrist")

    @property
    def right_wrist(self) -> Optional[Tuple[float, float, float]]:
        return self.keypoints.get("right_wrist")

    @property
    def left_elbow(self) -> Optional[Tuple[float, float, float]]:
        return self.keypoints.get("left_elbow")

    @property
    def right_elbow(self) -> Optional[Tuple[float, float, float]]:
        return self.keypoints.get("right_elbow")

    @property
    def left_index(self) -> Optional[Tuple[float, float, float]]:
        return self.keypoints.get("left_index")

    @property
    def right_index(self) -> Optional[Tuple[float, float, float]]:
        return self.keypoints.get("right_index")

    def is_arms_raised(self, threshold: float = ARM_RAISED_THRESHOLD) -> bool:
        """
        Check whether both wrists are above their respective shoulders.

        In image coordinates, "above" means smaller Y value.
        The threshold is expressed as a fraction of frame height:
        wrist_y must be at least (threshold * frame_height) pixels above shoulder_y.
        """
        lw = self.left_wrist
        rw = self.right_wrist
        ls = self.left_shoulder
        rs = self.right_shoulder

        if not all([lw, rw, ls, rs]):
            return False

        assert lw is not None and rw is not None
        assert ls is not None and rs is not None

        min_gap = threshold * self.frame_height if self.frame_height > 0 else 10.0

        left_raised = (ls[1] - lw[1]) > min_gap
        right_raised = (rs[1] - rw[1]) > min_gap

        return left_raised and right_raised

    def get_hand_positions(
        self,
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """
        Return (left_hand_pos, right_hand_pos) in pixel coordinates.

        Uses index finger tip if available, falls back to wrist.
        """
        left: Optional[Tuple[float, float]] = None
        right: Optional[Tuple[float, float]] = None

        li = self.left_index
        lw = self.left_wrist
        if li and li[2] > 0.3:
            left = (li[0], li[1])
        elif lw and lw[2] > 0.3:
            left = (lw[0], lw[1])

        ri = self.right_index
        rw = self.right_wrist
        if ri and ri[2] > 0.3:
            right = (ri[0], ri[1])
        elif rw and rw[2] > 0.3:
            right = (rw[0], rw[1])

        return left, right

    def get_bounding_box(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Compute a rough bounding box (x1, y1, x2, y2) from all visible keypoints.
        """
        xs = []
        ys = []
        for _, (x, y, c) in self.keypoints.items():
            if c > 0.3:
                xs.append(x)
                ys.append(y)

        if len(xs) < 3:
            return None

        margin_x = (max(xs) - min(xs)) * 0.2
        margin_y = (max(ys) - min(ys)) * 0.2
        return (
            min(xs) - margin_x,
            min(ys) - margin_y,
            max(xs) + margin_x,
            max(ys) + margin_y,
        )


class PoseEstimator:
    """
    Wraps MediaPipe Pose to estimate body keypoints from a BGR frame.

    Lightweight and suitable for CPU inference alongside YOLO.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ) -> None:
        if mp is None:
            raise ImportError(
                "mediapipe is required for pose estimation. "
                "Install with: pip install mediapipe"
            )

        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        log(TASK, "ok", "PoseEstimator initialized",
            det_conf=min_detection_confidence,
            track_conf=min_tracking_confidence)

    def estimate(self, frame: np.ndarray) -> List[PoseResult]:
        """
        Run pose estimation on a BGR frame.

        Args:
            frame: BGR image (H, W, 3) numpy array.

        Returns:
            List of PoseResult (currently MediaPipe supports single-person,
            so the list has 0 or 1 elements).
        """
        h, w = frame.shape[:2]

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        results = self._pose.process(rgb)

        if results.pose_landmarks is None:
            return []

        # Convert normalized landmarks to pixel coordinates
        keypoints: Dict[str, Tuple[float, float, float]] = {}
        for idx, name in LANDMARK_NAMES.items():
            lm = results.pose_landmarks.landmark[idx]
            keypoints[name] = (
                lm.x * w,
                lm.y * h,
                lm.visibility,
            )

        pose_result = PoseResult(
            keypoints=keypoints,
            frame_width=w,
            frame_height=h,
        )
        return [pose_result]

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._pose.close()
        log(TASK, "ok", "PoseEstimator closed")
