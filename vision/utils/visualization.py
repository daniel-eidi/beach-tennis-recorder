"""
Beach Tennis Recorder - Debug Visualization
AGENT-02 | TASK-02-14

Draw bounding boxes, tracking IDs, trajectories, and rally state overlays
on frames for debugging and validation.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..tracking.byte_tracker import Detection, Track
from ..tracking.gesture_detector import GestureDetector, GestureState
from ..tracking.pose_estimator import PoseResult
from ..tracking.rally_detector import RallyDetector, RallyState


AGENT = "02"
TASK = "02-14"

# Color palette (BGR format for OpenCV)
COLORS: Dict[str, Tuple[int, int, int]] = {
    "ball": (0, 255, 0),        # Green
    "net": (255, 128, 0),       # Blue-ish
    "court_line": (0, 200, 255),# Yellow-ish
    "trajectory": (255, 0, 255),# Magenta
    "text_bg": (0, 0, 0),       # Black
    "idle": (128, 128, 128),    # Gray
    "em_jogo": (0, 255, 0),     # Green
    "fim_rally": (0, 0, 255),   # Red
    # Gesture visualization colors
    "gesture_idle": (128, 128, 128),           # Gray
    "gesture_arms_raised": (0, 255, 255),      # Yellow
    "gesture_approaching": (0, 165, 255),      # Orange
    "gesture_clap": (0, 0, 255),               # Red
    "gesture_cooldown": (255, 0, 255),         # Magenta
    "skeleton": (0, 255, 128),                 # Light green
    "wrist": (0, 255, 255),                    # Yellow
    "shoulder": (255, 128, 0),                 # Blue-ish
    "highlight_flash": (0, 255, 255),          # Yellow
}

CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: COLORS["ball"],
    1: COLORS["net"],
    2: COLORS["court_line"],
}

CLASS_NAMES: Dict[int, str] = {
    0: "ball",
    1: "net",
    2: "court_line",
}


def log(task: str, status: str, message: str = "", **kwargs) -> None:
    """Emit structured JSON log line."""
    entry = {
        "agent": AGENT,
        "task": task,
        "status": status,
        "message": message,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    entry.update(kwargs)
    print(json.dumps(entry), flush=True)


def draw_detection(
    frame: np.ndarray,
    detection: Detection,
    label: str = "",
    thickness: int = 2,
) -> np.ndarray:
    """Draw a single detection bounding box on the frame."""
    color = CLASS_COLORS.get(detection.class_id, (255, 255, 255))

    x1 = int(detection.x_center - detection.width / 2)
    y1 = int(detection.y_center - detection.height / 2)
    x2 = int(detection.x_center + detection.width / 2)
    y2 = int(detection.y_center + detection.height / 2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    if not label:
        cls_name = CLASS_NAMES.get(detection.class_id, "?")
        label = f"{cls_name} {detection.confidence:.2f}"

    # Label background
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    return frame


def draw_track(
    frame: np.ndarray,
    track: Track,
    show_trajectory: bool = True,
    trajectory_length: int = 30,
    thickness: int = 2,
) -> np.ndarray:
    """Draw a tracked object with ID and optional trajectory."""
    if track.last_detection is None:
        return frame

    det = track.last_detection
    color = CLASS_COLORS.get(track.class_id, (255, 255, 255))

    # Draw bounding box
    label = f"ID:{track.track_id} {CLASS_NAMES.get(track.class_id, '?')} {det.confidence:.2f}"
    draw_detection(frame, det, label=label, thickness=thickness)

    # Draw velocity vector
    if track.speed > 1.0:
        cx, cy = int(det.x_center), int(det.y_center)
        vx, vy = track.velocity
        scale = 3.0
        end_x = int(cx + vx * scale)
        end_y = int(cy + vy * scale)
        cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), color, 2, tipLength=0.3)

    # Draw trajectory
    if show_trajectory:
        points = track.trajectory[-trajectory_length:]
        if len(points) > 1:
            pts = np.array([(int(x), int(y)) for x, y in points], dtype=np.int32)
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                t_color = tuple(int(c * alpha) for c in COLORS["trajectory"])
                cv2.line(frame, tuple(pts[i - 1]), tuple(pts[i]), t_color, 2, cv2.LINE_AA)

    return frame


def draw_rally_state(
    frame: np.ndarray,
    rally_detector: RallyDetector,
    position: Tuple[int, int] = (10, 30),
) -> np.ndarray:
    """Draw rally state overlay on frame."""
    state = rally_detector.state
    rally_num = rally_detector.rally_number

    # State color
    state_colors = {
        RallyState.IDLE: COLORS["idle"],
        RallyState.EM_JOGO: COLORS["em_jogo"],
        RallyState.FIM_RALLY: COLORS["fim_rally"],
    }
    color = state_colors.get(state, (255, 255, 255))

    # State indicator circle
    x, y = position
    cv2.circle(frame, (x + 8, y - 5), 8, color, -1)
    cv2.circle(frame, (x + 8, y - 5), 8, (255, 255, 255), 1)

    # State text
    state_text = f"  {state.value} | Rally #{rally_num}"
    cv2.putText(frame, state_text, (x + 20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def draw_court_overlay(
    frame: np.ndarray,
    net_y_min: float = 280.0,
    net_y_max: float = 360.0,
    court_bounds: Optional[Tuple[float, float, float, float]] = None,
    alpha: float = 0.2,
) -> np.ndarray:
    """Draw semi-transparent court zone overlays."""
    overlay = frame.copy()

    # Net zone
    h, w = frame.shape[:2]
    cv2.rectangle(overlay, (0, int(net_y_min)), (w, int(net_y_max)),
                  COLORS["net"], -1)

    # Court bounds
    if court_bounds:
        x_min, y_min, x_max, y_max = court_bounds
        cv2.rectangle(overlay, (int(x_min), int(y_min)), (int(x_max), int(y_max)),
                      COLORS["court_line"], 2)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_info_panel(
    frame: np.ndarray,
    info: Dict[str, str],
    position: Tuple[int, int] = (10, 60),
    line_height: int = 25,
) -> np.ndarray:
    """Draw an info panel with key-value pairs."""
    x, y = position
    for key, value in info.items():
        text = f"{key}: {value}"
        # Background
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x - 2, y - th - 4), (x + tw + 4, y + 4),
                      COLORS["text_bg"], -1)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += line_height

    return frame


# Gesture state to color mapping
GESTURE_STATE_COLORS: Dict[GestureState, Tuple[int, int, int]] = {
    GestureState.IDLE: COLORS["gesture_idle"],
    GestureState.ARMS_RAISED: COLORS["gesture_arms_raised"],
    GestureState.HAND_APPROACHING_RACKET: COLORS["gesture_approaching"],
    GestureState.CLAP_DETECTED: COLORS["gesture_clap"],
    GestureState.COOLDOWN: COLORS["gesture_cooldown"],
}

# Skeleton connections: pairs of keypoint names to draw lines between
SKELETON_CONNECTIONS: List[Tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
]


def draw_skeleton(
    frame: np.ndarray,
    pose: PoseResult,
    color: Tuple[int, int, int] = (0, 255, 128),
    thickness: int = 2,
) -> np.ndarray:
    """Draw skeleton overlay from pose keypoints."""
    # Draw connections
    for kp_a, kp_b in SKELETON_CONNECTIONS:
        a = pose.keypoints.get(kp_a)
        b = pose.keypoints.get(kp_b)
        if a is not None and b is not None and a[2] > 0.3 and b[2] > 0.3:
            pt_a = (int(a[0]), int(a[1]))
            pt_b = (int(b[0]), int(b[1]))
            cv2.line(frame, pt_a, pt_b, color, thickness, cv2.LINE_AA)

    # Highlight wrists
    for name in ("left_wrist", "right_wrist"):
        kp = pose.keypoints.get(name)
        if kp is not None and kp[2] > 0.3:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 8, COLORS["wrist"], -1)
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 8, (0, 0, 0), 1)

    # Highlight shoulders
    for name in ("left_shoulder", "right_shoulder"):
        kp = pose.keypoints.get(name)
        if kp is not None and kp[2] > 0.3:
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 6, COLORS["shoulder"], -1)
            cv2.circle(frame, (int(kp[0]), int(kp[1])), 6, (0, 0, 0), 1)

    # Draw all other keypoints as small dots
    for name, (x, y, conf) in pose.keypoints.items():
        if conf > 0.3 and name not in ("left_wrist", "right_wrist",
                                        "left_shoulder", "right_shoulder"):
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)

    return frame


def draw_gesture_state(
    frame: np.ndarray,
    gesture_detector: GestureDetector,
    pose_result: Optional[PoseResult] = None,
    position: Tuple[int, int] = (10, 30),
    flash_duration_frames: int = 30,
) -> np.ndarray:
    """
    Draw gesture state visualization overlay on frame.

    Includes:
      - Skeleton overlay from pose keypoints
      - Highlighted wrists and shoulders
      - Gesture state indicator (color-coded)
      - Line between free hand and racket when approaching
      - "HIGHLIGHT SAVED!" flash when gesture detected

    Args:
        frame: BGR image to annotate.
        gesture_detector: GestureDetector instance.
        pose_result: Optional PoseResult (uses detector's last_pose if None).
        position: Position for the state indicator text.
        flash_duration_frames: How many frames to show the flash after gesture.

    Returns:
        Annotated frame.
    """
    state = gesture_detector.state
    color = GESTURE_STATE_COLORS.get(state, (255, 255, 255))

    pose = pose_result or gesture_detector.last_pose

    # Draw skeleton if pose is available
    if pose is not None:
        draw_skeleton(frame, pose, color=COLORS["skeleton"])

    # Draw gesture state indicator
    x, y = position
    # State circle
    cv2.circle(frame, (x + 8, y - 5), 10, color, -1)
    cv2.circle(frame, (x + 8, y - 5), 10, (255, 255, 255), 1)

    state_label = f"  Gesture: {state.value}"
    cv2.putText(frame, state_label, (x + 22, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw line between free hand and racket when approaching
    if state == GestureState.HAND_APPROACHING_RACKET:
        free_hand = gesture_detector.last_free_hand_pos
        racket_bbox = gesture_detector.last_racket_bbox
        if free_hand is not None and racket_bbox is not None:
            racket_cx = int((racket_bbox[0] + racket_bbox[2]) / 2)
            racket_cy = int((racket_bbox[1] + racket_bbox[3]) / 2)
            hand_pt = (int(free_hand[0]), int(free_hand[1]))
            racket_pt = (racket_cx, racket_cy)
            # Animated dashed line (orange)
            cv2.line(frame, hand_pt, racket_pt, COLORS["gesture_approaching"], 2, cv2.LINE_AA)
            # Arrow tip toward racket
            cv2.arrowedLine(frame, hand_pt, racket_pt,
                            COLORS["gesture_approaching"], 2, tipLength=0.15)

    # Draw racket bbox highlight in relevant states
    racket_bbox = gesture_detector.last_racket_bbox
    if racket_bbox is not None and state in (
        GestureState.ARMS_RAISED,
        GestureState.HAND_APPROACHING_RACKET,
        GestureState.CLAP_DETECTED,
    ):
        rx1, ry1, rx2, ry2 = racket_bbox
        cv2.rectangle(frame, (int(rx1), int(ry1)), (int(rx2), int(ry2)),
                      color, 2)
        cv2.putText(frame, "RACKET", (int(rx1), int(ry1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # "HIGHLIGHT SAVED!" flash
    if gesture_detector.events:
        last_event = gesture_detector.events[-1]
        frames_since = gesture_detector._frame_index - last_event.frame_index
        if 0 <= frames_since < flash_duration_frames:
            # Pulsing effect: alpha fades out
            alpha = 1.0 - (frames_since / flash_duration_frames)
            h, w = frame.shape[:2]
            overlay = frame.copy()
            # Yellow banner
            banner_y = h // 2 - 30
            cv2.rectangle(overlay, (0, banner_y), (w, banner_y + 60),
                          COLORS["highlight_flash"], -1)
            cv2.addWeighted(overlay, alpha * 0.6, frame, 1 - alpha * 0.6, 0, frame)

            text = "HIGHLIGHT SAVED!"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            tx = (w - tw) // 2
            ty = banner_y + 40
            text_color = (0, 0, int(255 * alpha))
            cv2.putText(frame, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3, cv2.LINE_AA)

    return frame


def annotate_frame(
    frame: np.ndarray,
    tracks: List[Track],
    rally_detector: Optional[RallyDetector] = None,
    show_trajectories: bool = True,
    show_court: bool = True,
    fps: Optional[float] = None,
    inference_ms: Optional[float] = None,
) -> np.ndarray:
    """
    Full debug annotation pipeline for a single frame.

    Args:
        frame: BGR image (numpy array).
        tracks: Active tracks from ByteTracker.
        rally_detector: Optional rally detector for state overlay.
        show_trajectories: Whether to draw ball trajectories.
        show_court: Whether to draw court zone overlays.
        fps: Current FPS to display.
        inference_ms: Inference time to display.

    Returns:
        Annotated frame.
    """
    annotated = frame.copy()

    # Court overlay
    if show_court:
        annotated = draw_court_overlay(annotated)

    # Draw all tracks
    for track in tracks:
        annotated = draw_track(
            annotated, track,
            show_trajectory=show_trajectories and track.class_id == 0,
        )

    # Rally state
    if rally_detector:
        annotated = draw_rally_state(annotated, rally_detector)

    # Info panel
    info: Dict[str, str] = {
        "Tracks": str(len(tracks)),
        "Balls": str(len([t for t in tracks if t.class_id == 0])),
    }
    if fps is not None:
        info["FPS"] = f"{fps:.1f}"
    if inference_ms is not None:
        info["Inference"] = f"{inference_ms:.1f}ms"

    annotated = draw_info_panel(annotated, info, position=(10, 70))

    return annotated


if __name__ == "__main__":
    # Generate a demo visualization frame
    demo = np.zeros((640, 640, 3), dtype=np.uint8) + 40

    # Simulated detections
    dets = [
        Detection(320, 200, 15, 15, 0.92, 0),   # ball
        Detection(320, 320, 400, 40, 0.95, 1),   # net
        Detection(100, 500, 500, 10, 0.88, 2),   # court_line
    ]

    for det in dets:
        demo = draw_detection(demo, det)

    demo = draw_court_overlay(demo)

    info = {"State": "IDLE", "Rally": "0", "FPS": "30.0", "Inference": "23.4ms"}
    demo = draw_info_panel(demo, info)

    out_path = str(Path(__file__).resolve().parent.parent / "debug_visualization.jpg")
    cv2.imwrite(out_path, demo)
    log(TASK, "ok", f"Demo visualization saved to {out_path}")
