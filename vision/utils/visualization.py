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
