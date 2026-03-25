"""
Synthetic beach tennis video generator for integration testing.

AGENT-05 | TASK-05-01

Generates MP4 videos with:
  - A moving white circle on green background (simulating ball)
  - A horizontal white line in the middle (simulating net)
  - Rectangle outline (court boundaries)

Also produces a ground-truth JSON with expected detections per frame.

Usage:
    python tests/mock_data/generate_mock_video.py --scenario normal --duration 10
    python tests/mock_data/generate_mock_video.py --scenario all --output-dir tests/mock_data/videos/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:
    print("numpy is required: pip install numpy", file=sys.stderr)
    sys.exit(1)


# Dimensions matching model input expectations
FRAME_WIDTH: int = 640
FRAME_HEIGHT: int = 640
FPS: int = 30
BALL_RADIUS: int = 8
BG_COLOR: Tuple[int, int, int] = (34, 139, 34)       # forest green (BGR)
BALL_COLOR: Tuple[int, int, int] = (255, 255, 255)     # white
NET_COLOR: Tuple[int, int, int] = (200, 200, 200)      # light gray
COURT_COLOR: Tuple[int, int, int] = (255, 255, 255)    # white

# Court geometry
NET_Y: int = 320
COURT_X_MIN: int = 50
COURT_X_MAX: int = 590
COURT_Y_MIN: int = 80
COURT_Y_MAX: int = 560


def _draw_court(frame: "np.ndarray") -> None:
    """Draw net and court boundaries on a frame."""
    try:
        import cv2
    except ImportError:
        # Fallback: draw manually with numpy slicing
        # Net: horizontal line
        frame[NET_Y - 2:NET_Y + 2, COURT_X_MIN:COURT_X_MAX] = NET_COLOR
        # Court rectangle
        frame[COURT_Y_MIN:COURT_Y_MIN + 2, COURT_X_MIN:COURT_X_MAX] = COURT_COLOR
        frame[COURT_Y_MAX - 2:COURT_Y_MAX, COURT_X_MIN:COURT_X_MAX] = COURT_COLOR
        frame[COURT_Y_MIN:COURT_Y_MAX, COURT_X_MIN:COURT_X_MIN + 2] = COURT_COLOR
        frame[COURT_Y_MIN:COURT_Y_MAX, COURT_X_MAX - 2:COURT_X_MAX] = COURT_COLOR
        return

    # Net
    cv2.line(frame, (COURT_X_MIN, NET_Y), (COURT_X_MAX, NET_Y), NET_COLOR, 3)
    # Court rectangle
    cv2.rectangle(frame, (COURT_X_MIN, COURT_Y_MIN),
                  (COURT_X_MAX, COURT_Y_MAX), COURT_COLOR, 2)


def _draw_ball(frame: "np.ndarray", x: int, y: int) -> None:
    """Draw a ball circle on a frame."""
    try:
        import cv2
        cv2.circle(frame, (x, y), BALL_RADIUS, BALL_COLOR, -1)
    except ImportError:
        # Manual circle approximation
        for dy in range(-BALL_RADIUS, BALL_RADIUS + 1):
            for dx in range(-BALL_RADIUS, BALL_RADIUS + 1):
                if dx * dx + dy * dy <= BALL_RADIUS * BALL_RADIUS:
                    py, px = y + dy, x + dx
                    if 0 <= py < FRAME_HEIGHT and 0 <= px < FRAME_WIDTH:
                        frame[py, px] = BALL_COLOR


def _generate_ball_trajectory(
    scenario: str,
    duration: float,
) -> List[Optional[Tuple[int, int]]]:
    """Generate per-frame (x, y) ball positions for a scenario.

    Returns None for frames where the ball is not visible.
    """
    total_frames = int(duration * FPS)
    positions: List[Optional[Tuple[int, int]]] = []

    if scenario == "normal":
        # Ball oscillates across net with serve from bottom
        for i in range(total_frames):
            t = i / total_frames
            x = int(150 + 300 * t)
            y = int(NET_Y + 200 * math.sin(2 * math.pi * 2 * t))
            y = max(COURT_Y_MIN + 10, min(COURT_Y_MAX - 10, y))
            positions.append((x, y))

    elif scenario == "serve_ace":
        # Ball moves fast from bottom to out of bounds
        serve_frames = min(total_frames, FPS * 2)
        for i in range(serve_frames):
            t = i / serve_frames
            x = int(200 + 200 * t)
            y = int(500 - 500 * t)
            positions.append((max(0, min(x, FRAME_WIDTH - 1)),
                              max(0, min(y, FRAME_HEIGHT - 1))))
        # Rest of video: no ball
        for _ in range(total_frames - serve_frames):
            positions.append(None)

    elif scenario == "out_of_bounds":
        # Ball goes wide
        phase1 = total_frames // 2
        for i in range(phase1):
            t = i / phase1
            x = int(200 + 200 * t)
            y = int(450 - 200 * t)
            positions.append((x, y))
        for i in range(total_frames - phase1):
            t = i / max(1, total_frames - phase1)
            x = int(400 + 300 * t)
            y = int(250 + 50 * t)
            positions.append((min(x, FRAME_WIDTH - 1), y))

    elif scenario == "long_rally":
        # Many net crossings
        for i in range(total_frames):
            t = i / total_frames
            x = int(150 + 300 * t)
            y = int(NET_Y + 150 * math.sin(2 * math.pi * 12 * t))
            y = max(COURT_Y_MIN + 10, min(COURT_Y_MAX - 10, y))
            positions.append((x, y))

    elif scenario == "warmup":
        # Slow random-like movement
        cx, cy = 320, 400
        for i in range(total_frames):
            t = i / total_frames
            x = int(cx + 15 * math.sin(2 * math.pi * 0.5 * t))
            y = int(cy + 10 * math.cos(2 * math.pi * 0.3 * t))
            positions.append((x, y))

    else:
        # Default: static ball
        for _ in range(total_frames):
            positions.append((320, 400))

    return positions


def generate_video(
    scenario: str,
    duration: float,
    output_path: str,
    ground_truth_path: Optional[str] = None,
) -> str:
    """Generate a synthetic test video with ground truth.

    Args:
        scenario: Scenario name.
        duration: Video duration in seconds.
        output_path: Path for the output MP4.
        ground_truth_path: Optional path for ground-truth JSON.

    Returns:
        Path to the generated video.
    """
    positions = _generate_ball_trajectory(scenario, duration)
    total_frames = len(positions)

    ground_truth: List[Dict[str, Any]] = []

    # Write raw frames to a pipe and encode with ffmpeg
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    for frame_idx, pos in enumerate(positions):
        frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), BG_COLOR, dtype=np.uint8)
        _draw_court(frame)

        frame_gt: Dict[str, Any] = {
            "frame_index": frame_idx,
            "timestamp_s": round(frame_idx / FPS, 4),
            "detections": [],
        }

        # Add net detection to ground truth
        frame_gt["detections"].append({
            "class_id": 1,
            "class_name": "net",
            "x_center": float((COURT_X_MIN + COURT_X_MAX) / 2),
            "y_center": float(NET_Y),
            "width": float(COURT_X_MAX - COURT_X_MIN),
            "height": 4.0,
            "confidence": 1.0,
        })

        if pos is not None:
            bx, by = pos
            _draw_ball(frame, bx, by)
            frame_gt["detections"].append({
                "class_id": 0,
                "class_name": "ball",
                "x_center": float(bx),
                "y_center": float(by),
                "width": float(BALL_RADIUS * 2),
                "height": float(BALL_RADIUS * 2),
                "confidence": 1.0,
            })

        ground_truth.append(frame_gt)

        assert proc.stdin is not None
        proc.stdin.write(frame.tobytes())

    assert proc.stdin is not None
    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode() if proc.stderr else ""
        raise RuntimeError(f"ffmpeg failed (rc={proc.returncode}): {stderr}")

    # Save ground truth
    gt_path = ground_truth_path or output_path.replace(".mp4", "_ground_truth.json")
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump({
            "scenario": scenario,
            "duration_s": duration,
            "fps": FPS,
            "total_frames": total_frames,
            "frame_width": FRAME_WIDTH,
            "frame_height": FRAME_HEIGHT,
            "frames": ground_truth,
        }, f, indent=2)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic beach tennis videos for testing"
    )
    parser.add_argument(
        "--scenario", type=str, default="normal",
        choices=["normal", "serve_ace", "out_of_bounds", "long_rally", "warmup", "all"],
        help="Scenario to generate (default: normal)",
    )
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Video duration in seconds (default: 10)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="tests/mock_data/videos",
        help="Output directory (default: tests/mock_data/videos/)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    scenarios = (
        ["normal", "serve_ace", "out_of_bounds", "long_rally", "warmup"]
        if args.scenario == "all"
        else [args.scenario]
    )

    for sc in scenarios:
        out_path = os.path.join(args.output_dir, f"mock_{sc}.mp4")
        print(f"Generating: {sc} ({args.duration}s) -> {out_path}")
        generate_video(sc, args.duration, out_path)
        print(f"  Done: {out_path}")

    print("\nAll mock videos generated.")


if __name__ == "__main__":
    main()
