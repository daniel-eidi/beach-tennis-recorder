"""
Shared test fixtures for the integration test suite.

AGENT-05 | TASK-05-01, TASK-05-02, TASK-05-03

Provides:
  - Synthetic test video creation via ffmpeg
  - Temp directory management for test artifacts
  - Helpers to generate fake detection sequences (simulating model output)
  - Helpers to generate realistic rally sequences
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Make the project root importable
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vision.tracking.byte_tracker import ByteTracker, Detection, Track
from vision.tracking.rally_detector import (
    CourtConfig,
    RallyDetector,
    RallyEvent,
    RallyState,
)
from tests.mock_data.scenarios import (
    DetectionFrame,
    Scenario,
    get_scenario,
    list_scenarios,
)

# ---------------------------------------------------------------------------
# ffmpeg availability
# ---------------------------------------------------------------------------
FFMPEG_AVAILABLE: bool = shutil.which("ffmpeg") is not None

requires_ffmpeg = pytest.mark.skipif(
    not FFMPEG_AVAILABLE,
    reason="ffmpeg binary not found on PATH",
)


# ---------------------------------------------------------------------------
# Temp directory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def session_tmp_dir() -> str:
    """Session-scoped temp directory.  Cleaned up after all tests."""
    tmpdir = tempfile.mkdtemp(prefix="bt_integration_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def tmp_dir() -> str:
    """Per-test temp directory.  Cleaned up after each test."""
    tmpdir = tempfile.mkdtemp(prefix="bt_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Synthetic video creation
# ---------------------------------------------------------------------------

def create_test_video(
    path: str,
    duration: float = 5.0,
    fps: int = 30,
    width: int = 320,
    height: int = 240,
    color: str = "blue",
    with_audio: bool = True,
) -> str:
    """Create a minimal test video using the ffmpeg CLI.

    Returns the path to the created video.
    """
    inputs = [
        "-f", "lavfi", "-i",
        f"color=c={color}:size={width}x{height}:rate={fps}:d={duration}",
    ]
    if with_audio:
        inputs += ["-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono"]

    cmd = [
        "ffmpeg", "-y",
        *inputs,
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
    ]
    if with_audio:
        cmd += ["-c:a", "aac", "-b:a", "32k"]
    cmd += ["-pix_fmt", "yuv420p", path]

    subprocess.run(cmd, check=True, capture_output=True)
    return path


@pytest.fixture(scope="session")
def short_video(session_tmp_dir: str) -> str:
    """A 5-second test video (320x240, 30fps)."""
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg not available")
    path = os.path.join(session_tmp_dir, "short_5s.mp4")
    return create_test_video(path, duration=5.0)


@pytest.fixture(scope="session")
def long_video(session_tmp_dir: str) -> str:
    """A 20-second test video (320x240, 30fps)."""
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg not available")
    path = os.path.join(session_tmp_dir, "long_20s.mp4")
    return create_test_video(path, duration=20.0)


@pytest.fixture(scope="session")
def tiny_video(session_tmp_dir: str) -> str:
    """A 0.5-second video that should fail min-duration validation."""
    if not FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg not available")
    path = os.path.join(session_tmp_dir, "tiny_0.5s.mp4")
    return create_test_video(path, duration=0.5)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def make_detection(
    x: float,
    y: float,
    confidence: float = 0.85,
    class_id: int = 0,
    width: float = 15.0,
    height: float = 15.0,
) -> Detection:
    """Create a Detection object."""
    return Detection(
        x_center=x,
        y_center=y,
        width=width,
        height=height,
        confidence=confidence,
        class_id=class_id,
    )


def make_ball_detection(x: float, y: float, confidence: float = 0.85) -> Detection:
    """Create a ball detection (class_id=0)."""
    return make_detection(x, y, confidence, class_id=0)


def make_net_detection(
    x: float = 320.0,
    confidence: float = 0.90,
) -> Detection:
    """Create a net detection (class_id=1)."""
    return Detection(
        x_center=x,
        y_center=320.0,
        width=540.0,
        height=80.0,
        confidence=confidence,
        class_id=1,
    )


def make_court_detection(confidence: float = 0.88) -> Detection:
    """Create a court line detection (class_id=2)."""
    return Detection(
        x_center=320.0,
        y_center=320.0,
        width=540.0,
        height=480.0,
        confidence=confidence,
        class_id=2,
    )


def scenario_detections_to_objects(
    scenario: Scenario,
) -> List[List[Detection]]:
    """Convert a Scenario's detection dicts into Detection objects per frame."""
    result: List[List[Detection]] = []
    for frame in scenario.frames:
        dets = [
            Detection(
                x_center=d["x_center"],
                y_center=d["y_center"],
                width=d["width"],
                height=d["height"],
                confidence=d["confidence"],
                class_id=d["class_id"],
            )
            for d in frame.detections
        ]
        result.append(dets)
    return result


def run_scenario_through_pipeline(
    scenario: Scenario,
    net_cross_required: bool = True,
    velocity_threshold: float = 15.0,
    simulate_timeout: bool = False,
    timeout_after_frame: Optional[int] = None,
    timeout_seconds: float = 9.0,
) -> Tuple[List[RallyEvent], RallyDetector, ByteTracker]:
    """Run a scenario through ByteTracker + RallyDetector.

    Args:
        scenario: Test scenario to run.
        net_cross_required: NET_CROSS_REQUIRED flag.
        velocity_threshold: Velocity threshold for rally start.
        simulate_timeout: If True, patch time for timeout scenarios.
        timeout_after_frame: Frame index after which to simulate elapsed time.
        timeout_seconds: Seconds to advance when simulating timeout.

    Returns:
        (events, rally_detector, tracker) tuple.
    """
    import unittest.mock as mock

    # Use max_frames_lost=0 when simulating timeout so the tracker
    # immediately deactivates the track when detections stop.
    # This prevents the tracker from feeding stale ball tracks to the
    # rally detector, which would keep resetting _last_detection_time.
    if simulate_timeout:
        tracker = ByteTracker(max_frames_lost=0)
    else:
        tracker = ByteTracker()
    events: List[RallyEvent] = []

    def on_rally_end(event: RallyEvent) -> None:
        events.append(event)

    detector = RallyDetector(
        fps=scenario.fps,
        on_rally_end=on_rally_end,
        velocity_threshold=velocity_threshold,
        net_cross_required=net_cross_required,
    )

    detection_frames = scenario_detections_to_objects(scenario)

    # If simulating timeout, we need to control time.time()
    if simulate_timeout:
        base_time = 1000000.0
        # Use a mutable container so the nested function can update it
        time_state = {"current": base_time}

        def fake_time() -> float:
            return time_state["current"]

        with mock.patch("vision.tracking.rally_detector.time") as mock_time:
            mock_time.time = fake_time
            mock_time.strftime = lambda fmt: "2026-03-25T00:00:00"

            for i, frame_dets in enumerate(detection_frames):
                if timeout_after_frame is not None and i > timeout_after_frame:
                    # Jump time: last detection was at timeout_after_frame,
                    # then time advances from the jump point forward so
                    # that the gap from the last detection exceeds the timeout.
                    frames_after_cutoff = i - timeout_after_frame
                    time_state["current"] = (
                        base_time
                        + timeout_after_frame * (1.0 / scenario.fps)
                        + timeout_seconds
                        + frames_after_cutoff * (1.0 / scenario.fps)
                    )
                else:
                    # Normal time advancement
                    time_state["current"] = base_time + i * (1.0 / scenario.fps)

                active_tracks = tracker.update(frame_dets)
                ball_tracks = [t for t in active_tracks if t.class_id == 0]
                detector.update(ball_tracks)
    else:
        for frame_dets in detection_frames:
            active_tracks = tracker.update(frame_dets)
            ball_tracks = [t for t in active_tracks if t.class_id == 0]
            detector.update(ball_tracks)

    return events, detector, tracker


# ---------------------------------------------------------------------------
# Scenario fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def normal_rally_scenario() -> Scenario:
    return get_scenario("normal_rally")


@pytest.fixture
def serve_ace_scenario() -> Scenario:
    return get_scenario("serve_ace")


@pytest.fixture
def warmup_scenario() -> Scenario:
    return get_scenario("warmup_noise")


@pytest.fixture
def timeout_scenario() -> Scenario:
    return get_scenario("timeout_rally")


@pytest.fixture
def multiple_rallies_scenario() -> Scenario:
    return get_scenario("multiple_rallies")


@pytest.fixture
def long_rally_scenario() -> Scenario:
    return get_scenario("long_rally")


@pytest.fixture
def out_of_bounds_scenario() -> Scenario:
    return get_scenario("out_of_bounds")


@pytest.fixture
def noisy_scenario() -> Scenario:
    return get_scenario("noisy_detections")


@pytest.fixture
def fresh_tracker() -> ByteTracker:
    """A fresh ByteTracker instance."""
    return ByteTracker()


@pytest.fixture
def fresh_detector() -> RallyDetector:
    """A fresh RallyDetector with default settings."""
    return RallyDetector(fps=30.0)
