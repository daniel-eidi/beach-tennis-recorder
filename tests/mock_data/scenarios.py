"""
Predefined detection sequences for integration testing.

AGENT-05 | TASK-05-01

Provides realistic detection sequences that simulate various beach tennis
rally scenarios.  Each scenario function returns a list of per-frame
detections (as dicts) plus the expected rally events the pipeline should
produce when consuming them.

Detection format matches vision.tracking.byte_tracker.Detection fields:
    x_center, y_center, width, height, confidence, class_id

Class IDs:
    0 = ball
    1 = net
    2 = court_line
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


BALL: int = 0
NET: int = 1
COURT_LINE: int = 2

# Court geometry (matches CourtConfig defaults in rally_detector.py)
NET_Y_MIN: float = 280.0
NET_Y_MAX: float = 360.0
NET_Y_MID: float = (NET_Y_MIN + NET_Y_MAX) / 2  # 320.0

COURT_X_MIN: float = 50.0
COURT_X_MAX: float = 590.0
COURT_Y_MIN: float = 80.0
COURT_Y_MAX: float = 560.0


@dataclass
class DetectionFrame:
    """Detections for a single frame."""
    frame_index: int
    detections: List[Dict[str, Any]]


@dataclass
class ExpectedRallyEvent:
    """Expected rally event for validation."""
    rally_number: int
    end_reason: str
    min_net_crossings: int = 0
    min_bounces: int = 0


@dataclass
class Scenario:
    """A complete test scenario with frame detections and expected results."""
    name: str
    description: str
    frames: List[DetectionFrame]
    expected_rallies: List[ExpectedRallyEvent]
    fps: float = 30.0

    @property
    def total_frames(self) -> int:
        return len(self.frames)


def _make_detection(
    x: float,
    y: float,
    confidence: float = 0.85,
    class_id: int = BALL,
    width: float = 15.0,
    height: float = 15.0,
) -> Dict[str, Any]:
    """Create a detection dict."""
    return {
        "x_center": x,
        "y_center": y,
        "width": width,
        "height": height,
        "confidence": confidence,
        "class_id": class_id,
    }


def _make_net_detection(
    x: float = 320.0,
    confidence: float = 0.90,
) -> Dict[str, Any]:
    """Create a net detection at the standard location."""
    return _make_detection(
        x=x, y=NET_Y_MID,
        confidence=confidence,
        class_id=NET,
        width=540.0,
        height=NET_Y_MAX - NET_Y_MIN,
    )


def _make_court_detection(confidence: float = 0.88) -> Dict[str, Any]:
    """Create a court line detection."""
    cx = (COURT_X_MIN + COURT_X_MAX) / 2
    cy = (COURT_Y_MIN + COURT_Y_MAX) / 2
    return _make_detection(
        x=cx, y=cy,
        confidence=confidence,
        class_id=COURT_LINE,
        width=COURT_X_MAX - COURT_X_MIN,
        height=COURT_Y_MAX - COURT_Y_MIN,
    )


def _linear_path(
    start: Tuple[float, float],
    end: Tuple[float, float],
    num_frames: int,
    confidence: float = 0.85,
) -> List[Dict[str, Any]]:
    """Generate ball detections along a linear path."""
    detections: List[Dict[str, Any]] = []
    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        x = start[0] + (end[0] - start[0]) * t
        y = start[1] + (end[1] - start[1]) * t
        detections.append(_make_detection(x, y, confidence))
    return detections


# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------

def normal_rally() -> Scenario:
    """Standard rally with 4 net crossings, ends with bounce.

    Ball path:
    1. Serve from bottom (y~500) moving up fast
    2. Crosses net (y crosses 320)
    3. Return: crosses net back
    4. Another volley across
    5. Ball bounces (Y reversal) after crossing net

    The RallyDetector requires net_cross >= 1 + bounce to end rally.
    """
    frames: List[DetectionFrame] = []
    frame_idx = 0

    # Phase 1: Serve from bottom side moving fast upward (high velocity)
    serve_positions = [
        (200.0, 500.0), (210.0, 475.0), (220.0, 450.0),
        (230.0, 420.0), (240.0, 385.0), (250.0, 350.0),
    ]
    for x, y in serve_positions:
        frames.append(DetectionFrame(
            frame_index=frame_idx,
            detections=[_make_detection(x, y, 0.88), _make_net_detection()],
        ))
        frame_idx += 1

    # Phase 2: Ball crosses net and continues to other side
    cross_net_positions = [
        (260.0, 310.0), (270.0, 280.0), (280.0, 250.0),
        (290.0, 220.0), (300.0, 200.0),
    ]
    for x, y in cross_net_positions:
        frames.append(DetectionFrame(
            frame_index=frame_idx,
            detections=[_make_detection(x, y, 0.85), _make_net_detection()],
        ))
        frame_idx += 1

    # Phase 3: Return - ball comes back across net
    return_positions = [
        (310.0, 220.0), (320.0, 250.0), (330.0, 280.0),
        (340.0, 310.0), (350.0, 340.0), (360.0, 370.0),
        (370.0, 400.0),
    ]
    for x, y in return_positions:
        frames.append(DetectionFrame(
            frame_index=frame_idx,
            detections=[_make_detection(x, y, 0.83), _make_net_detection()],
        ))
        frame_idx += 1

    # Phase 4: Another volley - crosses net again
    volley_positions = [
        (380.0, 380.0), (390.0, 350.0), (400.0, 320.0),
        (410.0, 290.0), (420.0, 260.0), (430.0, 230.0),
    ]
    for x, y in volley_positions:
        frames.append(DetectionFrame(
            frame_index=frame_idx,
            detections=[_make_detection(x, y, 0.82), _make_net_detection()],
        ))
        frame_idx += 1

    # Phase 5: Ball comes back across net
    return2_positions = [
        (440.0, 260.0), (445.0, 290.0), (450.0, 320.0),
        (455.0, 350.0), (460.0, 380.0), (465.0, 410.0),
    ]
    for x, y in return2_positions:
        frames.append(DetectionFrame(
            frame_index=frame_idx,
            detections=[_make_detection(x, y, 0.80), _make_net_detection()],
        ))
        frame_idx += 1

    # Phase 6: Bounce - sudden Y reversal (ball going down then up)
    bounce_positions = [
        (470.0, 440.0), (475.0, 460.0), (478.0, 475.0),
        # Y reversal here (bounce)
        (480.0, 455.0), (482.0, 430.0),
    ]
    for x, y in bounce_positions:
        frames.append(DetectionFrame(
            frame_index=frame_idx,
            detections=[_make_detection(x, y, 0.78), _make_net_detection()],
        ))
        frame_idx += 1

    return Scenario(
        name="normal_rally",
        description="Standard rally with 4 net crossings ending with bounce",
        frames=frames,
        expected_rallies=[
            ExpectedRallyEvent(
                rally_number=1,
                end_reason="bounce",
                min_net_crossings=1,
                min_bounces=1,
            ),
        ],
    )


def serve_ace() -> Scenario:
    """Fast serve with no return -- ball goes out of bounds.

    Ball moves fast from one side, crosses net, continues out of bounds.
    """
    frames: List[DetectionFrame] = []
    frame_idx = 0

    # Fast serve: large steps per frame (high velocity)
    positions = [
        (100.0, 500.0), (130.0, 460.0), (160.0, 420.0),
        (190.0, 370.0), (220.0, 320.0),  # crosses net
        (250.0, 270.0), (280.0, 220.0), (310.0, 160.0),
        (340.0, 100.0),
        # Ball goes out of bounds (y < court_y_min - margin)
        (370.0, 50.0),
    ]
    for x, y in positions:
        frames.append(DetectionFrame(
            frame_index=frame_idx,
            detections=[_make_detection(x, y, 0.90)],
        ))
        frame_idx += 1

    return Scenario(
        name="serve_ace",
        description="Fast serve ace, ball goes out of bounds",
        frames=frames,
        expected_rallies=[
            ExpectedRallyEvent(
                rally_number=1,
                end_reason="out_of_bounds",
                min_net_crossings=1,
            ),
        ],
    )


def out_of_bounds() -> Scenario:
    """Rally ends with ball going wide (x exceeds court bounds).

    Ball starts moving fast, crosses net, then veers off the side.
    """
    frames: List[DetectionFrame] = []
    frame_idx = 0

    # Serve
    serve = [
        (200.0, 480.0), (220.0, 440.0), (240.0, 400.0),
        (260.0, 360.0), (280.0, 320.0),  # crosses net
        (300.0, 280.0), (320.0, 250.0),
    ]
    for x, y in serve:
        frames.append(DetectionFrame(
            frame_index=frame_idx,
            detections=[_make_detection(x, y, 0.87)],
        ))
        frame_idx += 1

    # Return veers wide
    wide_positions = [
        (350.0, 270.0), (390.0, 290.0), (440.0, 310.0),
        (500.0, 320.0), (560.0, 330.0),
        # Out of bounds (x > court_x_max + margin = 590 + 20 = 610)
        (620.0, 340.0),
    ]
    for x, y in wide_positions:
        frames.append(DetectionFrame(
            frame_index=frame_idx,
            detections=[_make_detection(x, y, 0.82)],
        ))
        frame_idx += 1

    return Scenario(
        name="out_of_bounds",
        description="Rally ends with ball going wide of the court",
        frames=frames,
        expected_rallies=[
            ExpectedRallyEvent(
                rally_number=1,
                end_reason="out_of_bounds",
                min_net_crossings=1,
            ),
        ],
    )


def long_rally() -> Scenario:
    """Extended rally with 20+ net crossings, ends with bounce.

    Generates a sinusoidal ball trajectory that crosses the net repeatedly.
    """
    frames: List[DetectionFrame] = []
    num_frames = 120
    net_crossings_target = 22

    # Ball oscillates around the net line with sinusoidal Y
    for i in range(num_frames):
        t = i / num_frames
        # X moves slowly across the court
        x = 150.0 + t * 300.0
        # Y oscillates across net line (period chosen for ~22 crossings)
        y = NET_Y_MID + 120.0 * math.sin(2 * math.pi * net_crossings_target / 2 * t)
        conf = 0.85 - 0.1 * abs(math.sin(t * math.pi))
        frames.append(DetectionFrame(
            frame_index=i,
            detections=[_make_detection(x, y, max(conf, 0.50))],
        ))

    # Add bounce at the end (Y direction reversal)
    last_y = frames[-1].detections[0]["y_center"]
    for j in range(5):
        idx = num_frames + j
        x = 460.0 + j * 3.0
        # Going down then reversing
        y = last_y + (15.0 if j < 3 else -20.0)
        last_y = y
        frames.append(DetectionFrame(
            frame_index=idx,
            detections=[_make_detection(x, y, 0.78)],
        ))

    return Scenario(
        name="long_rally",
        description="Extended rally with 20+ net crossings ending with bounce",
        frames=frames,
        expected_rallies=[
            ExpectedRallyEvent(
                rally_number=1,
                end_reason="bounce",
                min_net_crossings=5,
                min_bounces=1,
            ),
        ],
    )


def timeout_rally() -> Scenario:
    """Ball disappears mid-rally (occlusion), timeout triggers end.

    Ball moves fast, crosses net, then disappears for >8s worth of frames.
    The timeout is checked against wall-clock time in the real detector,
    so this scenario provides enough empty frames for the test to inject
    the necessary time delay.
    """
    frames: List[DetectionFrame] = []
    frame_idx = 0

    # Fast ball movement (triggers EM_JOGO)
    moving = [
        (200.0, 480.0), (220.0, 440.0), (240.0, 400.0),
        (260.0, 360.0), (280.0, 320.0),  # crosses net
        (300.0, 280.0), (320.0, 250.0),
    ]
    for x, y in moving:
        frames.append(DetectionFrame(
            frame_index=frame_idx,
            detections=[_make_detection(x, y, 0.88)],
        ))
        frame_idx += 1

    # Ball disappears - empty detections for many frames
    # (In the actual test, we simulate wall-clock timeout by patching time.)
    for _ in range(60):
        frames.append(DetectionFrame(
            frame_index=frame_idx,
            detections=[],
        ))
        frame_idx += 1

    return Scenario(
        name="timeout_rally",
        description="Ball disappears, rally ends via timeout",
        frames=frames,
        expected_rallies=[
            ExpectedRallyEvent(
                rally_number=1,
                end_reason="timeout",
            ),
        ],
    )


def warmup_noise() -> Scenario:
    """Random slow movement that should NOT trigger a rally.

    Ball moves at low velocity (below VELOCITY_THRESHOLD of 15 px/frame),
    simulating casual warmup tosses.
    """
    frames: List[DetectionFrame] = []

    # Slow random-ish movement: < 15 px/frame displacement
    positions = [
        (300.0, 400.0), (302.0, 398.0), (304.0, 396.0),
        (306.0, 397.0), (305.0, 399.0), (303.0, 401.0),
        (301.0, 400.0), (300.0, 402.0), (299.0, 401.0),
        (300.0, 400.0), (301.0, 399.0), (302.0, 400.0),
        (303.0, 401.0), (302.0, 402.0), (301.0, 401.0),
        (300.0, 400.0), (299.0, 399.0), (298.0, 400.0),
        (299.0, 401.0), (300.0, 400.0),
    ]
    for i, (x, y) in enumerate(positions):
        frames.append(DetectionFrame(
            frame_index=i,
            detections=[_make_detection(x, y, 0.75)],
        ))

    return Scenario(
        name="warmup_noise",
        description="Slow warmup movement that should not trigger a rally",
        frames=frames,
        expected_rallies=[],  # No rallies expected
    )


def multiple_rallies() -> Scenario:
    """Two consecutive rallies separated by an idle period.

    Rally 1: Normal serve + net cross + bounce
    Idle: 20 frames of no detections
    Rally 2: Another serve + out of bounds
    """
    frames: List[DetectionFrame] = []
    frame_idx = 0

    # --- Rally 1: Serve + cross net + bounce ---
    rally1_positions = [
        (150.0, 500.0), (170.0, 460.0), (190.0, 420.0),
        (210.0, 380.0), (230.0, 340.0), (250.0, 300.0),  # crosses net
        (270.0, 260.0), (290.0, 230.0),
        # Return across net
        (300.0, 250.0), (310.0, 280.0), (320.0, 310.0),
        (330.0, 340.0), (340.0, 370.0), (350.0, 400.0),
        # Bounce (Y reversal)
        (360.0, 430.0), (365.0, 450.0), (368.0, 460.0),
        (370.0, 440.0), (372.0, 415.0),
    ]
    for x, y in rally1_positions:
        frames.append(DetectionFrame(
            frame_index=frame_idx,
            detections=[_make_detection(x, y, 0.86)],
        ))
        frame_idx += 1

    # Idle period (no detections)
    for _ in range(20):
        frames.append(DetectionFrame(
            frame_index=frame_idx,
            detections=[],
        ))
        frame_idx += 1

    # --- Rally 2: Serve + cross net clearly + out of bounds ---
    rally2_positions = [
        (400.0, 500.0), (420.0, 460.0), (440.0, 420.0),
        (460.0, 380.0), (470.0, 350.0),
        # Cross net zone clearly (y goes from >320 to <320)
        (480.0, 300.0), (490.0, 270.0), (500.0, 240.0),
        # Goes wide
        (540.0, 240.0), (580.0, 240.0),
        # Out of bounds (x > 610)
        (620.0, 240.0),
    ]
    for x, y in rally2_positions:
        frames.append(DetectionFrame(
            frame_index=frame_idx,
            detections=[_make_detection(x, y, 0.84)],
        ))
        frame_idx += 1

    return Scenario(
        name="multiple_rallies",
        description="Two consecutive rallies: bounce then out-of-bounds",
        frames=frames,
        expected_rallies=[
            ExpectedRallyEvent(
                rally_number=1,
                end_reason="bounce",
                min_net_crossings=1,
                min_bounces=1,
            ),
            ExpectedRallyEvent(
                rally_number=2,
                end_reason="out_of_bounds",
            ),
        ],
    )


def noisy_detections() -> Scenario:
    """Ball detections with jitter, confidence drops, and missed frames.

    Tests tracker resilience.  Ball still crosses net and bounces.
    """
    frames: List[DetectionFrame] = []
    frame_idx = 0

    # Noisy serve with some missed frames and jitter
    base_positions = [
        (200.0, 490.0), (220.0, 455.0), None,  # missed frame
        (260.0, 385.0), (280.0, 350.0), (300.0, 315.0),
        None,  # missed frame
        (335.0, 250.0), (350.0, 220.0),
        # Return
        (360.0, 240.0), None, (380.0, 290.0),
        (390.0, 320.0), (400.0, 350.0), (410.0, 380.0),
        (415.0, 400.0),
        # Bounce
        (420.0, 430.0), (425.0, 450.0), (428.0, 460.0),
        (430.0, 440.0), (432.0, 415.0),
    ]

    import random
    random.seed(42)

    for pos in base_positions:
        if pos is None:
            frames.append(DetectionFrame(
                frame_index=frame_idx,
                detections=[],
            ))
        else:
            x, y = pos
            # Add jitter
            jitter_x = random.uniform(-5, 5)
            jitter_y = random.uniform(-5, 5)
            conf = random.uniform(0.55, 0.92)
            frames.append(DetectionFrame(
                frame_index=frame_idx,
                detections=[_make_detection(x + jitter_x, y + jitter_y, conf)],
            ))
        frame_idx += 1

    return Scenario(
        name="noisy_detections",
        description="Ball detections with jitter and missed frames",
        frames=frames,
        expected_rallies=[
            ExpectedRallyEvent(
                rally_number=1,
                end_reason="bounce",
                min_net_crossings=1,
            ),
        ],
    )


ALL_SCENARIOS: Dict[str, Scenario] = {
    "normal_rally": normal_rally(),
    "serve_ace": serve_ace(),
    "out_of_bounds": out_of_bounds(),
    "long_rally": long_rally(),
    "timeout_rally": timeout_rally(),
    "warmup_noise": warmup_noise(),
    "multiple_rallies": multiple_rallies(),
    "noisy_detections": noisy_detections(),
}


def get_scenario(name: str) -> Scenario:
    """Retrieve a scenario by name."""
    if name not in ALL_SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{name}'. Available: {list(ALL_SCENARIOS.keys())}"
        )
    return ALL_SCENARIOS[name]


def list_scenarios() -> List[str]:
    """Return available scenario names."""
    return list(ALL_SCENARIOS.keys())
