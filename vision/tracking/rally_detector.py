"""
Beach Tennis Recorder - Rally Detection Logic
AGENT-02 | TASK-02-09, TASK-02-10

Implements the rally state machine:
  IDLE -> EM_JOGO -> FIM_RALLY -> IDLE

Detects:
  - Ball bounce (sudden Y-direction change)
  - Net crossing (ball passes through central zone)
  - Ball out of bounds
  - Timeout (>8s without ball detection)
"""

import enum
import json
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .byte_tracker import Track


AGENT = "02"

# Constants from the state machine contract
VELOCITY_THRESHOLD: float = 15.0       # px/frame to consider ball in motion
CONFIDENCE_THRESHOLD: float = 0.45     # minimum model confidence
RALLY_TIMEOUT_SECONDS: float = 8.0     # max time without detection before ending rally
NET_CROSS_REQUIRED: bool = True        # ball must cross net to confirm rally
BUFFER_PRE_RALLY_SECONDS: float = 3.0  # seconds before serve to capture
BUFFER_POST_RALLY_SECONDS: float = 2.0 # seconds after rally end


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


class RallyState(enum.Enum):
    """States of the rally state machine."""
    IDLE = "IDLE"
    EM_JOGO = "EM_JOGO"
    FIM_RALLY = "FIM_RALLY"


@dataclass
class RallyEvent:
    """Event emitted when a rally ends."""
    rally_number: int
    start_time: float       # epoch timestamp
    end_time: float         # epoch timestamp
    duration_seconds: float
    end_reason: str         # "bounce", "out_of_bounds", "timeout", "manual"
    net_crossings: int
    ball_bounces: int


@dataclass
class CourtConfig:
    """Court boundaries for detection (in pixel coordinates of the model input)."""
    # Net zone: horizontal band where the net is located
    net_y_min: float = 280.0
    net_y_max: float = 360.0

    # Court bounds (approximate for 640x640 input)
    court_x_min: float = 50.0
    court_x_max: float = 590.0
    court_y_min: float = 80.0
    court_y_max: float = 560.0

    # Frame dimensions
    frame_width: float = 640.0
    frame_height: float = 640.0


class BounceDetector:
    """Detects ball bounces by analyzing Y-velocity direction changes."""

    def __init__(self, min_speed: float = 5.0, direction_change_threshold: float = 0.7):
        self._min_speed = min_speed
        self._direction_change_threshold = direction_change_threshold
        self._prev_vy: Optional[float] = None
        self._bounce_count: int = 0

    def update(self, track: Track) -> bool:
        """Check if a bounce occurred based on track velocity."""
        vx, vy = track.velocity

        if track.speed < self._min_speed:
            return False

        is_bounce = False
        if self._prev_vy is not None:
            # Bounce = sudden reversal of Y direction with significant magnitude
            if (self._prev_vy * vy < 0 and
                    abs(vy) > self._direction_change_threshold and
                    abs(self._prev_vy) > self._direction_change_threshold):
                is_bounce = True
                self._bounce_count += 1

        self._prev_vy = vy
        return is_bounce

    @property
    def bounce_count(self) -> int:
        return self._bounce_count

    def reset(self) -> None:
        self._prev_vy = None
        self._bounce_count = 0


class NetCrossingDetector:
    """Detects when the ball crosses the net zone."""

    def __init__(self, court: CourtConfig):
        self._court = court
        self._prev_y: Optional[float] = None
        self._crossing_count: int = 0

    def update(self, track: Track) -> bool:
        """Check if ball crossed the net zone since last frame."""
        pos = track.position
        if pos is None:
            return False

        _, cy = pos
        crossed = False

        if self._prev_y is not None:
            net_mid = (self._court.net_y_min + self._court.net_y_max) / 2
            # Crossed if previous and current Y are on opposite sides of net
            if ((self._prev_y < net_mid and cy > net_mid) or
                    (self._prev_y > net_mid and cy < net_mid)):
                crossed = True
                self._crossing_count += 1

        self._prev_y = cy
        return crossed

    @property
    def crossing_count(self) -> int:
        return self._crossing_count

    def reset(self) -> None:
        self._prev_y = None
        self._crossing_count = 0


class OutOfBoundsDetector:
    """Detects when the ball goes out of court bounds."""

    def __init__(self, court: CourtConfig, margin: float = 20.0):
        self._court = court
        self._margin = margin

    def is_out(self, track: Track) -> bool:
        """Check if ball position is outside court bounds."""
        pos = track.position
        if pos is None:
            return False

        cx, cy = pos
        return (
            cx < self._court.court_x_min - self._margin or
            cx > self._court.court_x_max + self._margin or
            cy < self._court.court_y_min - self._margin or
            cy > self._court.court_y_max + self._margin
        )


class RallyDetector:
    """
    Main rally detection state machine.

    States:
        IDLE       - No rally in progress, waiting for ball motion
        EM_JOGO    - Rally is active, tracking ball
        FIM_RALLY  - Rally just ended, emitting event

    Transitions:
        IDLE -> EM_JOGO:    Ball detected with speed > VELOCITY_THRESHOLD
        EM_JOGO -> FIM_RALLY: Bounce, out-of-bounds, or timeout
        FIM_RALLY -> IDLE:   Automatic after event emission
    """

    def __init__(
        self,
        court: Optional[CourtConfig] = None,
        fps: float = 30.0,
        on_rally_end: Optional[Callable[[RallyEvent], None]] = None,
        velocity_threshold: float = VELOCITY_THRESHOLD,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        rally_timeout_seconds: float = RALLY_TIMEOUT_SECONDS,
        net_cross_required: bool = NET_CROSS_REQUIRED,
    ):
        self._court = court or CourtConfig()
        self._fps = fps
        self._on_rally_end = on_rally_end

        # Configurable thresholds
        self._velocity_threshold = velocity_threshold
        self._confidence_threshold = confidence_threshold
        self._rally_timeout_seconds = rally_timeout_seconds
        self._net_cross_required = net_cross_required

        # State
        self._state = RallyState.IDLE
        self._rally_number: int = 0
        self._rally_start_time: float = 0.0
        self._last_detection_time: float = 0.0

        # Sub-detectors
        self._bounce_detector = BounceDetector()
        self._net_detector = NetCrossingDetector(self._court)
        self._oob_detector = OutOfBoundsDetector(self._court)

        # Rally history
        self._events: List[RallyEvent] = []

    @property
    def state(self) -> RallyState:
        return self._state

    @property
    def rally_number(self) -> int:
        return self._rally_number

    @property
    def events(self) -> List[RallyEvent]:
        return list(self._events)

    @property
    def is_rally_active(self) -> bool:
        return self._state == RallyState.EM_JOGO

    def _transition(self, new_state: RallyState, reason: str = "") -> None:
        """Transition state machine."""
        old = self._state
        self._state = new_state
        log("02-09", "info", f"State: {old.value} -> {new_state.value}",
            reason=reason, rally=self._rally_number)

    def _start_rally(self) -> None:
        """Start a new rally."""
        self._rally_number += 1
        self._rally_start_time = time.time()
        self._bounce_detector.reset()
        self._net_detector.reset()
        self._transition(RallyState.EM_JOGO, "ball_in_motion")

    def _end_rally(self, reason: str) -> RallyEvent:
        """End the current rally and emit an event."""
        now = time.time()
        event = RallyEvent(
            rally_number=self._rally_number,
            start_time=self._rally_start_time,
            end_time=now,
            duration_seconds=round(now - self._rally_start_time, 2),
            end_reason=reason,
            net_crossings=self._net_detector.crossing_count,
            ball_bounces=self._bounce_detector.bounce_count,
        )
        self._events.append(event)

        log("02-09", "ok", "Rally ended", rally=event.rally_number,
            duration_s=event.duration_seconds, reason=reason,
            net_crossings=event.net_crossings, bounces=event.ball_bounces)

        if self._on_rally_end:
            self._on_rally_end(event)

        self._transition(RallyState.IDLE, f"rally_end_{reason}")
        return event

    def update(self, ball_tracks: List[Track]) -> Optional[RallyEvent]:
        """
        Process a frame's ball tracks and update rally state.

        Args:
            ball_tracks: List of active ball Track objects from ByteTracker.

        Returns:
            RallyEvent if a rally just ended, None otherwise.
        """
        now = time.time()

        # Use the highest-confidence ball track
        ball: Optional[Track] = None
        if ball_tracks:
            ball = max(ball_tracks, key=lambda t: (
                t.last_detection.confidence if t.last_detection else 0.0
            ))
            self._last_detection_time = now

        # --- IDLE state ---
        if self._state == RallyState.IDLE:
            if ball and ball.speed > self._velocity_threshold:
                if (ball.last_detection and
                        ball.last_detection.confidence >= self._confidence_threshold):
                    self._start_rally()
            return None

        # --- EM_JOGO state ---
        if self._state == RallyState.EM_JOGO:
            # Check timeout
            elapsed_since_detection = now - self._last_detection_time
            if elapsed_since_detection > self._rally_timeout_seconds:
                return self._end_rally("timeout")

            if ball is None:
                return None

            # Check bounce (ball touched ground)
            if self._bounce_detector.update(ball):
                log("02-09", "info", "Bounce detected",
                    rally=self._rally_number, count=self._bounce_detector.bounce_count)
                # A bounce alone doesn't end rally; it's part of play.
                # Rally ends on second bounce on same side, or out of bounds.
                # For MVP: end on bounce if net has been crossed at least once
                if self._net_detector.crossing_count > 0 or not self._net_cross_required:
                    return self._end_rally("bounce")

            # Check net crossing
            if self._net_detector.update(ball):
                log("02-10", "info", "Net crossing detected",
                    rally=self._rally_number, count=self._net_detector.crossing_count)

            # Check out of bounds
            if self._oob_detector.is_out(ball):
                return self._end_rally("out_of_bounds")

            return None

        # FIM_RALLY auto-transitions to IDLE (handled in _end_rally)
        return None

    def reset(self) -> None:
        """Reset detector to initial state."""
        self._state = RallyState.IDLE
        self._rally_number = 0
        self._bounce_detector.reset()
        self._net_detector.reset()
        self._events.clear()
        log("02-09", "info", "Rally detector reset")


if __name__ == "__main__":
    from .byte_tracker import ByteTracker, Detection

    # Simulate a simple rally
    tracker = ByteTracker()
    detector = RallyDetector(fps=30.0)

    print("Simulating a rally sequence...")

    # Ball moving fast (serve) -> crosses net -> bounces -> rally ends
    frames = [
        # Serve: ball moving fast on one side
        (100, 200, 0.9),
        (120, 195, 0.88),
        (140, 190, 0.85),
        # Ball crosses net zone (y ~ 320)
        (160, 250, 0.82),
        (180, 300, 0.80),
        (200, 350, 0.78),
        # Ball on other side, moving down (toward ground)
        (220, 400, 0.82),
        (240, 440, 0.85),
        (250, 470, 0.83),
        # Bounce: Y direction reverses
        (255, 450, 0.80),
        (260, 420, 0.78),
    ]

    for i, (x, y, conf) in enumerate(frames):
        dets = [Detection(float(x), float(y), 15.0, 15.0, conf, class_id=0)]
        active = tracker.update(dets)
        ball_tracks = [t for t in active if t.class_id == 0]
        event = detector.update(ball_tracks)

        if event:
            print(f"\nRally #{event.rally_number} ended: {event.end_reason}")
            print(f"  Duration: {event.duration_seconds}s")
            print(f"  Net crossings: {event.net_crossings}")
            print(f"  Bounces: {event.ball_bounces}")
