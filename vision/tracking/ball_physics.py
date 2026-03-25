"""
Beach Tennis Recorder - Ball Physics Analysis
AGENT-02 | Sprint 2

Estimates ball speed, detects serves, lobs, smashes, and classifies
shots based on trajectory patterns from ByteTracker tracks.

These features enhance rally detection accuracy by providing context
about the type of play occurring.
"""

import json
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .byte_tracker import Track

AGENT = "02"
TASK = "ball-physics"


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


class ShotType(Enum):
    """Recognized shot types in beach tennis."""
    UNKNOWN = "unknown"
    SERVE = "serve"
    LOB = "lob"
    SMASH = "smash"
    DRIVE = "drive"
    DROP = "drop"


@dataclass
class CourtCalibration:
    """
    Maps pixel coordinates to real-world meters.

    Default values assume a standard beach tennis court (16m x 8m)
    viewed from a roughly centered camera filling a 640x640 frame.
    """
    court_length_m: float = 16.0
    court_width_m: float = 8.0
    frame_court_x_min: float = 50.0
    frame_court_x_max: float = 590.0
    frame_court_y_min: float = 80.0
    frame_court_y_max: float = 560.0

    @property
    def px_per_meter_x(self) -> float:
        """Horizontal pixels per meter."""
        return (self.frame_court_x_max - self.frame_court_x_min) / self.court_width_m

    @property
    def px_per_meter_y(self) -> float:
        """Vertical pixels per meter."""
        return (self.frame_court_y_max - self.frame_court_y_min) / self.court_length_m


class BallPhysicsAnalyzer:
    """
    Analyzes ball trajectory from tracking data to estimate physical
    properties and classify shot types.
    """

    def __init__(
        self,
        calibration: Optional[CourtCalibration] = None,
        fps: float = 30.0,
    ):
        self._calibration = calibration or CourtCalibration()
        self._fps = fps

    def estimate_speed_px_per_frame(self, track: Track) -> float:
        """
        Estimate ball speed in pixels/frame from the track's velocity.

        Args:
            track: A ball Track from ByteTracker.

        Returns:
            Speed magnitude in px/frame.
        """
        return track.speed

    def estimate_speed_ms(self, track: Track) -> float:
        """
        Estimate ball speed in meters/second using court calibration.

        Args:
            track: A ball Track from ByteTracker.

        Returns:
            Estimated speed in m/s. Returns 0.0 if calibration is unavailable.
        """
        vx, vy = track.velocity
        # Convert px/frame to m/s
        vx_mps = (vx / self._calibration.px_per_meter_x) * self._fps
        vy_mps = (vy / self._calibration.px_per_meter_y) * self._fps
        return float(math.sqrt(vx_mps ** 2 + vy_mps ** 2))

    def estimate_speed_kmh(self, track: Track) -> float:
        """Estimate ball speed in km/h."""
        return self.estimate_speed_ms(track) * 3.6

    def get_trajectory_direction(self, track: Track) -> Optional[float]:
        """
        Get trajectory direction angle in degrees.

        0 = moving right, 90 = moving down, -90 = moving up.

        Returns:
            Angle in degrees, or None if track has no velocity.
        """
        vx, vy = track.velocity
        if abs(vx) < 0.01 and abs(vy) < 0.01:
            return None
        return float(math.degrees(math.atan2(vy, vx)))

    def is_moving_up(self, track: Track) -> bool:
        """Check if the ball is moving upward (negative Y in image coords)."""
        _, vy = track.velocity
        return vy < -1.0

    def is_moving_down(self, track: Track) -> bool:
        """Check if the ball is moving downward (positive Y in image coords)."""
        _, vy = track.velocity
        return vy > 1.0

    def detect_serve(
        self,
        track: Track,
        min_speed_px: float = 20.0,
        min_trajectory_len: int = 3,
    ) -> bool:
        """
        Detect if the current track motion looks like a serve.

        A serve is characterized by:
        - Ball moving fast from one side of the court
        - Relatively horizontal trajectory initially
        - Track starts near one end of the court

        Args:
            track: Ball track.
            min_speed_px: Minimum speed in px/frame to be considered a serve.
            min_trajectory_len: Minimum trajectory history required.

        Returns:
            True if the motion pattern matches a serve.
        """
        if len(track.trajectory) < min_trajectory_len:
            return False

        if track.speed < min_speed_px:
            return False

        # Serve: mostly horizontal component (ball moving across court)
        vx, vy = track.velocity
        if abs(vx) < 0.01:
            return False

        horizontal_ratio = abs(vx) / (abs(vx) + abs(vy)) if (abs(vx) + abs(vy)) > 0 else 0
        # Serve has strong horizontal component
        if horizontal_ratio < 0.4:
            return False

        # Check if trajectory starts near one end of the court
        first_pos = track.trajectory[0]
        _, first_y = first_pos
        cal = self._calibration
        court_third = (cal.frame_court_y_max - cal.frame_court_y_min) / 3

        # Ball should start in the top or bottom third (near baseline)
        near_baseline = (
            first_y < cal.frame_court_y_min + court_third or
            first_y > cal.frame_court_y_max - court_third
        )

        return near_baseline

    def detect_lob(
        self,
        track: Track,
        min_upward_speed: float = 8.0,
        min_trajectory_len: int = 4,
    ) -> bool:
        """
        Detect a lob shot (high Y trajectory - ball going up).

        A lob is characterized by:
        - Significant upward (negative Y) velocity
        - Moderate horizontal component
        - Ball position in mid-court area

        Args:
            track: Ball track.
            min_upward_speed: Minimum upward speed (negative vy) in px/frame.
            min_trajectory_len: Minimum trajectory length.

        Returns:
            True if trajectory matches a lob pattern.
        """
        if len(track.trajectory) < min_trajectory_len:
            return False

        _, vy = track.velocity
        # Lob: ball moving upward (negative Y in image coords) with
        # significant magnitude
        if vy > -min_upward_speed:
            return False

        # Should have at least some horizontal movement
        vx, _ = track.velocity
        if abs(vx) < 2.0:
            return False

        # Lob typically shows a parabolic trajectory
        # Check if recent Y positions form an arc (decreasing Y = going up)
        recent = track.trajectory[-min_trajectory_len:]
        y_values = [p[1] for p in recent]

        # Y should be generally decreasing (ball going up)
        decreasing_count = sum(
            1 for i in range(1, len(y_values)) if y_values[i] < y_values[i - 1]
        )
        return decreasing_count >= len(y_values) // 2

    def detect_smash(
        self,
        track: Track,
        min_downward_speed: float = 15.0,
        min_total_speed: float = 25.0,
        min_trajectory_len: int = 3,
    ) -> bool:
        """
        Detect a smash shot (fast downward trajectory).

        A smash is characterized by:
        - High speed overall
        - Strong downward (positive Y) velocity
        - Ball in upper portion of frame (hit from above)

        Args:
            track: Ball track.
            min_downward_speed: Minimum downward speed in px/frame.
            min_total_speed: Minimum total speed in px/frame.
            min_trajectory_len: Minimum trajectory length.

        Returns:
            True if trajectory matches a smash pattern.
        """
        if len(track.trajectory) < min_trajectory_len:
            return False

        if track.speed < min_total_speed:
            return False

        _, vy = track.velocity
        # Smash: fast downward motion
        if vy < min_downward_speed:
            return False

        # Ball should be in upper half of court / frame
        pos = track.position
        if pos is None:
            return False

        _, cy = pos
        cal = self._calibration
        court_mid_y = (cal.frame_court_y_min + cal.frame_court_y_max) / 2
        return cy < court_mid_y

    def classify_shot(self, track: Track) -> ShotType:
        """
        Classify the current ball motion into a shot type.

        Priority order: smash > serve > lob > drive > unknown.

        Args:
            track: Ball track with recent trajectory.

        Returns:
            Detected ShotType.
        """
        if len(track.trajectory) < 3:
            return ShotType.UNKNOWN

        if self.detect_smash(track):
            return ShotType.SMASH

        if self.detect_serve(track):
            return ShotType.SERVE

        if self.detect_lob(track):
            return ShotType.LOB

        # Drive: fast horizontal ball with moderate vertical component
        if track.speed > 10.0:
            vx, vy = track.velocity
            if abs(vx) > abs(vy):
                return ShotType.DRIVE

        # Drop: slow ball falling (small speed, positive vy)
        if track.speed < 8.0 and track.speed > 2.0:
            _, vy = track.velocity
            if vy > 1.0:
                return ShotType.DROP

        return ShotType.UNKNOWN

    def analyze_track(self, track: Track) -> Dict[str, Any]:
        """
        Full analysis of a ball track.

        Returns a dict with speed, direction, shot classification, and
        trajectory statistics.
        """
        result: Dict[str, Any] = {
            "track_id": track.track_id,
            "trajectory_length": len(track.trajectory),
            "speed_px_per_frame": round(self.estimate_speed_px_per_frame(track), 2),
            "speed_ms": round(self.estimate_speed_ms(track), 2),
            "speed_kmh": round(self.estimate_speed_kmh(track), 1),
            "direction_degrees": None,
            "moving_up": self.is_moving_up(track),
            "moving_down": self.is_moving_down(track),
            "shot_type": self.classify_shot(track).value,
            "velocity": {
                "vx": round(track.velocity[0], 2),
                "vy": round(track.velocity[1], 2),
            },
        }

        direction = self.get_trajectory_direction(track)
        if direction is not None:
            result["direction_degrees"] = round(direction, 1)

        # Trajectory bounding box
        if track.trajectory:
            xs = [p[0] for p in track.trajectory]
            ys = [p[1] for p in track.trajectory]
            result["trajectory_bounds"] = {
                "x_min": round(min(xs), 1),
                "x_max": round(max(xs), 1),
                "y_min": round(min(ys), 1),
                "y_max": round(max(ys), 1),
            }

        return result
