"""
Beach Tennis Recorder - ByteTrack Implementation
AGENT-02 | TASK-02-08

Simple ByteTrack for ball tracking between frames.
Handles track ID assignment, velocity calculation, occlusion, and re-identification.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


AGENT = "02"
TASK = "02-08"


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


@dataclass
class Detection:
    """A single detection from the model."""
    x_center: float
    y_center: float
    width: float
    height: float
    confidence: float
    class_id: int

    @property
    def tlbr(self) -> Tuple[float, float, float, float]:
        """Return (top, left, bottom, right) bounding box."""
        x1 = self.x_center - self.width / 2
        y1 = self.y_center - self.height / 2
        x2 = self.x_center + self.width / 2
        y2 = self.y_center + self.height / 2
        return (y1, x1, y2, x2)

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x_center, self.y_center)


@dataclass
class Track:
    """A tracked object with history."""
    track_id: int
    class_id: int
    detections: List[Detection] = field(default_factory=list)
    frames_since_update: int = 0
    is_active: bool = True

    # Kalman-like state (simplified: position + velocity)
    _vx: float = 0.0
    _vy: float = 0.0

    @property
    def last_detection(self) -> Optional[Detection]:
        return self.detections[-1] if self.detections else None

    @property
    def position(self) -> Optional[Tuple[float, float]]:
        if self.last_detection:
            return self.last_detection.center
        return None

    @property
    def velocity(self) -> Tuple[float, float]:
        """Velocity in px/frame."""
        return (self._vx, self._vy)

    @property
    def speed(self) -> float:
        """Speed magnitude in px/frame."""
        return float(np.sqrt(self._vx ** 2 + self._vy ** 2))

    @property
    def trajectory(self) -> List[Tuple[float, float]]:
        """Return list of (x, y) center positions."""
        return [d.center for d in self.detections]

    def predict(self) -> Tuple[float, float]:
        """Predict next position based on velocity."""
        if self.last_detection is None:
            return (0.0, 0.0)
        return (
            self.last_detection.x_center + self._vx,
            self.last_detection.y_center + self._vy,
        )

    def update(self, detection: Detection) -> None:
        """Update track with a new detection."""
        if self.last_detection is not None:
            dt_x = detection.x_center - self.last_detection.x_center
            dt_y = detection.y_center - self.last_detection.y_center
            # Exponential moving average for velocity
            alpha = 0.6
            self._vx = alpha * dt_x + (1 - alpha) * self._vx
            self._vy = alpha * dt_y + (1 - alpha) * self._vy

        self.detections.append(detection)
        self.frames_since_update = 0
        self.is_active = True

    def mark_missed(self) -> None:
        """Mark frame where track was not matched to a detection."""
        self.frames_since_update += 1


class ByteTracker:
    """
    Simplified ByteTrack for beach tennis ball tracking.

    Two-stage association:
    1. Match high-confidence detections to existing tracks (IoU + distance)
    2. Match remaining low-confidence detections to unmatched tracks
    """

    def __init__(
        self,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        max_frames_lost: int = 30,
        match_distance_threshold: float = 100.0,
    ):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.max_frames_lost = max_frames_lost
        self.match_distance_threshold = match_distance_threshold

        self._tracks: Dict[int, Track] = {}
        self._next_id: int = 1
        self._frame_count: int = 0

    @property
    def active_tracks(self) -> List[Track]:
        """Return currently active tracks."""
        return [t for t in self._tracks.values() if t.is_active]

    @property
    def all_tracks(self) -> Dict[int, Track]:
        return dict(self._tracks)

    def _distance(
        self, track: Track, detection: Detection
    ) -> float:
        """Compute distance between predicted track position and detection."""
        pred = track.predict()
        dx = pred[0] - detection.x_center
        dy = pred[1] - detection.y_center
        return float(np.sqrt(dx ** 2 + dy ** 2))

    def _iou(self, track: Track, detection: Detection) -> float:
        """Compute IoU between track's last bbox and detection bbox."""
        if track.last_detection is None:
            return 0.0

        t1 = track.last_detection.tlbr
        t2 = detection.tlbr

        y1 = max(t1[0], t2[0])
        x1 = max(t1[1], t2[1])
        y2 = min(t1[2], t2[2])
        x2 = min(t1[3], t2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (t1[2] - t1[0]) * (t1[3] - t1[1])
        area2 = (t2[2] - t2[0]) * (t2[3] - t2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    def _match_cost(self, track: Track, detection: Detection) -> float:
        """Combined matching cost (lower is better)."""
        dist = self._distance(track, detection)
        iou = self._iou(track, detection)
        # Weighted combination: distance + (1 - IoU)
        return dist * 0.7 + (1.0 - iou) * 0.3 * self.match_distance_threshold

    def _greedy_match(
        self,
        tracks: List[Track],
        detections: List[Detection],
    ) -> Tuple[List[Tuple[Track, Detection]], List[Track], List[Detection]]:
        """Greedy matching by minimum cost."""
        if not tracks or not detections:
            return [], list(tracks), list(detections)

        # Compute cost matrix
        costs = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                costs[i, j] = self._match_cost(track, det)

        matched: List[Tuple[Track, Detection]] = []
        used_tracks: set = set()
        used_dets: set = set()

        # Greedy: pick lowest cost pair iteratively
        flat_indices = np.argsort(costs, axis=None)
        for idx in flat_indices:
            i, j = divmod(int(idx), len(detections))
            if i in used_tracks or j in used_dets:
                continue
            if costs[i, j] > self.match_distance_threshold:
                continue
            matched.append((tracks[i], detections[j]))
            used_tracks.add(i)
            used_dets.add(j)

        unmatched_tracks = [t for i, t in enumerate(tracks) if i not in used_tracks]
        unmatched_dets = [d for j, d in enumerate(detections) if j not in used_dets]

        return matched, unmatched_tracks, unmatched_dets

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Process a new frame's detections and return active tracks.

        Args:
            detections: List of Detection objects from the current frame.

        Returns:
            List of active Track objects after update.
        """
        self._frame_count += 1

        # Split detections into high and low confidence
        high_dets = [d for d in detections if d.confidence >= self.high_threshold]
        low_dets = [
            d for d in detections
            if self.low_threshold <= d.confidence < self.high_threshold
        ]

        active = [t for t in self._tracks.values() if t.is_active]

        # Stage 1: Match high-confidence detections to active tracks
        matched, unmatched_tracks, unmatched_high = self._greedy_match(
            active, high_dets
        )

        for track, det in matched:
            track.update(det)

        # Stage 2: Match low-confidence detections to remaining tracks
        matched_low, still_unmatched, _ = self._greedy_match(
            unmatched_tracks, low_dets
        )

        for track, det in matched_low:
            track.update(det)

        # Mark unmatched tracks as missed
        for track in still_unmatched:
            track.mark_missed()
            if track.frames_since_update > self.max_frames_lost:
                track.is_active = False

        # Create new tracks for unmatched high-confidence detections
        for det in unmatched_high:
            new_track = Track(
                track_id=self._next_id,
                class_id=det.class_id,
            )
            new_track.update(det)
            self._tracks[self._next_id] = new_track
            self._next_id += 1

        return self.active_tracks

    def get_ball_tracks(self) -> List[Track]:
        """Return active tracks for ball class (class_id=0)."""
        return [t for t in self.active_tracks if t.class_id == 0]

    def reset(self) -> None:
        """Reset all tracks."""
        self._tracks.clear()
        self._next_id = 1
        self._frame_count = 0
        log(TASK, "info", "Tracker reset")


if __name__ == "__main__":
    # Simple demo
    tracker = ByteTracker()

    # Simulate a ball moving across frames
    for frame_idx in range(10):
        x = 100.0 + frame_idx * 20.0
        y = 200.0 + frame_idx * 5.0
        dets = [Detection(x, y, 15.0, 15.0, 0.85, class_id=0)]
        active = tracker.update(dets)

        for t in active:
            if t.class_id == 0:
                log(TASK, "info", f"Frame {frame_idx}", track_id=t.track_id,
                    position=t.position, speed=round(t.speed, 2))
