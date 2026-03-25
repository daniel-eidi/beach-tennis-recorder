"""
Main clip processing orchestrator.

AGENT-03 · TASK-03-02 through TASK-03-07

Coordinates cutting, thumbnailing, and validating clips produced from
the camera buffer when a rally ends.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from video.async_queue import ClipQueue, ClipTask
from video.ffmpeg_wrapper import (
    cut_clip,
    generate_thumbnail,
    get_video_info,
    validate_clip,
)
from video.naming_convention import (
    generate_clip_name,
    generate_thumbnail_name,
)

logger = logging.getLogger(__name__)

# Constants from the rally state machine contract
BUFFER_PRE_RALLY_SECONDS: float = 3.0
BUFFER_POST_RALLY_SECONDS: float = 2.0


def _structured_log(
    task: str,
    status: str,
    ms: Optional[float] = None,
    **extra: Any,
) -> None:
    entry: dict[str, Any] = {"agent": "03", "task": task, "status": status}
    if ms is not None:
        entry["ms"] = round(ms, 2)
    entry.update(extra)
    logger.info(json.dumps(entry))


@dataclass
class ClipResult:
    """Result of processing a single rally clip."""

    clip_path: str
    thumbnail_path: str
    duration_seconds: float
    file_size_bytes: int


class ClipProcessor:
    """Orchestrates the full clip-processing pipeline.

    Typical usage::

        processor = ClipProcessor(output_dir="/data/clips")
        processor.start()

        result = processor.process_rally(
            buffer_path="/tmp/buffer.mp4",
            start_time=42.5,
            end_time=58.3,
            match_id=1,
            rally_number=7,
        )

        # Or enqueue for async processing:
        processor.enqueue_rally(
            buffer_path="/tmp/buffer.mp4",
            start_time=42.5,
            end_time=58.3,
            match_id=1,
            rally_number=7,
            callback=lambda r: print("Done!", r),
        )
    """

    def __init__(
        self,
        output_dir: str = "clips",
        thumbnail_dir: Optional[str] = None,
        max_queue_size: int = 50,
        num_workers: int = 1,
    ) -> None:
        """
        Args:
            output_dir: Directory where clips are written.
            thumbnail_dir: Directory for thumbnails. Defaults to output_dir/thumbnails.
            max_queue_size: Maximum pending tasks for async queue.
            num_workers: Number of background worker threads.
        """
        self.output_dir = output_dir
        self.thumbnail_dir = thumbnail_dir or os.path.join(output_dir, "thumbnails")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.thumbnail_dir, exist_ok=True)

        self._queue = ClipQueue(
            processor_fn=self._process_task,
            max_size=max_queue_size,
            num_workers=num_workers,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background processing queue."""
        self._queue.start()

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the background processing queue."""
        self._queue.stop(timeout=timeout)

    # ------------------------------------------------------------------
    # Synchronous API
    # ------------------------------------------------------------------

    def process_rally(
        self,
        buffer_path: str,
        start_time: float,
        end_time: float,
        match_id: int | str,
        rally_number: int,
        timestamp: Optional[datetime] = None,
    ) -> ClipResult:
        """Process a rally synchronously (blocking).

        Applies pre/post padding, cuts the clip, generates a thumbnail,
        and validates the output.

        Args:
            buffer_path: Path to the camera buffer video.
            start_time: Rally start timestamp in seconds within the buffer.
            end_time: Rally end timestamp in seconds within the buffer.
            match_id: Unique match identifier.
            rally_number: Sequential rally number.
            timestamp: When the rally occurred (for filename). Defaults to now.

        Returns:
            A :class:`ClipResult` with paths and metadata.

        Raises:
            FileNotFoundError: If buffer_path does not exist.
            ValueError: If the generated clip fails validation.
        """
        t0 = time.perf_counter()

        if not os.path.isfile(buffer_path):
            raise FileNotFoundError(f"Buffer file not found: {buffer_path}")

        # Apply padding
        padded_start = max(0.0, start_time - BUFFER_PRE_RALLY_SECONDS)
        padded_end = end_time + BUFFER_POST_RALLY_SECONDS

        # Clamp end to buffer duration
        buffer_info = get_video_info(buffer_path)
        padded_end = min(padded_end, buffer_info["duration"])

        if padded_start >= padded_end:
            raise ValueError(
                f"Invalid time range after padding: {padded_start:.2f} - {padded_end:.2f}"
            )

        # Generate filenames
        ts = timestamp or datetime.now()
        clip_name = generate_clip_name(match_id, rally_number, ts)
        thumb_name = generate_thumbnail_name(clip_name)

        clip_path = os.path.join(self.output_dir, clip_name)
        thumb_path = os.path.join(self.thumbnail_dir, thumb_name)

        # Cut clip (stream copy for speed)
        cut_clip(buffer_path, padded_start, padded_end, clip_path, reencode=False)

        # Generate thumbnail from the middle of the clip
        generate_thumbnail(clip_path, thumb_path)

        # Validate
        if not validate_clip(clip_path, min_duration=1.0, max_size_mb=200.0):
            elapsed = (time.perf_counter() - t0) * 1000
            _structured_log(
                "03-07", "fail", ms=elapsed,
                reason="clip failed validation",
                clip=clip_path,
            )
            raise ValueError(f"Generated clip failed validation: {clip_path}")

        # Gather result
        clip_info = get_video_info(clip_path)
        result = ClipResult(
            clip_path=clip_path,
            thumbnail_path=thumb_path,
            duration_seconds=clip_info["duration"],
            file_size_bytes=clip_info["file_size_bytes"],
        )

        elapsed = (time.perf_counter() - t0) * 1000
        _structured_log(
            "03-02", "ok", ms=elapsed,
            clip=clip_path,
            duration=result.duration_seconds,
            size_bytes=result.file_size_bytes,
        )
        return result

    # ------------------------------------------------------------------
    # Asynchronous API
    # ------------------------------------------------------------------

    def enqueue_rally(
        self,
        buffer_path: str,
        start_time: float,
        end_time: float,
        match_id: int | str,
        rally_number: int,
        callback: Optional[Any] = None,
        error_callback: Optional[Any] = None,
    ) -> bool:
        """Enqueue a rally for background processing.

        Returns True if enqueued, False if the queue is full.
        """
        task = ClipTask(
            buffer_path=buffer_path,
            start_time=start_time,
            end_time=end_time,
            match_id=match_id,
            rally_number=rally_number,
            callback=callback,
            error_callback=error_callback,
        )
        return self._queue.enqueue(task)

    @property
    def pending_tasks(self) -> int:
        return self._queue.pending

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_task(self, task: ClipTask) -> ClipResult:
        """Adapter: convert a ClipTask into a process_rally call."""
        return self.process_rally(
            buffer_path=task.buffer_path,
            start_time=task.start_time,
            end_time=task.end_time,
            match_id=task.match_id,
            rally_number=task.rally_number,
        )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 5:
        print(
            "Usage: python -m video.clip_processor "
            "<buffer_path> <start_ts> <end_ts> <match_id> [rally_number]"
        )
        sys.exit(1)

    buf = sys.argv[1]
    s = float(sys.argv[2])
    e = float(sys.argv[3])
    mid = int(sys.argv[4])
    rn = int(sys.argv[5]) if len(sys.argv) > 5 else 1

    processor = ClipProcessor(output_dir="clips")
    result = processor.process_rally(buf, s, e, mid, rn)
    print(json.dumps({
        "clip_path": result.clip_path,
        "thumbnail_path": result.thumbnail_path,
        "duration_seconds": result.duration_seconds,
        "file_size_bytes": result.file_size_bytes,
    }, indent=2))
