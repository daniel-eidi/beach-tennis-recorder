"""
FFmpeg wrapper with clean interface for clip processing.

AGENT-03 · TASK-03-01, TASK-03-02, TASK-03-04, TASK-03-05, TASK-03-07

Provides functions for cutting clips, generating thumbnails,
compressing video, and validating output files.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

import ffmpeg

logger = logging.getLogger(__name__)


def _structured_log(
    task: str,
    status: str,
    ms: Optional[float] = None,
    **extra: Any,
) -> None:
    """Emit a structured JSON log line."""
    entry: dict[str, Any] = {
        "agent": "03",
        "task": task,
        "status": status,
    }
    if ms is not None:
        entry["ms"] = round(ms, 2)
    entry.update(extra)
    logger.info(json.dumps(entry))


def cut_clip(
    input_path: str,
    start_ts: float,
    end_ts: float,
    output_path: str,
    reencode: bool = False,
) -> str:
    """Cut a segment from a video file.

    By default uses stream copy (no re-encoding) for speed.

    Args:
        input_path: Path to the source video.
        start_ts: Start timestamp in seconds.
        end_ts: End timestamp in seconds.
        output_path: Path for the output clip.
        reencode: If True, re-encode the clip (slower but frame-accurate).

    Returns:
        The output_path on success.

    Raises:
        FileNotFoundError: If input_path does not exist.
        ffmpeg.Error: If ffmpeg processing fails.
        ValueError: If start_ts >= end_ts.
    """
    t0 = time.perf_counter()

    if not os.path.isfile(input_path):
        _structured_log("03-02", "error", error="input file not found", path=input_path)
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if start_ts >= end_ts:
        _structured_log("03-02", "error", error="start_ts >= end_ts")
        raise ValueError(f"start_ts ({start_ts}) must be less than end_ts ({end_ts})")

    duration = end_ts - start_ts

    try:
        stream = ffmpeg.input(input_path, ss=start_ts, t=duration)
        if reencode:
            stream = ffmpeg.output(stream, output_path, vcodec="libx264", acodec="aac")
        else:
            stream = ffmpeg.output(stream, output_path, c="copy")
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
    except ffmpeg.Error as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        _structured_log("03-02", "error", ms=elapsed, error=str(exc))
        raise

    elapsed = (time.perf_counter() - t0) * 1000
    _structured_log("03-02", "ok", ms=elapsed, output=output_path)
    return output_path


def generate_thumbnail(
    video_path: str,
    output_path: str,
    timestamp: Optional[float] = None,
) -> str:
    """Extract a single frame as a JPEG thumbnail.

    If *timestamp* is None the middle frame of the video is used.

    Args:
        video_path: Path to the source video.
        output_path: Path for the output JPEG.
        timestamp: Timestamp in seconds to extract. Defaults to midpoint.

    Returns:
        The output_path on success.
    """
    t0 = time.perf_counter()

    if not os.path.isfile(video_path):
        _structured_log("03-04", "error", error="video file not found", path=video_path)
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if timestamp is None:
        info = get_video_info(video_path)
        timestamp = info["duration"] / 2.0

    try:
        stream = (
            ffmpeg
            .input(video_path, ss=timestamp)
            .output(output_path, vframes=1, format="image2", vcodec="mjpeg")
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
    except ffmpeg.Error as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        _structured_log("03-04", "error", ms=elapsed, error=str(exc))
        raise

    elapsed = (time.perf_counter() - t0) * 1000
    _structured_log("03-04", "ok", ms=elapsed, output=output_path)
    return output_path


def compress_clip(
    input_path: str,
    output_path: str,
    codec: str = "h264",
    crf: int = 23,
) -> str:
    """Compress a video clip using H.264 or H.265.

    Args:
        input_path: Path to the source clip.
        output_path: Path for the compressed output.
        codec: ``"h264"`` or ``"h265"``.
        crf: Constant Rate Factor (lower = higher quality). 0-51.

    Returns:
        The output_path on success.
    """
    t0 = time.perf_counter()

    if not os.path.isfile(input_path):
        _structured_log("03-05", "error", error="input file not found", path=input_path)
        raise FileNotFoundError(f"Input file not found: {input_path}")

    codec_map = {
        "h264": "libx264",
        "h265": "libx265",
        "hevc": "libx265",
    }
    vcodec = codec_map.get(codec.lower())
    if vcodec is None:
        _structured_log("03-05", "error", error=f"unsupported codec: {codec}")
        raise ValueError(f"Unsupported codec: {codec}. Use 'h264' or 'h265'.")

    try:
        stream = (
            ffmpeg
            .input(input_path)
            .output(output_path, vcodec=vcodec, crf=crf, acodec="aac")
        )
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
    except ffmpeg.Error as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        _structured_log("03-05", "error", ms=elapsed, error=str(exc))
        raise

    elapsed = (time.perf_counter() - t0) * 1000
    _structured_log("03-05", "ok", ms=elapsed, output=output_path, codec=codec, crf=crf)
    return output_path


def get_video_info(video_path: str) -> dict[str, Any]:
    """Return metadata about a video file.

    Returns:
        Dict with keys: duration (float, seconds), width (int), height (int),
        fps (float), codec (str), file_size_bytes (int).
    """
    t0 = time.perf_counter()

    if not os.path.isfile(video_path):
        _structured_log("03-01", "error", error="video file not found", path=video_path)
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        probe = ffmpeg.probe(video_path)
    except ffmpeg.Error as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        _structured_log("03-01", "error", ms=elapsed, error=str(exc))
        raise

    video_stream = next(
        (s for s in probe["streams"] if s["codec_type"] == "video"),
        None,
    )

    if video_stream is None:
        _structured_log("03-01", "error", error="no video stream found")
        raise ValueError(f"No video stream found in {video_path}")

    # Parse frame rate from ratio string like "30/1"
    r_frame_rate = video_stream.get("r_frame_rate", "0/1")
    num, den = r_frame_rate.split("/")
    fps = float(num) / float(den) if float(den) != 0 else 0.0

    duration = float(probe.get("format", {}).get("duration", 0))
    file_size_bytes = os.path.getsize(video_path)

    info: dict[str, Any] = {
        "duration": duration,
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "fps": fps,
        "codec": video_stream.get("codec_name", "unknown"),
        "file_size_bytes": file_size_bytes,
    }

    elapsed = (time.perf_counter() - t0) * 1000
    _structured_log("03-01", "ok", ms=elapsed, info=info)
    return info


def validate_clip(
    video_path: str,
    min_duration: float = 1.0,
    max_size_mb: float = 200.0,
) -> bool:
    """Validate a generated clip meets quality constraints.

    Args:
        video_path: Path to the clip to validate.
        min_duration: Minimum acceptable duration in seconds.
        max_size_mb: Maximum acceptable file size in megabytes.

    Returns:
        True if the clip passes all checks.
    """
    t0 = time.perf_counter()

    if not os.path.isfile(video_path):
        _structured_log("03-07", "error", error="clip file not found", path=video_path)
        return False

    try:
        info = get_video_info(video_path)
    except Exception as exc:
        _structured_log("03-07", "error", error=f"probe failed: {exc}")
        return False

    duration_ok = info["duration"] >= min_duration
    size_mb = info["file_size_bytes"] / (1024 * 1024)
    size_ok = size_mb <= max_size_mb

    is_valid = duration_ok and size_ok

    elapsed = (time.perf_counter() - t0) * 1000
    _structured_log(
        "03-07",
        "ok" if is_valid else "fail",
        ms=elapsed,
        duration=info["duration"],
        size_mb=round(size_mb, 2),
        duration_ok=duration_ok,
        size_ok=size_ok,
    )
    return is_valid


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python ffmpeg_wrapper.py <video_path>")
        sys.exit(1)

    path = sys.argv[1]
    info = get_video_info(path)
    print(json.dumps(info, indent=2))
    print(f"Valid: {validate_clip(path)}")
