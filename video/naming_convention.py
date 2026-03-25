"""
Clip naming convention for Beach Tennis Recorder.

AGENT-03 · TASK-03-03

Format: rally_{match_id}_{number}_{date}_{time}.mp4
Example: rally_42_007_20260325_143022.mp4
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Pattern: rally_{match_id}_{rally_number}_{YYYYMMDD}_{HHMMSS}.mp4
_CLIP_PATTERN = re.compile(
    r"^rally_(?P<match_id>[A-Za-z0-9_-]+)_(?P<rally_number>\d+)_"
    r"(?P<date>\d{8})_(?P<time>\d{6})\.mp4$"
)
_THUMB_PATTERN = re.compile(
    r"^rally_(?P<match_id>[A-Za-z0-9_-]+)_(?P<rally_number>\d+)_"
    r"(?P<date>\d{8})_(?P<time>\d{6})\.jpg$"
)


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


def _sanitize_id(value: Any) -> str:
    """Sanitize a match_id to contain only safe filename characters."""
    return re.sub(r"[^A-Za-z0-9_-]", "", str(value))


def generate_clip_name(
    match_id: int | str,
    rally_number: int,
    timestamp: Optional[datetime] = None,
) -> str:
    """Generate a standardized clip filename.

    Args:
        match_id: Unique match identifier.
        rally_number: Sequential rally number within the match.
        timestamp: When the rally occurred. Defaults to now.

    Returns:
        Filename string like ``rally_42_007_20260325_143022.mp4``.
    """
    if timestamp is None:
        timestamp = datetime.now()

    safe_id = _sanitize_id(match_id)
    date_str = timestamp.strftime("%Y%m%d")
    time_str = timestamp.strftime("%H%M%S")
    rally_str = f"{rally_number:03d}"

    name = f"rally_{safe_id}_{rally_str}_{date_str}_{time_str}.mp4"
    _structured_log("03-03", "ok", name=name)
    return name


def generate_thumbnail_name(clip_name: str) -> str:
    """Derive the thumbnail filename from a clip filename.

    Args:
        clip_name: The clip filename (with or without directory path).

    Returns:
        The same base name but with ``.jpg`` extension.
    """
    base = os.path.basename(clip_name)
    if base.endswith(".mp4"):
        return base[:-4] + ".jpg"
    return base + ".jpg"


def parse_clip_name(filename: str) -> dict[str, Any]:
    """Parse a clip or thumbnail filename back into its components.

    Args:
        filename: A filename (basename only or full path).

    Returns:
        Dict with keys: match_id (str), rally_number (int),
        date (str YYYYMMDD), time (str HHMMSS), timestamp (datetime).

    Raises:
        ValueError: If the filename does not match the expected pattern.
    """
    base = os.path.basename(filename)

    match = _CLIP_PATTERN.match(base) or _THUMB_PATTERN.match(base)
    if not match:
        _structured_log("03-03", "error", error="filename does not match pattern", filename=base)
        raise ValueError(
            f"Filename '{base}' does not match expected pattern "
            "rally_{{match_id}}_{{number}}_{{date}}_{{time}}.mp4|jpg"
        )

    date_str = match.group("date")
    time_str = match.group("time")
    ts = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")

    result: dict[str, Any] = {
        "match_id": match.group("match_id"),
        "rally_number": int(match.group("rally_number")),
        "date": date_str,
        "time": time_str,
        "timestamp": ts,
    }
    _structured_log("03-03", "ok", parsed=result["match_id"])
    return result


if __name__ == "__main__":
    # Quick demo
    name = generate_clip_name(match_id=42, rally_number=7)
    print(f"Generated: {name}")
    print(f"Thumbnail: {generate_thumbnail_name(name)}")
    parsed = parse_clip_name(name)
    print(f"Parsed: match_id={parsed['match_id']}, rally={parsed['rally_number']}, ts={parsed['timestamp']}")
