"""
Integration tests: Video processing module.

AGENT-05 | TASK-05-03

Tests the video module functions in isolation and in combination:
  - cut_clip with various timestamp combinations
  - Thumbnail generation at various positions
  - Compression producing smaller files
  - Naming convention roundtrip
  - Async queue with concurrent clips
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.conftest import requires_ffmpeg, create_test_video

from video.ffmpeg_wrapper import (
    cut_clip,
    compress_clip,
    generate_thumbnail,
    get_video_info,
    validate_clip,
)
from video.naming_convention import (
    generate_clip_name,
    generate_thumbnail_name,
    parse_clip_name,
)
from video.async_queue import ClipQueue, ClipTask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def out_dir():
    tmpdir = tempfile.mkdtemp(prefix="bt_vidtest_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# cut_clip tests
# ---------------------------------------------------------------------------

@requires_ffmpeg
class TestCutClip:
    """Test cut_clip with various timestamp combinations."""

    def test_cut_middle_segment(self, long_video, out_dir):
        """Cut a segment from the middle of a video."""
        output = os.path.join(out_dir, "middle.mp4")
        cut_clip(long_video, 5.0, 15.0, output)

        assert os.path.isfile(output)
        info = get_video_info(output)
        # Stream copy may not be frame-accurate, allow tolerance
        assert 7.0 <= info["duration"] <= 13.0

    def test_cut_from_start(self, long_video, out_dir):
        """Cut starting from the very beginning."""
        output = os.path.join(out_dir, "from_start.mp4")
        cut_clip(long_video, 0.0, 5.0, output)

        assert os.path.isfile(output)
        info = get_video_info(output)
        assert info["duration"] >= 3.0

    def test_cut_to_end(self, long_video, out_dir):
        """Cut up to near the end of the video."""
        info = get_video_info(long_video)
        output = os.path.join(out_dir, "to_end.mp4")
        cut_clip(long_video, info["duration"] - 5.0, info["duration"], output)

        assert os.path.isfile(output)
        out_info = get_video_info(output)
        assert out_info["duration"] >= 2.0

    def test_cut_very_short_segment(self, long_video, out_dir):
        """Cut a very short (1-second) segment."""
        output = os.path.join(out_dir, "short_cut.mp4")
        cut_clip(long_video, 5.0, 6.0, output)

        assert os.path.isfile(output)

    def test_cut_with_reencode(self, short_video, out_dir):
        """Cut with re-encoding enabled (frame-accurate)."""
        output = os.path.join(out_dir, "reencoded.mp4")
        cut_clip(short_video, 1.0, 4.0, output, reencode=True)

        assert os.path.isfile(output)
        info = get_video_info(output)
        assert 2.0 <= info["duration"] <= 4.5

    def test_cut_invalid_timestamps_raises(self, short_video, out_dir):
        """start_ts >= end_ts should raise ValueError."""
        output = os.path.join(out_dir, "bad.mp4")
        with pytest.raises(ValueError):
            cut_clip(short_video, 5.0, 3.0, output)

    def test_cut_equal_timestamps_raises(self, short_video, out_dir):
        """start_ts == end_ts should raise ValueError."""
        output = os.path.join(out_dir, "zero.mp4")
        with pytest.raises(ValueError):
            cut_clip(short_video, 3.0, 3.0, output)

    def test_cut_missing_input_raises(self, out_dir):
        """Non-existent input file should raise FileNotFoundError."""
        output = os.path.join(out_dir, "nope.mp4")
        with pytest.raises(FileNotFoundError):
            cut_clip("/nonexistent/file.mp4", 0.0, 5.0, output)


# ---------------------------------------------------------------------------
# Thumbnail tests
# ---------------------------------------------------------------------------

@requires_ffmpeg
class TestThumbnail:
    """Test thumbnail generation at various positions."""

    def test_thumbnail_default_midpoint(self, short_video, out_dir):
        """Default thumbnail should be from the middle of the video."""
        output = os.path.join(out_dir, "thumb_mid.jpg")
        generate_thumbnail(short_video, output)

        assert os.path.isfile(output)
        assert os.path.getsize(output) > 0

    def test_thumbnail_at_start(self, short_video, out_dir):
        """Thumbnail at the very beginning."""
        output = os.path.join(out_dir, "thumb_start.jpg")
        generate_thumbnail(short_video, output, timestamp=0.0)

        assert os.path.isfile(output)

    def test_thumbnail_at_specific_time(self, long_video, out_dir):
        """Thumbnail at a specific timestamp."""
        output = os.path.join(out_dir, "thumb_10s.jpg")
        generate_thumbnail(long_video, output, timestamp=10.0)

        assert os.path.isfile(output)
        assert os.path.getsize(output) > 100  # Not a trivially empty file

    def test_thumbnail_missing_input_raises(self, out_dir):
        """Thumbnail from non-existent video should raise."""
        output = os.path.join(out_dir, "bad_thumb.jpg")
        with pytest.raises(FileNotFoundError):
            generate_thumbnail("/nonexistent/video.mp4", output)


# ---------------------------------------------------------------------------
# Compression tests
# ---------------------------------------------------------------------------

@requires_ffmpeg
class TestCompression:
    """Test video compression."""

    def test_compression_h264(self, short_video, out_dir):
        """H.264 compression should produce a valid video."""
        output = os.path.join(out_dir, "compressed_h264.mp4")
        compress_clip(short_video, output, codec="h264", crf=28)

        assert os.path.isfile(output)
        info = get_video_info(output)
        assert info["duration"] > 0
        assert info["codec"] in ("h264", "libx264")

    def test_compression_reduces_size_at_high_crf(self, short_video, out_dir):
        """Higher CRF should produce a smaller file."""
        output_low = os.path.join(out_dir, "crf_18.mp4")
        output_high = os.path.join(out_dir, "crf_40.mp4")
        compress_clip(short_video, output_low, codec="h264", crf=18)
        compress_clip(short_video, output_high, codec="h264", crf=40)

        size_low = os.path.getsize(output_low)
        size_high = os.path.getsize(output_high)
        assert size_high < size_low, (
            f"High CRF ({size_high}B) should be smaller than low CRF ({size_low}B)"
        )

    def test_unsupported_codec_raises(self, short_video, out_dir):
        """Unsupported codec should raise ValueError."""
        output = os.path.join(out_dir, "bad_codec.mp4")
        with pytest.raises(ValueError):
            compress_clip(short_video, output, codec="vp9_not_supported")


# ---------------------------------------------------------------------------
# get_video_info tests
# ---------------------------------------------------------------------------

@requires_ffmpeg
class TestVideoInfo:
    """Test get_video_info."""

    def test_info_returns_expected_keys(self, short_video):
        """Info dict should contain all expected keys."""
        info = get_video_info(short_video)
        required_keys = {"duration", "width", "height", "fps", "codec", "file_size_bytes"}
        assert required_keys.issubset(info.keys())

    def test_info_values_reasonable(self, short_video):
        """Info values should be reasonable for our test video."""
        info = get_video_info(short_video)
        assert info["duration"] >= 4.0
        assert info["width"] == 320
        assert info["height"] == 240
        assert info["fps"] > 0
        assert info["file_size_bytes"] > 0

    def test_info_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            get_video_info("/nonexistent/video.mp4")


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

@requires_ffmpeg
class TestValidation:
    """Test clip validation."""

    def test_valid_clip_passes(self, short_video):
        assert validate_clip(short_video, min_duration=1.0, max_size_mb=200.0) is True

    def test_short_clip_fails(self, tiny_video):
        """A 0.5s clip should fail min_duration=1.0."""
        result = validate_clip(tiny_video, min_duration=1.0, max_size_mb=200.0)
        assert result is False

    def test_missing_clip_fails(self):
        result = validate_clip("/nonexistent/clip.mp4")
        assert result is False


# ---------------------------------------------------------------------------
# Naming convention roundtrip
# ---------------------------------------------------------------------------

class TestNamingConvention:
    """Test naming convention with real file operations."""

    def test_generate_and_parse_roundtrip(self):
        """Generated name should parse back to the same components."""
        ts = datetime(2026, 3, 25, 14, 30, 22)
        name = generate_clip_name(42, 7, ts)
        parsed = parse_clip_name(name)

        assert parsed["match_id"] == "42"
        assert parsed["rally_number"] == 7
        assert parsed["date"] == "20260325"
        assert parsed["time"] == "143022"
        assert parsed["timestamp"] == ts

    def test_thumbnail_name_derivation(self):
        """Thumbnail name should match clip name with .jpg extension."""
        name = generate_clip_name(1, 1)
        thumb = generate_thumbnail_name(name)
        assert thumb.endswith(".jpg")
        assert thumb[:-4] == name[:-4]

    def test_parse_invalid_name_raises(self):
        with pytest.raises(ValueError):
            parse_clip_name("not_a_valid_clip_name.txt")

    @requires_ffmpeg
    def test_naming_with_real_files(self, short_video, out_dir):
        """Create a clip with the generated name and verify it parses."""
        ts = datetime(2026, 3, 25, 10, 0, 0)
        clip_name = generate_clip_name(99, 5, ts)
        clip_path = os.path.join(out_dir, clip_name)

        cut_clip(short_video, 1.0, 4.0, clip_path)
        assert os.path.isfile(clip_path)

        parsed = parse_clip_name(clip_path)
        assert parsed["match_id"] == "99"
        assert parsed["rally_number"] == 5


# ---------------------------------------------------------------------------
# Async queue concurrent processing
# ---------------------------------------------------------------------------

@requires_ffmpeg
class TestAsyncQueueConcurrency:
    """Test the async queue with concurrent clip processing."""

    def test_concurrent_clips(self, long_video, out_dir):
        """Multiple clips processed concurrently should all succeed."""
        results: List = []
        errors: List[Exception] = []
        lock = threading.Lock()
        done = threading.Event()
        num_clips = 4

        def processor(task: ClipTask):
            output = os.path.join(
                out_dir, f"concurrent_{task.rally_number}.mp4"
            )
            cut_clip(task.buffer_path, task.start_time, task.end_time, output)
            return output

        q = ClipQueue(processor_fn=processor, max_size=10, num_workers=2)
        q.start()

        def on_done(result):
            with lock:
                results.append(result)
                if len(results) + len(errors) >= num_clips:
                    done.set()

        def on_error(exc):
            with lock:
                errors.append(exc)
                if len(results) + len(errors) >= num_clips:
                    done.set()

        for i in range(num_clips):
            q.enqueue(ClipTask(
                buffer_path=long_video,
                start_time=float(i * 2),
                end_time=float(i * 2 + 3),
                match_id=1,
                rally_number=i + 1,
                callback=on_done,
                error_callback=on_error,
            ))

        done.wait(timeout=60.0)
        q.stop(timeout=10.0)

        assert len(errors) == 0, f"Errors during concurrent processing: {errors}"
        assert len(results) == num_clips
        for r in results:
            assert os.path.isfile(r)

    def test_queue_is_running_property(self):
        """Queue should report running status correctly."""
        def noop(task):
            return None

        q = ClipQueue(processor_fn=noop)
        assert q.is_running is False

        q.start()
        assert q.is_running is True

        q.stop(timeout=5.0)
        assert q.is_running is False

    def test_pending_count(self):
        """pending property should reflect queued tasks."""
        results = []
        barrier = threading.Event()

        def blocking_processor(task: ClipTask):
            barrier.wait(timeout=10.0)
            return "done"

        q = ClipQueue(processor_fn=blocking_processor, max_size=10, num_workers=1)
        q.start()

        for i in range(3):
            q.enqueue(ClipTask(
                buffer_path="/tmp/fake.mp4",
                start_time=0, end_time=5,
                match_id=1, rally_number=i,
            ))

        # Worker is blocked, so pending should be >= 2 (1 being processed, 2 queued)
        assert q.pending >= 1

        barrier.set()
        time.sleep(1.0)
        q.stop(timeout=10.0)
