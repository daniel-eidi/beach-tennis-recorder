"""
Integration tests: Rally -> Clip generation pipeline.

AGENT-05 | TASK-05-02

Tests the clip_processor module: cutting clips from buffer videos,
applying pre/post rally padding, naming, thumbnailing, and validation.
"""

from __future__ import annotations

import os
import sys
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.conftest import requires_ffmpeg, create_test_video

from video.clip_processor import ClipProcessor, ClipResult, BUFFER_PRE_RALLY_SECONDS, BUFFER_POST_RALLY_SECONDS
from video.ffmpeg_wrapper import cut_clip, generate_thumbnail, get_video_info, validate_clip
from video.naming_convention import generate_clip_name, generate_thumbnail_name, parse_clip_name
from video.async_queue import ClipQueue, ClipTask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def output_dir():
    tmpdir = tempfile.mkdtemp(prefix="bt_clip_out_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def buffer_video(session_tmp_dir):
    """A 20-second buffer video simulating camera output."""
    path = os.path.join(session_tmp_dir, "buffer_20s.mp4")
    if not os.path.exists(path):
        create_test_video(path, duration=20.0, fps=30, width=640, height=480)
    return path


# ---------------------------------------------------------------------------
# ClipProcessor integration
# ---------------------------------------------------------------------------

@requires_ffmpeg
class TestClipProcessor:
    """Test ClipProcessor synchronous processing."""

    def test_process_rally_creates_clip(self, buffer_video, output_dir):
        """Processing a rally should produce a clip and thumbnail."""
        processor = ClipProcessor(output_dir=output_dir)
        result = processor.process_rally(
            buffer_path=buffer_video,
            start_time=5.0,
            end_time=12.0,
            match_id=1,
            rally_number=3,
            timestamp=datetime(2026, 3, 25, 14, 30, 0),
        )

        assert isinstance(result, ClipResult)
        assert os.path.isfile(result.clip_path)
        assert os.path.isfile(result.thumbnail_path)
        assert result.duration_seconds > 0
        assert result.file_size_bytes > 0

    def test_clip_includes_pre_post_padding(self, buffer_video, output_dir):
        """Clip should include BUFFER_PRE_RALLY_SECONDS before and
        BUFFER_POST_RALLY_SECONDS after the rally timestamps."""
        processor = ClipProcessor(output_dir=output_dir)
        result = processor.process_rally(
            buffer_path=buffer_video,
            start_time=5.0,
            end_time=10.0,
            match_id=1,
            rally_number=1,
        )

        # Expected duration: (10 - 5) + 3 + 2 = 10 seconds (with padding)
        # But start is clamped to max(0, 5-3) = 2.0, end = min(20, 10+2) = 12.0
        # So expected ~10.0 seconds
        info = get_video_info(result.clip_path)
        expected_min = 5.0 + BUFFER_PRE_RALLY_SECONDS + BUFFER_POST_RALLY_SECONDS - 2.0
        # Allow some tolerance for stream copy inaccuracy
        assert info["duration"] >= expected_min * 0.7, (
            f"Clip duration {info['duration']}s shorter than expected ~{expected_min}s"
        )

    def test_padding_clamped_to_buffer_start(self, buffer_video, output_dir):
        """Pre-rally padding should not go below 0 (start of buffer)."""
        processor = ClipProcessor(output_dir=output_dir)
        result = processor.process_rally(
            buffer_path=buffer_video,
            start_time=1.0,  # 1.0 - 3.0 = -2.0 -> clamped to 0.0
            end_time=5.0,
            match_id=1,
            rally_number=1,
        )

        info = get_video_info(result.clip_path)
        # Should start from 0 and go to 7 (5 + 2), so ~7 seconds
        assert info["duration"] >= 4.0  # conservative check

    def test_padding_clamped_to_buffer_end(self, buffer_video, output_dir):
        """Post-rally padding should not exceed buffer duration."""
        processor = ClipProcessor(output_dir=output_dir)
        result = processor.process_rally(
            buffer_path=buffer_video,
            start_time=15.0,
            end_time=19.5,  # 19.5 + 2 = 21.5 -> clamped to 20.0
            match_id=1,
            rally_number=1,
        )

        info = get_video_info(result.clip_path)
        assert info["duration"] > 0

    def test_clip_naming_convention(self, buffer_video, output_dir):
        """Generated clip name should match the naming convention."""
        ts = datetime(2026, 3, 25, 14, 30, 22)
        processor = ClipProcessor(output_dir=output_dir)
        result = processor.process_rally(
            buffer_path=buffer_video,
            start_time=5.0,
            end_time=10.0,
            match_id=42,
            rally_number=7,
            timestamp=ts,
        )

        clip_basename = os.path.basename(result.clip_path)
        assert clip_basename == "rally_42_007_20260325_143022.mp4"

        thumb_basename = os.path.basename(result.thumbnail_path)
        assert thumb_basename == "rally_42_007_20260325_143022.jpg"

    def test_thumbnail_is_jpeg(self, buffer_video, output_dir):
        """Generated thumbnail should be a JPEG file."""
        processor = ClipProcessor(output_dir=output_dir)
        result = processor.process_rally(
            buffer_path=buffer_video,
            start_time=5.0,
            end_time=10.0,
            match_id=1,
            rally_number=1,
        )

        assert result.thumbnail_path.endswith(".jpg")
        assert os.path.getsize(result.thumbnail_path) > 0

    def test_clip_validation_passes(self, buffer_video, output_dir):
        """A normal clip should pass validation (>1s, <200MB)."""
        processor = ClipProcessor(output_dir=output_dir)
        result = processor.process_rally(
            buffer_path=buffer_video,
            start_time=5.0,
            end_time=10.0,
            match_id=1,
            rally_number=1,
        )

        assert validate_clip(result.clip_path, min_duration=1.0, max_size_mb=200.0)

    def test_missing_buffer_raises(self, output_dir):
        """Processing with a non-existent buffer should raise."""
        processor = ClipProcessor(output_dir=output_dir)
        with pytest.raises(FileNotFoundError):
            processor.process_rally(
                buffer_path="/nonexistent/buffer.mp4",
                start_time=0.0,
                end_time=5.0,
                match_id=1,
                rally_number=1,
            )

    def test_invalid_time_range_raises(self, buffer_video, output_dir):
        """start_time >= end_time after padding should raise ValueError.

        The processor applies padding: padded_start = max(0, start - 3),
        padded_end = end + 2. So to trigger padded_start >= padded_end,
        we need max(0, start - 3) >= end + 2.
        Example: start=5, end=0 -> padded_start=2, padded_end=2 -> raises.
        """
        processor = ClipProcessor(output_dir=output_dir)
        with pytest.raises(ValueError):
            processor.process_rally(
                buffer_path=buffer_video,
                start_time=5.0,
                end_time=0.0,  # padded_start=2.0, padded_end=2.0 -> raises
                match_id=1,
                rally_number=1,
            )


# ---------------------------------------------------------------------------
# Async queue integration
# ---------------------------------------------------------------------------

@requires_ffmpeg
class TestAsyncQueue:
    """Test the async clip processing queue."""

    def test_async_enqueue_and_process(self, buffer_video, output_dir):
        """Enqueued clips should be processed in the background."""
        import threading

        processor = ClipProcessor(output_dir=output_dir, num_workers=1)
        processor.start()

        results: List[ClipResult] = []
        errors: List[Exception] = []
        done_event = threading.Event()

        def on_done(result):
            results.append(result)
            done_event.set()

        def on_error(exc):
            errors.append(exc)
            done_event.set()

        enqueued = processor.enqueue_rally(
            buffer_path=buffer_video,
            start_time=2.0,
            end_time=8.0,
            match_id=1,
            rally_number=1,
            callback=on_done,
            error_callback=on_error,
        )

        assert enqueued is True

        # Wait for processing to complete (timeout 30s)
        done_event.wait(timeout=30.0)
        processor.stop(timeout=10.0)

        assert len(errors) == 0, f"Async processing failed: {errors}"
        assert len(results) == 1
        assert os.path.isfile(results[0].clip_path)

    def test_queue_processes_multiple_clips(self, buffer_video, output_dir):
        """Multiple enqueued clips should all be processed."""
        import threading

        processor = ClipProcessor(output_dir=output_dir, num_workers=2)
        processor.start()

        results: List[ClipResult] = []
        count = 3
        done_event = threading.Event()
        lock = threading.Lock()

        def on_done(result):
            with lock:
                results.append(result)
                if len(results) >= count:
                    done_event.set()

        for i in range(count):
            processor.enqueue_rally(
                buffer_path=buffer_video,
                start_time=2.0 + i,
                end_time=6.0 + i,
                match_id=1,
                rally_number=i + 1,
                callback=on_done,
            )

        done_event.wait(timeout=60.0)
        processor.stop(timeout=10.0)

        assert len(results) == count
        for r in results:
            assert os.path.isfile(r.clip_path)

    def test_queue_full_drops_task(self):
        """When the queue is full, new tasks should be dropped."""
        def slow_processor(task: ClipTask):
            import time
            time.sleep(10)
            return None

        q = ClipQueue(processor_fn=slow_processor, max_size=2, num_workers=1)
        q.start()

        # Fill the queue
        for i in range(5):
            q.enqueue(ClipTask(
                buffer_path="/tmp/fake.mp4",
                start_time=0, end_time=5,
                match_id=1, rally_number=i,
            ))

        # Some should have been dropped
        # (worker consumes 1, queue holds 2, so at least 2 dropped)
        q.stop(timeout=2.0)
        assert q.tasks_dropped >= 1
