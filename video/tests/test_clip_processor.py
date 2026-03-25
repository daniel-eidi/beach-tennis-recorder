"""
Unit tests for clip_processor module.

AGENT-03 · TASK-03-08
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from video.clip_processor import (
    BUFFER_POST_RALLY_SECONDS,
    BUFFER_PRE_RALLY_SECONDS,
    ClipProcessor,
    ClipResult,
)
from video.ffmpeg_wrapper import get_video_info


class TestClipProcessorSync:
    """Synchronous process_rally tests."""

    def test_process_rally_produces_clip_and_thumbnail(
        self, long_video: str, tmp_output_dir: str
    ) -> None:
        processor = ClipProcessor(output_dir=tmp_output_dir)
        result = processor.process_rally(
            buffer_path=long_video,
            start_time=5.0,
            end_time=12.0,
            match_id=1,
            rally_number=1,
        )
        assert isinstance(result, ClipResult)
        assert os.path.isfile(result.clip_path)
        assert os.path.isfile(result.thumbnail_path)
        assert result.duration_seconds > 0
        assert result.file_size_bytes > 0

    def test_padding_applied(
        self, long_video: str, tmp_output_dir: str
    ) -> None:
        """Clip should be longer than end-start due to pre/post padding."""
        processor = ClipProcessor(output_dir=tmp_output_dir)
        start, end = 8.0, 12.0
        rally_duration = end - start  # 4 seconds

        result = processor.process_rally(
            buffer_path=long_video,
            start_time=start,
            end_time=end,
            match_id=2,
            rally_number=1,
        )

        # Expected: ~4s rally + 3s pre + 2s post = ~9s
        # Allow tolerance for stream-copy inaccuracy
        expected_min = rally_duration + BUFFER_PRE_RALLY_SECONDS - 1.0
        assert result.duration_seconds >= expected_min

    def test_padding_clamped_at_buffer_start(
        self, long_video: str, tmp_output_dir: str
    ) -> None:
        """Pre-padding should not go below 0."""
        processor = ClipProcessor(output_dir=tmp_output_dir)
        result = processor.process_rally(
            buffer_path=long_video,
            start_time=1.0,  # pre-padding would go to -2s, clamped to 0
            end_time=5.0,
            match_id=3,
            rally_number=1,
        )
        assert os.path.isfile(result.clip_path)
        assert result.duration_seconds > 0

    def test_buffer_not_found_raises(self, tmp_output_dir: str) -> None:
        processor = ClipProcessor(output_dir=tmp_output_dir)
        with pytest.raises(FileNotFoundError):
            processor.process_rally(
                buffer_path="/nonexistent/buffer.mp4",
                start_time=0.0,
                end_time=5.0,
                match_id=1,
                rally_number=1,
            )

    def test_clip_naming_follows_convention(
        self, long_video: str, tmp_output_dir: str
    ) -> None:
        processor = ClipProcessor(output_dir=tmp_output_dir)
        result = processor.process_rally(
            buffer_path=long_video,
            start_time=5.0,
            end_time=10.0,
            match_id=99,
            rally_number=3,
        )
        basename = os.path.basename(result.clip_path)
        assert basename.startswith("rally_99_003_")
        assert basename.endswith(".mp4")

        thumb_basename = os.path.basename(result.thumbnail_path)
        assert thumb_basename.startswith("rally_99_003_")
        assert thumb_basename.endswith(".jpg")


class TestClipProcessorAsync:
    """Async queue processing tests."""

    def test_enqueue_and_callback(
        self, long_video: str, tmp_output_dir: str
    ) -> None:
        processor = ClipProcessor(output_dir=tmp_output_dir)
        processor.start()

        results: list[ClipResult] = []
        event = threading.Event()

        def on_done(result: ClipResult) -> None:
            results.append(result)
            event.set()

        processor.enqueue_rally(
            buffer_path=long_video,
            start_time=5.0,
            end_time=10.0,
            match_id=10,
            rally_number=1,
            callback=on_done,
        )

        event.wait(timeout=30.0)
        processor.stop()

        assert len(results) == 1
        assert os.path.isfile(results[0].clip_path)

    def test_enqueue_returns_false_when_full(
        self, long_video: str, tmp_output_dir: str
    ) -> None:
        # Create processor with a tiny queue
        processor = ClipProcessor(
            output_dir=tmp_output_dir,
            max_queue_size=1,
        )
        # Do NOT start workers so tasks pile up
        # Manually fill the queue
        result1 = processor.enqueue_rally(
            buffer_path=long_video,
            start_time=0.0,
            end_time=5.0,
            match_id=1,
            rally_number=1,
        )
        assert result1 is True

        # Queue is full (size=1), next enqueue should fail
        result2 = processor.enqueue_rally(
            buffer_path=long_video,
            start_time=0.0,
            end_time=5.0,
            match_id=1,
            rally_number=2,
        )
        assert result2 is False

    def test_error_callback_on_failure(
        self, tmp_output_dir: str
    ) -> None:
        processor = ClipProcessor(output_dir=tmp_output_dir)
        processor.start()

        errors: list[Exception] = []
        event = threading.Event()

        def on_error(exc: Exception) -> None:
            errors.append(exc)
            event.set()

        processor.enqueue_rally(
            buffer_path="/nonexistent/buffer.mp4",
            start_time=0.0,
            end_time=5.0,
            match_id=1,
            rally_number=1,
            error_callback=on_error,
        )

        event.wait(timeout=10.0)
        processor.stop()

        assert len(errors) == 1
        assert isinstance(errors[0], FileNotFoundError)
