"""
Unit tests for ffmpeg_wrapper module.

AGENT-03 · TASK-03-08
"""

from __future__ import annotations

import os

import pytest

from video.ffmpeg_wrapper import (
    compress_clip,
    cut_clip,
    generate_thumbnail,
    get_video_info,
    validate_clip,
)


class TestGetVideoInfo:
    """Tests for get_video_info."""

    def test_returns_expected_keys(self, short_video: str) -> None:
        info = get_video_info(short_video)
        assert "duration" in info
        assert "width" in info
        assert "height" in info
        assert "fps" in info
        assert "codec" in info
        assert "file_size_bytes" in info

    def test_duration_approximately_correct(self, short_video: str) -> None:
        info = get_video_info(short_video)
        assert 4.5 <= info["duration"] <= 6.0

    def test_resolution(self, short_video: str) -> None:
        info = get_video_info(short_video)
        assert info["width"] == 320
        assert info["height"] == 240

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            get_video_info("/nonexistent/video.mp4")


class TestCutClip:
    """Tests for cut_clip."""

    def test_cut_produces_output(self, short_video: str, tmp_output_dir: str) -> None:
        out = os.path.join(tmp_output_dir, "cut.mp4")
        result = cut_clip(short_video, 1.0, 3.0, out)
        assert result == out
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0

    def test_cut_duration_approximately_correct(
        self, short_video: str, tmp_output_dir: str
    ) -> None:
        out = os.path.join(tmp_output_dir, "cut_dur.mp4")
        cut_clip(short_video, 1.0, 3.0, out)
        info = get_video_info(out)
        # Stream copy may not be perfectly frame-accurate
        assert 1.0 <= info["duration"] <= 3.5

    def test_cut_with_reencode(self, short_video: str, tmp_output_dir: str) -> None:
        out = os.path.join(tmp_output_dir, "cut_re.mp4")
        cut_clip(short_video, 0.5, 2.5, out, reencode=True)
        assert os.path.isfile(out)

    def test_invalid_times_raises(self, short_video: str, tmp_output_dir: str) -> None:
        out = os.path.join(tmp_output_dir, "bad.mp4")
        with pytest.raises(ValueError, match="start_ts"):
            cut_clip(short_video, 5.0, 2.0, out)

    def test_file_not_found_raises(self, tmp_output_dir: str) -> None:
        out = os.path.join(tmp_output_dir, "out.mp4")
        with pytest.raises(FileNotFoundError):
            cut_clip("/nonexistent/video.mp4", 0.0, 1.0, out)


class TestGenerateThumbnail:
    """Tests for generate_thumbnail."""

    def test_thumbnail_created(self, short_video: str, tmp_output_dir: str) -> None:
        out = os.path.join(tmp_output_dir, "thumb.jpg")
        result = generate_thumbnail(short_video, out)
        assert result == out
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 0

    def test_thumbnail_at_specific_time(
        self, short_video: str, tmp_output_dir: str
    ) -> None:
        out = os.path.join(tmp_output_dir, "thumb_t1.jpg")
        generate_thumbnail(short_video, out, timestamp=1.0)
        assert os.path.isfile(out)

    def test_file_not_found_raises(self, tmp_output_dir: str) -> None:
        out = os.path.join(tmp_output_dir, "thumb.jpg")
        with pytest.raises(FileNotFoundError):
            generate_thumbnail("/nonexistent/video.mp4", out)


class TestCompressClip:
    """Tests for compress_clip."""

    def test_compress_h264(self, short_video: str, tmp_output_dir: str) -> None:
        out = os.path.join(tmp_output_dir, "compressed.mp4")
        result = compress_clip(short_video, out, codec="h264", crf=28)
        assert result == out
        assert os.path.isfile(out)

    def test_unsupported_codec_raises(
        self, short_video: str, tmp_output_dir: str
    ) -> None:
        out = os.path.join(tmp_output_dir, "bad_codec.mp4")
        with pytest.raises(ValueError, match="Unsupported codec"):
            compress_clip(short_video, out, codec="vp9")

    def test_file_not_found_raises(self, tmp_output_dir: str) -> None:
        out = os.path.join(tmp_output_dir, "out.mp4")
        with pytest.raises(FileNotFoundError):
            compress_clip("/nonexistent/video.mp4", out)


class TestValidateClip:
    """Tests for validate_clip."""

    def test_valid_clip_passes(self, short_video: str) -> None:
        assert validate_clip(short_video, min_duration=1.0, max_size_mb=200.0) is True

    def test_too_short_clip_fails(self, tiny_video: str) -> None:
        assert validate_clip(tiny_video, min_duration=1.0) is False

    def test_nonexistent_file_fails(self) -> None:
        assert validate_clip("/nonexistent/video.mp4") is False

    def test_max_size_exceeded_fails(self, short_video: str) -> None:
        # Set absurdly low max size
        assert validate_clip(short_video, max_size_mb=0.0001) is False
