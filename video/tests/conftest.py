"""
Shared pytest fixtures for video module tests.

Creates small synthetic test videos using ffmpeg for use in unit tests.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest

# Check if ffmpeg binary is available
_FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


def _create_test_video(path: str, duration: float = 5.0, fps: int = 30) -> str:
    """Create a minimal test video using ffmpeg CLI.

    Generates a video with a color test pattern and silent audio.
    """
    import subprocess

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=blue:size=320x240:rate={fps}:d={duration}",
        "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=mono",
        "-t", str(duration),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "aac", "-b:a", "32k",
        "-pix_fmt", "yuv420p",
        path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


@pytest.fixture(scope="session")
def test_video_dir():
    """Session-scoped temp directory for test videos.

    Cleaned up after all tests complete.
    """
    tmpdir = tempfile.mkdtemp(prefix="bt_video_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="session")
def short_video(test_video_dir: str) -> str:
    """A 5-second test video (320x240, 30fps, H.264)."""
    if not _FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg binary not found on PATH")
    path = os.path.join(test_video_dir, "short_5s.mp4")
    return _create_test_video(path, duration=5.0)


@pytest.fixture(scope="session")
def long_video(test_video_dir: str) -> str:
    """A 20-second test video (320x240, 30fps, H.264)."""
    if not _FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg binary not found on PATH")
    path = os.path.join(test_video_dir, "long_20s.mp4")
    return _create_test_video(path, duration=20.0)


@pytest.fixture
def tmp_output_dir():
    """Per-test temp directory for output files. Cleaned up after each test."""
    tmpdir = tempfile.mkdtemp(prefix="bt_video_out_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture(scope="session")
def tiny_video(test_video_dir: str) -> str:
    """A very short 0.5-second video (will fail min-duration validation)."""
    if not _FFMPEG_AVAILABLE:
        pytest.skip("ffmpeg binary not found on PATH")
    path = os.path.join(test_video_dir, "tiny_0.5s.mp4")
    return _create_test_video(path, duration=0.5)
