"""
Unit tests for naming_convention module.

AGENT-03 · TASK-03-08
"""

from __future__ import annotations

from datetime import datetime

import pytest

from video.naming_convention import (
    generate_clip_name,
    generate_thumbnail_name,
    parse_clip_name,
)


class TestGenerateClipName:
    """Tests for generate_clip_name."""

    def test_format_matches_convention(self) -> None:
        ts = datetime(2026, 3, 25, 14, 30, 22)
        name = generate_clip_name(match_id=42, rally_number=7, timestamp=ts)
        assert name == "rally_42_007_20260325_143022.mp4"

    def test_rally_number_zero_padded(self) -> None:
        ts = datetime(2026, 1, 1, 0, 0, 0)
        name = generate_clip_name(match_id=1, rally_number=1, timestamp=ts)
        assert "_001_" in name

    def test_large_rally_number(self) -> None:
        ts = datetime(2026, 12, 31, 23, 59, 59)
        name = generate_clip_name(match_id=1, rally_number=999, timestamp=ts)
        assert "_999_" in name

    def test_default_timestamp_is_now(self) -> None:
        name = generate_clip_name(match_id=1, rally_number=1)
        assert name.startswith("rally_1_001_")
        assert name.endswith(".mp4")

    def test_string_match_id(self) -> None:
        ts = datetime(2026, 6, 15, 10, 0, 0)
        name = generate_clip_name(match_id="abc-123", rally_number=5, timestamp=ts)
        assert name == "rally_abc-123_005_20260615_100000.mp4"

    def test_special_chars_sanitized(self) -> None:
        ts = datetime(2026, 1, 1, 0, 0, 0)
        name = generate_clip_name(match_id="a/b\\c@d", rally_number=1, timestamp=ts)
        # Special characters should be stripped
        assert "/" not in name
        assert "\\" not in name
        assert "@" not in name
        assert name.startswith("rally_abcd_")


class TestGenerateThumbnailName:
    """Tests for generate_thumbnail_name."""

    def test_mp4_to_jpg(self) -> None:
        assert generate_thumbnail_name("rally_42_007_20260325_143022.mp4") == \
            "rally_42_007_20260325_143022.jpg"

    def test_with_directory_path(self) -> None:
        result = generate_thumbnail_name("/some/dir/rally_1_001_20260101_000000.mp4")
        assert result == "rally_1_001_20260101_000000.jpg"

    def test_non_mp4_gets_jpg_appended(self) -> None:
        result = generate_thumbnail_name("somefile.avi")
        assert result == "somefile.avi.jpg"


class TestParseClipName:
    """Tests for parse_clip_name."""

    def test_roundtrip(self) -> None:
        ts = datetime(2026, 3, 25, 14, 30, 22)
        name = generate_clip_name(match_id=42, rally_number=7, timestamp=ts)
        parsed = parse_clip_name(name)
        assert parsed["match_id"] == "42"
        assert parsed["rally_number"] == 7
        assert parsed["date"] == "20260325"
        assert parsed["time"] == "143022"
        assert parsed["timestamp"] == ts

    def test_parse_thumbnail(self) -> None:
        parsed = parse_clip_name("rally_42_007_20260325_143022.jpg")
        assert parsed["match_id"] == "42"
        assert parsed["rally_number"] == 7

    def test_parse_with_path(self) -> None:
        parsed = parse_clip_name("/clips/rally_1_001_20260101_120000.mp4")
        assert parsed["match_id"] == "1"
        assert parsed["rally_number"] == 1

    def test_parse_string_match_id(self) -> None:
        parsed = parse_clip_name("rally_abc-123_005_20260615_100000.mp4")
        assert parsed["match_id"] == "abc-123"

    def test_invalid_filename_raises(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            parse_clip_name("not_a_valid_name.mp4")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_clip_name("")

    def test_wrong_extension_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_clip_name("rally_1_001_20260101_000000.avi")
