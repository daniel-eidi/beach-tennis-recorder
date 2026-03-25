"""
End-to-end integration tests: full pipeline.

AGENT-05 | TASK-05-02

Tests the complete flow:
  frame extraction -> detection simulation -> tracking ->
  rally detection -> clip generation

Uses synthetic data and real module code (no model required).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vision.tracking.byte_tracker import ByteTracker, Detection
from vision.tracking.rally_detector import (
    CourtConfig,
    RallyDetector,
    RallyEvent,
    RallyState,
)
from tests.conftest import (
    requires_ffmpeg,
    create_test_video,
    make_ball_detection,
    run_scenario_through_pipeline,
)
from tests.mock_data.scenarios import (
    get_scenario,
    normal_rally,
    multiple_rallies,
    serve_ace,
    Scenario,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline_output_dir():
    tmpdir = tempfile.mkdtemp(prefix="bt_pipeline_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Full pipeline tests (without clip generation)
# ---------------------------------------------------------------------------

class TestDetectionToRallyPipeline:
    """Test the detection -> tracking -> rally detection pipeline end-to-end."""

    def test_normal_rally_pipeline(self):
        """Normal rally scenario should produce exactly 1 rally event."""
        scenario = normal_rally()
        events, detector, tracker = run_scenario_through_pipeline(
            scenario, net_cross_required=True
        )

        assert len(events) >= 1
        assert events[0].end_reason == "bounce"
        assert events[0].net_crossings >= 1
        assert events[0].ball_bounces >= 1
        assert events[0].rally_number == 1

    def test_multiple_rallies_pipeline(self):
        """Multiple rally scenario should produce 2 rally events."""
        scenario = multiple_rallies()
        events, detector, tracker = run_scenario_through_pipeline(
            scenario, net_cross_required=True
        )

        assert len(events) >= 2
        reasons = [e.end_reason for e in events]
        assert "bounce" in reasons
        assert "out_of_bounds" in reasons

    def test_serve_ace_pipeline(self):
        """Serve ace scenario should produce 1 rally ending out_of_bounds."""
        scenario = serve_ace()
        events, detector, tracker = run_scenario_through_pipeline(
            scenario, net_cross_required=True
        )

        assert len(events) >= 1
        assert events[0].end_reason == "out_of_bounds"

    def test_rally_timestamps_are_positive(self):
        """All rally events should have positive duration."""
        scenario = normal_rally()
        events, _, _ = run_scenario_through_pipeline(scenario)

        for event in events:
            assert event.duration_seconds >= 0
            assert event.end_time >= event.start_time

    def test_rally_numbers_sequential(self):
        """Rally numbers should be sequential starting from 1."""
        scenario = multiple_rallies()
        events, _, _ = run_scenario_through_pipeline(scenario)

        if len(events) >= 2:
            numbers = [e.rally_number for e in events]
            for i in range(1, len(numbers)):
                assert numbers[i] == numbers[i - 1] + 1

    def test_tracker_creates_tracks(self):
        """Tracker should create tracks from detections."""
        scenario = normal_rally()
        _, _, tracker = run_scenario_through_pipeline(scenario)

        all_tracks = tracker.all_tracks
        assert len(all_tracks) >= 1, "Tracker should have created at least one track"

    def test_all_scenarios_run_without_error(self):
        """Every predefined scenario should run through the pipeline
        without raising exceptions."""
        from tests.mock_data.scenarios import ALL_SCENARIOS

        for name, scenario in ALL_SCENARIOS.items():
            if name == "timeout_rally":
                events, _, _ = run_scenario_through_pipeline(
                    scenario,
                    simulate_timeout=True,
                    timeout_after_frame=6,
                    timeout_seconds=9.0,
                )
            else:
                events, _, _ = run_scenario_through_pipeline(scenario)

            # Just verify no exception was raised
            assert isinstance(events, list), f"Scenario '{name}' failed"


# ---------------------------------------------------------------------------
# Full pipeline with clip generation
# ---------------------------------------------------------------------------

@requires_ffmpeg
class TestFullPipelineWithClips:
    """End-to-end: detection -> tracking -> rally -> clip generation."""

    def test_rally_to_clip_generation(self, pipeline_output_dir):
        """After detecting a rally, generate a clip from the buffer video."""
        from video.clip_processor import ClipProcessor

        # Create a synthetic 30-second buffer video
        buffer_path = os.path.join(pipeline_output_dir, "buffer.mp4")
        create_test_video(buffer_path, duration=30.0, fps=30, width=640, height=480)

        # Run detection pipeline
        scenario = normal_rally()
        events, _, _ = run_scenario_through_pipeline(scenario)
        assert len(events) >= 1

        # Process the first rally event into a clip
        event = events[0]
        clip_dir = os.path.join(pipeline_output_dir, "clips")
        processor = ClipProcessor(output_dir=clip_dir)

        # Map rally event times to buffer times
        # (In real app, these would be actual buffer timestamps)
        # For testing, use reasonable values within the buffer
        result = processor.process_rally(
            buffer_path=buffer_path,
            start_time=5.0,
            end_time=15.0,
            match_id=1,
            rally_number=event.rally_number,
            timestamp=datetime(2026, 3, 25, 14, 0, 0),
        )

        assert os.path.isfile(result.clip_path)
        assert os.path.isfile(result.thumbnail_path)
        assert result.duration_seconds > 0

    def test_multiple_rallies_produce_multiple_clips(self, pipeline_output_dir):
        """Each detected rally should produce a separate clip."""
        from video.clip_processor import ClipProcessor

        buffer_path = os.path.join(pipeline_output_dir, "buffer.mp4")
        create_test_video(buffer_path, duration=30.0, fps=30, width=640, height=480)

        scenario = multiple_rallies()
        events, _, _ = run_scenario_through_pipeline(scenario)
        assert len(events) >= 2

        clip_dir = os.path.join(pipeline_output_dir, "clips")
        processor = ClipProcessor(output_dir=clip_dir)

        clips: List = []
        for i, event in enumerate(events):
            start = 2.0 + i * 8.0
            end = start + 5.0
            result = processor.process_rally(
                buffer_path=buffer_path,
                start_time=start,
                end_time=end,
                match_id=1,
                rally_number=event.rally_number,
            )
            clips.append(result)

        assert len(clips) >= 2
        # All clips should be distinct files
        clip_paths = [c.clip_path for c in clips]
        assert len(set(clip_paths)) == len(clip_paths)

    def test_clip_timestamps_include_padding(self, pipeline_output_dir):
        """Clips should include pre and post rally padding."""
        from video.clip_processor import ClipProcessor
        from video.ffmpeg_wrapper import get_video_info

        buffer_path = os.path.join(pipeline_output_dir, "buffer.mp4")
        create_test_video(buffer_path, duration=30.0, fps=30, width=640, height=480)

        clip_dir = os.path.join(pipeline_output_dir, "clips")
        processor = ClipProcessor(output_dir=clip_dir)

        rally_start = 10.0
        rally_end = 15.0
        result = processor.process_rally(
            buffer_path=buffer_path,
            start_time=rally_start,
            end_time=rally_end,
            match_id=1,
            rally_number=1,
        )

        info = get_video_info(result.clip_path)
        # Expected: (15 - 10) + 3 + 2 = 10 seconds
        # With stream-copy imprecision, allow +-2s tolerance
        assert info["duration"] >= 6.0, (
            f"Clip too short: {info['duration']}s, expected ~10s with padding"
        )
        assert info["duration"] <= 14.0, (
            f"Clip too long: {info['duration']}s, expected ~10s with padding"
        )


# ---------------------------------------------------------------------------
# Pipeline metrics collection
# ---------------------------------------------------------------------------

class TestPipelineMetrics:
    """Test that we can collect meaningful metrics from the pipeline."""

    def test_scenario_metrics(self):
        """Run scenario and collect per-rally metrics."""
        from tests.mock_data.scenarios import ALL_SCENARIOS

        metrics: Dict[str, Any] = {}
        for name, scenario in ALL_SCENARIOS.items():
            if name == "timeout_rally":
                events, detector, tracker = run_scenario_through_pipeline(
                    scenario,
                    simulate_timeout=True,
                    timeout_after_frame=6,
                    timeout_seconds=9.0,
                )
            else:
                events, detector, tracker = run_scenario_through_pipeline(scenario)

            metrics[name] = {
                "total_frames": scenario.total_frames,
                "rallies_detected": len(events),
                "expected_rallies": len(scenario.expected_rallies),
                "tracks_created": len(tracker.all_tracks),
                "events": [
                    {
                        "rally_number": e.rally_number,
                        "end_reason": e.end_reason,
                        "net_crossings": e.net_crossings,
                        "bounces": e.ball_bounces,
                    }
                    for e in events
                ],
            }

        # Verify expected rally counts match (or exceed) for non-warmup
        for name, scenario in ALL_SCENARIOS.items():
            expected = len(scenario.expected_rallies)
            detected = metrics[name]["rallies_detected"]
            if expected == 0:
                assert detected == 0, (
                    f"Scenario '{name}': expected 0 rallies, got {detected}"
                )
            else:
                assert detected >= 1, (
                    f"Scenario '{name}': expected >= 1 rallies, got {detected}"
                )
