"""
Beach Tennis Recorder - YouTube Video Testing Utility
AGENT-02 | TASK-02-14

Downloads beach tennis videos from YouTube and runs the full inference +
tracking + rally detection pipeline to validate the model.

Usage:
    python -m vision.utils.youtube_test --url "https://youtube.com/watch?v=..." --model models/best.tflite
    python -m vision.utils.youtube_test --channel "https://www.youtube.com/@btcanallive" --max-videos 3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Project imports — use relative imports when run as a module,
# or adjust sys.path for direct script execution.
try:
    from ..inference_test import (
        load_tflite_model,
        preprocess_image,
        postprocess_detections,
        run_tflite_inference,
        CONFIDENCE_THRESHOLD,
        IMG_SIZE,
    )
    from ..tracking.byte_tracker import ByteTracker, Detection
    from ..tracking.rally_detector import RallyDetector, RallyEvent, RallyState
    from .visualization import annotate_frame
    from .youtube_download import download_video, list_channel_videos
except ImportError:
    # Fallback for direct script execution
    _project_dir = str(Path(__file__).resolve().parent.parent.parent)
    if _project_dir not in sys.path:
        sys.path.insert(0, _project_dir)
    from vision.inference_test import (
        load_tflite_model,
        preprocess_image,
        postprocess_detections,
        run_tflite_inference,
        CONFIDENCE_THRESHOLD,
        IMG_SIZE,
    )
    from vision.tracking.byte_tracker import ByteTracker, Detection
    from vision.tracking.rally_detector import RallyDetector, RallyEvent, RallyState
    from vision.utils.visualization import annotate_frame
    from vision.utils.youtube_download import download_video, list_channel_videos


AGENT = "02"
TASK = "02-14"
DEFAULT_CHANNEL = "https://www.youtube.com/@btcanallive"
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"


def log(task: str, status: str, message: str = "", **kwargs: Any) -> None:
    """Emit structured JSON log line."""
    entry: Dict[str, Any] = {
        "agent": AGENT,
        "task": task,
        "status": status,
        "message": message,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    entry.update(kwargs)
    print(json.dumps(entry), flush=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="YouTube video testing utility for Beach Tennis vision model"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Specific YouTube video URL to test",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=DEFAULT_CHANNEL,
        help=f"YouTube channel URL (default: {DEFAULT_CHANNEL})",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=3,
        help="Max videos to download from channel (default: 3)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODELS_DIR / "best.tflite"),
        help="Path to TFLite model (default: models/best.tflite)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_results",
        help="Directory to save results (default: test_results/)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frames per second to extract for inference (default: 10)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse already downloaded videos in output-dir/downloads/",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Max seconds to process per video (default: 60)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="720p",
        choices=["360p", "480p", "720p", "1080p"],
        help="Download resolution (default: 720p)",
    )
    return parser.parse_args()


def _detections_to_tracker_format(
    raw_detections: List[dict],
) -> List[Detection]:
    """Convert postprocess_detections output to ByteTracker Detection objects."""
    result: List[Detection] = []
    for det in raw_detections:
        bbox = det["bbox"]
        result.append(Detection(
            x_center=bbox["x_center"],
            y_center=bbox["y_center"],
            width=bbox["width"],
            height=bbox["height"],
            confidence=det["confidence"],
            class_id=det["class_id"],
        ))
    return result


def _get_existing_videos(downloads_dir: str) -> List[str]:
    """Find already downloaded video files in the downloads directory."""
    video_extensions = {".mp4", ".mkv", ".webm", ".avi", ".mov"}
    dl_path = Path(downloads_dir)
    if not dl_path.exists():
        return []
    files: List[str] = []
    for f in dl_path.iterdir():
        if f.is_file() and f.suffix.lower() in video_extensions:
            files.append(str(f))
    return sorted(files)


def process_video(
    video_path: str,
    interpreter: Any,
    output_dir: str,
    target_fps: float = 10.0,
    max_duration_seconds: int = 60,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Run the full inference + tracking + rally detection pipeline on a video.

    Args:
        video_path: Path to the input video file.
        interpreter: Loaded TFLite interpreter.
        output_dir: Directory to save annotated output and metrics.
        target_fps: FPS to sample frames at for inference.
        max_duration_seconds: Maximum seconds of video to process.
        confidence_threshold: Minimum detection confidence.

    Returns:
        Metrics dict with per-frame and aggregate results.
    """
    video_name = Path(video_path).stem
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    log(TASK, "info", f"Processing video: {Path(video_path).name}",
        target_fps=target_fps, max_duration=max_duration_seconds)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(TASK, "error", f"Cannot open video: {video_path}")
        return {"error": f"Cannot open video: {video_path}"}

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = total_frames / video_fps if video_fps > 0 else 0

    log(TASK, "info", "Video info",
        fps=round(video_fps, 2), total_frames=total_frames,
        resolution=f"{video_width}x{video_height}",
        duration_s=round(video_duration, 2))

    # Calculate frame interval for target FPS extraction
    frame_interval = max(1, int(round(video_fps / target_fps)))
    max_frames_to_process = int(max_duration_seconds * target_fps)

    # Initialize tracker and rally detector
    tracker = ByteTracker()
    rally_events: List[Dict[str, Any]] = []

    def on_rally_end(event: RallyEvent) -> None:
        rally_events.append({
            "rally_number": event.rally_number,
            "start_time": event.start_time,
            "end_time": event.end_time,
            "duration_seconds": event.duration_seconds,
            "end_reason": event.end_reason,
            "net_crossings": event.net_crossings,
            "ball_bounces": event.ball_bounces,
        })

    rally_detector = RallyDetector(
        fps=target_fps,
        on_rally_end=on_rally_end,
    )

    # Setup annotated output video writer
    out_video_path = os.path.join(video_output_dir, f"{video_name}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(
        out_video_path, fourcc, target_fps,
        (video_width, video_height),
    )

    # Per-frame metrics
    frame_metrics: List[Dict[str, Any]] = []
    inference_times_ms: List[float] = []
    total_detections = 0
    frames_processed = 0
    frame_idx = 0

    start_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check duration limit
        current_time_s = frame_idx / video_fps if video_fps > 0 else 0
        if current_time_s > max_duration_seconds:
            break

        # Skip frames to match target FPS
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        if frames_processed >= max_frames_to_process:
            break

        # Preprocess frame for inference
        input_data = preprocess_image(frame, IMG_SIZE)

        # Run inference
        t0 = time.perf_counter()
        raw_output = run_tflite_inference(interpreter, input_data)
        inference_ms = (time.perf_counter() - t0) * 1000
        inference_times_ms.append(inference_ms)

        # Postprocess detections
        raw_dets = postprocess_detections(raw_output, confidence_threshold)
        det_objects = _detections_to_tracker_format(raw_dets)
        total_detections += len(det_objects)

        # Update tracker
        active_tracks = tracker.update(det_objects)

        # Update rally detector with ball tracks
        ball_tracks = [t for t in active_tracks if t.class_id == 0]
        rally_detector.update(ball_tracks)

        # Annotate frame using existing visualization module
        annotated = annotate_frame(
            frame=frame,
            tracks=active_tracks,
            rally_detector=rally_detector,
            show_trajectories=True,
            show_court=True,
            fps=target_fps,
            inference_ms=inference_ms,
        )

        out_writer.write(annotated)

        # Collect per-frame metrics
        frame_metric: Dict[str, Any] = {
            "frame_index": frame_idx,
            "time_s": round(current_time_s, 3),
            "detections": len(raw_dets),
            "active_tracks": len(active_tracks),
            "ball_tracks": len(ball_tracks),
            "inference_ms": round(inference_ms, 2),
            "rally_state": rally_detector.state.value,
            "detection_details": [
                {
                    "class": d["class_name"],
                    "confidence": d["confidence"],
                }
                for d in raw_dets[:10]
            ],
        }
        frame_metrics.append(frame_metric)
        frames_processed += 1
        frame_idx += 1

        # Progress logging every 50 frames
        if frames_processed % 50 == 0:
            log(TASK, "info", f"Progress: {frames_processed} frames processed",
                current_time_s=round(current_time_s, 1),
                avg_inference_ms=round(np.mean(inference_times_ms), 2))

    cap.release()
    out_writer.release()

    total_elapsed_ms = int((time.perf_counter() - start_time) * 1000)

    # Compute aggregate metrics
    avg_inference_ms = float(np.mean(inference_times_ms)) if inference_times_ms else 0.0
    median_inference_ms = float(np.median(inference_times_ms)) if inference_times_ms else 0.0
    min_inference_ms = float(min(inference_times_ms)) if inference_times_ms else 0.0
    max_inference_ms = float(max(inference_times_ms)) if inference_times_ms else 0.0

    avg_confidences: List[float] = []
    for fm in frame_metrics:
        for d in fm.get("detection_details", []):
            avg_confidences.append(d["confidence"])
    avg_confidence = float(np.mean(avg_confidences)) if avg_confidences else 0.0

    frames_with_detections = sum(1 for fm in frame_metrics if fm["detections"] > 0)
    detection_rate = (
        frames_with_detections / frames_processed if frames_processed > 0 else 0.0
    )

    metrics: Dict[str, Any] = {
        "video": {
            "path": video_path,
            "name": Path(video_path).name,
            "fps": round(video_fps, 2),
            "resolution": f"{video_width}x{video_height}",
            "total_duration_s": round(video_duration, 2),
            "processed_duration_s": round(
                max_duration_seconds
                if video_duration > max_duration_seconds
                else video_duration,
                2,
            ),
        },
        "inference": {
            "model_input_size": IMG_SIZE,
            "confidence_threshold": confidence_threshold,
            "target_fps": target_fps,
            "frames_processed": frames_processed,
            "total_detections": total_detections,
            "avg_detections_per_frame": round(
                total_detections / frames_processed if frames_processed > 0 else 0.0,
                2,
            ),
            "frames_with_detections": frames_with_detections,
            "detection_rate": round(detection_rate, 4),
            "avg_confidence": round(avg_confidence, 4),
        },
        "latency": {
            "avg_ms": round(avg_inference_ms, 2),
            "median_ms": round(median_inference_ms, 2),
            "min_ms": round(min_inference_ms, 2),
            "max_ms": round(max_inference_ms, 2),
            "target_met_50ms": median_inference_ms < 50.0,
        },
        "rally_detection": {
            "total_rallies_detected": len(rally_events),
            "events": rally_events,
        },
        "output": {
            "annotated_video": out_video_path,
            "output_dir": video_output_dir,
        },
        "processing": {
            "total_elapsed_ms": total_elapsed_ms,
        },
        "frame_metrics": frame_metrics,
    }

    # Save metrics JSON
    metrics_path = os.path.join(video_output_dir, f"{video_name}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    log(TASK, "ok", f"Video processing complete: {Path(video_path).name}",
        frames_processed=frames_processed,
        total_detections=total_detections,
        avg_inference_ms=round(avg_inference_ms, 2),
        rallies_detected=len(rally_events),
        annotated_video=out_video_path,
        metrics_file=metrics_path,
        ms=total_elapsed_ms)

    return metrics


def generate_summary_report(
    all_metrics: List[Dict[str, Any]],
    output_dir: str,
) -> str:
    """
    Generate a combined summary report across all processed videos.

    Args:
        all_metrics: List of per-video metrics dicts.
        output_dir: Directory to save the summary.

    Returns:
        Path to the summary JSON file.
    """
    total_frames = sum(m.get("inference", {}).get("frames_processed", 0) for m in all_metrics)
    total_detections = sum(m.get("inference", {}).get("total_detections", 0) for m in all_metrics)
    total_rallies = sum(
        m.get("rally_detection", {}).get("total_rallies_detected", 0) for m in all_metrics
    )

    all_latencies: List[float] = []
    all_confidences: List[float] = []
    for m in all_metrics:
        latency = m.get("latency", {})
        if latency.get("avg_ms"):
            all_latencies.append(latency["avg_ms"])
        inference = m.get("inference", {})
        if inference.get("avg_confidence"):
            all_confidences.append(inference["avg_confidence"])

    summary: Dict[str, Any] = {
        "summary": {
            "videos_processed": len(all_metrics),
            "total_frames_processed": total_frames,
            "total_detections": total_detections,
            "total_rallies_detected": total_rallies,
            "avg_inference_latency_ms": round(
                float(np.mean(all_latencies)) if all_latencies else 0.0, 2
            ),
            "avg_confidence": round(
                float(np.mean(all_confidences)) if all_confidences else 0.0, 4
            ),
            "avg_detections_per_frame": round(
                total_detections / total_frames if total_frames > 0 else 0.0, 2
            ),
        },
        "per_video": [
            {
                "name": m.get("video", {}).get("name", "unknown"),
                "frames_processed": m.get("inference", {}).get("frames_processed", 0),
                "total_detections": m.get("inference", {}).get("total_detections", 0),
                "detection_rate": m.get("inference", {}).get("detection_rate", 0),
                "avg_inference_ms": m.get("latency", {}).get("avg_ms", 0),
                "rallies_detected": m.get("rally_detection", {}).get(
                    "total_rallies_detected", 0
                ),
            }
            for m in all_metrics
        ],
    }

    summary_path = os.path.join(output_dir, "summary_report.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log(TASK, "ok", "Summary report generated",
        path=summary_path,
        videos=len(all_metrics),
        total_frames=total_frames,
        total_rallies=total_rallies)

    return summary_path


def main() -> None:
    """Main entry point for the YouTube testing utility."""
    args = parse_args()

    output_dir = args.output_dir
    downloads_dir = os.path.join(output_dir, "downloads")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(downloads_dir, exist_ok=True)

    log(TASK, "start", "YouTube video testing utility started",
        model=args.model, output_dir=output_dir)

    # ------------------------------------------------------------------
    # Step 1: Gather video file paths (download or reuse existing)
    # ------------------------------------------------------------------
    video_paths: List[str] = []

    if args.skip_download:
        log(TASK, "info", "Skipping download, reusing existing videos")
        video_paths = _get_existing_videos(downloads_dir)
        if not video_paths:
            log(TASK, "error", f"No existing videos found in {downloads_dir}")
            sys.exit(1)
        log(TASK, "info", f"Found {len(video_paths)} existing video(s)")

    elif args.url:
        # Download a specific video
        path = download_video(
            url=args.url,
            output_dir=downloads_dir,
            max_duration=args.duration,
            resolution=args.resolution,
        )
        if path:
            video_paths.append(path)
        else:
            log(TASK, "error", "Failed to download video")
            sys.exit(1)

    else:
        # List and download videos from channel
        videos = list_channel_videos(args.channel, args.max_videos)
        if not videos:
            log(TASK, "error", "No videos found on channel")
            sys.exit(1)

        for video_info in videos:
            path = download_video(
                url=video_info["url"],
                output_dir=downloads_dir,
                max_duration=args.duration,
                resolution=args.resolution,
            )
            if path:
                video_paths.append(path)
            else:
                log(TASK, "warning",
                    f"Failed to download: {video_info.get('title', video_info['url'])}")

    if not video_paths:
        log(TASK, "error", "No videos available for processing")
        sys.exit(1)

    log(TASK, "info", f"Videos to process: {len(video_paths)}")

    # ------------------------------------------------------------------
    # Step 2: Load the TFLite model
    # ------------------------------------------------------------------
    model_path = args.model
    if not Path(model_path).exists():
        log(TASK, "error", f"Model file not found: {model_path}")
        sys.exit(1)

    log(TASK, "info", "Loading TFLite model", model=model_path)
    interpreter = load_tflite_model(model_path)
    log(TASK, "ok", "Model loaded successfully")

    # ------------------------------------------------------------------
    # Step 3: Process each video through the full pipeline
    # ------------------------------------------------------------------
    all_metrics: List[Dict[str, Any]] = []

    for i, video_path in enumerate(video_paths):
        log(TASK, "info", f"Processing video {i + 1}/{len(video_paths)}",
            video=Path(video_path).name)

        metrics = process_video(
            video_path=video_path,
            interpreter=interpreter,
            output_dir=output_dir,
            target_fps=args.fps,
            max_duration_seconds=args.duration,
            confidence_threshold=args.confidence,
        )

        if "error" not in metrics:
            all_metrics.append(metrics)

    # ------------------------------------------------------------------
    # Step 4: Generate summary report
    # ------------------------------------------------------------------
    if all_metrics:
        summary_path = generate_summary_report(all_metrics, output_dir)
        log(TASK, "ok", "All processing complete",
            videos_processed=len(all_metrics),
            summary=summary_path)

        # Print human-readable summary
        print("\n" + "=" * 70)
        print("  YOUTUBE VIDEO TEST RESULTS")
        print("=" * 70)
        for m in all_metrics:
            v = m.get("video", {})
            inf = m.get("inference", {})
            lat = m.get("latency", {})
            rally = m.get("rally_detection", {})
            print(f"\n  Video: {v.get('name', '?')}")
            print(f"    Resolution:        {v.get('resolution', '?')}")
            print(f"    Frames processed:  {inf.get('frames_processed', 0)}")
            print(f"    Total detections:  {inf.get('total_detections', 0)}")
            print(f"    Detection rate:    {inf.get('detection_rate', 0):.1%}")
            print(f"    Avg confidence:    {inf.get('avg_confidence', 0):.4f}")
            print(f"    Avg inference:     {lat.get('avg_ms', 0):.2f}ms")
            print(f"    Median inference:  {lat.get('median_ms', 0):.2f}ms")
            print(f"    Target (<50ms):    {'YES' if lat.get('target_met_50ms') else 'NO'}")
            print(f"    Rallies detected:  {rally.get('total_rallies_detected', 0)}")
            out = m.get("output", {})
            print(f"    Annotated video:   {out.get('annotated_video', '?')}")
        print("\n" + "=" * 70)
        print(f"  Summary report: {summary_path}")
        print("=" * 70 + "\n")
    else:
        log(TASK, "error", "No videos were successfully processed")
        sys.exit(1)


if __name__ == "__main__":
    main()
