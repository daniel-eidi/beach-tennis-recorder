"""
Beach Tennis Recorder - End-to-End Pipeline Validation
AGENT-02 | Sprint 2

Validates the full vision pipeline end-to-end:
  Model load -> Frame extraction -> Inference -> Tracking -> Rally detection

Downloads a short beach tennis clip, runs every stage, and produces a
structured JSON validation report.

Usage:
    python -m vision.scripts.validate_pipeline --model models/best.tflite --output-dir validation_results/
    python scripts/validate_pipeline.py --model models/best.tflite
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

AGENT = "02"
TASK = "validate-pipeline"
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
DEFAULT_CHANNEL = "https://www.youtube.com/@btcanallive"


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
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline validation for the vision module"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODELS_DIR / "best.tflite"),
        help="Path to TFLite model (default: models/best.tflite)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="",
        help="Path to a local video file (skips YouTube download if provided)",
    )
    parser.add_argument(
        "--youtube-url",
        type=str,
        default="",
        help="Specific YouTube URL to download for testing",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=DEFAULT_CHANNEL,
        help=f"YouTube channel to pick a test video from (default: {DEFAULT_CHANNEL})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_DIR / "validation_results"),
        help="Directory to save validation outputs",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=45,
        help="Max seconds of video to process (default: 45)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Target FPS for frame extraction (default: 10)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25, lower for pre-trained COCO model)",
    )
    return parser.parse_args()


def _get_or_download_video(
    video_path: str,
    youtube_url: str,
    channel: str,
    output_dir: str,
    duration: int,
) -> Optional[str]:
    """Get a video for validation: local file, specific URL, or from channel."""
    if video_path and Path(video_path).exists():
        log(TASK, "info", f"Using local video: {video_path}")
        return video_path

    if video_path and not Path(video_path).exists():
        log(TASK, "error", f"Video file not found: {video_path}")
        return None

    # Try YouTube download
    try:
        from vision.utils.youtube_download import download_video, list_channel_videos
    except ImportError:
        try:
            _proj = str(PROJECT_DIR.parent)
            if _proj not in sys.path:
                sys.path.insert(0, _proj)
            from vision.utils.youtube_download import download_video, list_channel_videos
        except ImportError:
            log(TASK, "error",
                "Cannot import youtube_download. Provide a local --video path instead.")
            return None

    downloads_dir = os.path.join(output_dir, "downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    # Check for existing downloads first
    existing = [
        str(f) for f in Path(downloads_dir).iterdir()
        if f.is_file() and f.suffix.lower() in {".mp4", ".mkv", ".webm"}
    ] if Path(downloads_dir).exists() else []

    if existing:
        log(TASK, "info", f"Reusing existing download: {existing[0]}")
        return existing[0]

    url = youtube_url
    if not url:
        log(TASK, "info", f"Listing videos from channel: {channel}")
        videos = list_channel_videos(channel, max_results=3)
        if not videos:
            log(TASK, "error", "No videos found on channel. Provide --video or --youtube-url.")
            return None
        url = videos[0]["url"]
        log(TASK, "info", f"Selected video: {videos[0].get('title', url)}")

    path = download_video(
        url=url,
        output_dir=downloads_dir,
        max_duration=duration,
        resolution="480p",
    )
    if not path:
        log(TASK, "error", "Failed to download video. Provide a local --video path.")
    return path


def _load_tflite(model_path: str) -> Any:
    """Load TFLite interpreter."""
    try:
        import tensorflow as tf
    except ImportError:
        log(TASK, "error",
            "tensorflow not installed. Run: pip install tensorflow")
        sys.exit(1)

    if not Path(model_path).exists():
        log(TASK, "error", f"Model not found: {model_path}")
        sys.exit(1)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    log(TASK, "ok", "TFLite model loaded",
        input_shape=str(input_details[0]["shape"].tolist()),
        output_shape=str(output_details[0]["shape"].tolist()),
        input_dtype=str(input_details[0]["dtype"]),
        model_size_mb=round(
            Path(model_path).stat().st_size / (1024 * 1024), 2))

    return interpreter


def _run_inference(interpreter: Any, input_data: np.ndarray) -> np.ndarray:
    """Run a single TFLite inference."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])


def _preprocess_frame(frame: np.ndarray, img_size: int = 640) -> np.ndarray:
    """Preprocess a frame for model input (resize, pad, normalize)."""
    h, w = frame.shape[:2]
    scale = min(img_size / h, img_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    pad_x = (img_size - new_w) // 2
    pad_y = (img_size - new_h) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    blob = padded.astype(np.float32) / 255.0
    return np.expand_dims(blob, axis=0)


def _parse_detections(
    output: np.ndarray,
    confidence_threshold: float = 0.25,
) -> List[Dict[str, Any]]:
    """Parse raw model output into detection dicts."""
    if output.ndim == 3:
        output = output[0]

    detections: List[Dict[str, Any]] = []
    for det in output:
        # YOLOv8 output may vary; handle both [x,y,w,h,conf,cls] and
        # transposed [cls+4, N] formats
        if len(det) >= 6:
            conf = float(det[4])
            if conf < confidence_threshold:
                continue
            detections.append({
                "x_center": float(det[0]),
                "y_center": float(det[1]),
                "width": float(det[2]),
                "height": float(det[3]),
                "confidence": round(conf, 4),
                "class_id": int(det[5]),
            })

    # Sort by confidence descending
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections[:50]  # Cap at 50 per frame


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log(TASK, "start", "Beginning end-to-end pipeline validation",
        model=args.model, output_dir=str(output_dir))
    overall_start = time.perf_counter()

    # ---- Step 1: Load model ----
    log(TASK, "info", "Step 1/6: Loading TFLite model")
    interpreter = _load_tflite(args.model)
    input_details = interpreter.get_input_details()
    img_size = int(input_details[0]["shape"][1])

    # ---- Step 2: Obtain test video ----
    log(TASK, "info", "Step 2/6: Obtaining test video")
    video_path = _get_or_download_video(
        args.video, args.youtube_url, args.channel,
        str(output_dir), args.duration,
    )
    if not video_path:
        log(TASK, "error", "No video available for validation. Exiting.")
        sys.exit(1)

    # ---- Step 3: Extract frames ----
    log(TASK, "info", "Step 3/6: Extracting frames from video")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(TASK, "error", f"Cannot open video: {video_path}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_interval = max(1, int(round(video_fps / args.fps)))
    max_frames = int(args.duration * args.fps)

    log(TASK, "info", "Video info",
        fps=round(video_fps, 2), total_frames=total_video_frames,
        resolution=f"{video_w}x{video_h}",
        frame_interval=frame_interval, max_frames=max_frames)

    # ---- Step 4: Inference + Tracking + Rally Detection ----
    log(TASK, "info", "Step 4/6: Running inference, tracking, and rally detection")

    # Import tracking modules
    try:
        from vision.tracking.byte_tracker import ByteTracker, Detection
        from vision.tracking.rally_detector import RallyDetector, RallyEvent
    except ImportError:
        _proj = str(PROJECT_DIR.parent)
        if _proj not in sys.path:
            sys.path.insert(0, _proj)
        from vision.tracking.byte_tracker import ByteTracker, Detection
        from vision.tracking.rally_detector import RallyDetector, RallyEvent

    tracker = ByteTracker(high_threshold=0.3, low_threshold=0.1)
    rally_events: List[Dict[str, Any]] = []

    def on_rally_end(event: RallyEvent) -> None:
        rally_events.append({
            "rally_number": event.rally_number,
            "duration_seconds": event.duration_seconds,
            "end_reason": event.end_reason,
            "net_crossings": event.net_crossings,
            "ball_bounces": event.ball_bounces,
        })

    rally_detector = RallyDetector(fps=args.fps, on_rally_end=on_rally_end)

    # Per-frame data collection
    inference_times_ms: List[float] = []
    detections_per_frame: List[int] = []
    class_counts: Dict[int, int] = {}
    tracks_created_total = 0
    track_lengths: List[int] = []

    frames_processed = 0
    frame_idx = 0

    # Optional: set up annotated output video
    annotated_video_path = str(output_dir / "validation_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer: Optional[cv2.VideoWriter] = None

    try:
        from vision.utils.visualization import annotate_frame
        writer = cv2.VideoWriter(
            annotated_video_path, fourcc, args.fps, (video_w, video_h))
        log(TASK, "info", "Annotated video output enabled")
    except ImportError:
        log(TASK, "warning", "visualization module not available, skipping annotated video")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time_s = frame_idx / video_fps if video_fps > 0 else 0
        if current_time_s > args.duration:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        if frames_processed >= max_frames:
            break

        # Preprocess
        input_data = _preprocess_frame(frame, img_size)

        # Inference
        t0 = time.perf_counter()
        raw_output = _run_inference(interpreter, input_data)
        inf_ms = (time.perf_counter() - t0) * 1000
        inference_times_ms.append(inf_ms)

        # Parse detections
        dets = _parse_detections(raw_output, args.confidence)
        detections_per_frame.append(len(dets))

        # Count classes
        for d in dets:
            cls_id = d["class_id"]
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

        # Convert to tracker format
        det_objects = [
            Detection(
                x_center=d["x_center"],
                y_center=d["y_center"],
                width=d["width"],
                height=d["height"],
                confidence=d["confidence"],
                class_id=d["class_id"],
            )
            for d in dets
        ]

        # Update tracker
        active_tracks = tracker.update(det_objects)

        # Update rally detector (using ball tracks - class 0 for custom model,
        # class 32 for COCO "sports ball")
        ball_tracks = [t for t in active_tracks if t.class_id in (0, 32)]
        rally_detector.update(ball_tracks)

        # Write annotated frame
        if writer is not None:
            try:
                annotated = annotate_frame(
                    frame=frame,
                    tracks=active_tracks,
                    rally_detector=rally_detector,
                    inference_ms=inf_ms,
                )
                writer.write(annotated)
            except Exception:
                writer.write(frame)

        frames_processed += 1
        frame_idx += 1

        if frames_processed % 50 == 0:
            avg_ms = float(np.mean(inference_times_ms))
            log(TASK, "info", f"Progress: {frames_processed}/{max_frames} frames",
                avg_inference_ms=round(avg_ms, 2))

    cap.release()
    if writer is not None:
        writer.release()

    # ---- Step 5: Collect tracking stats ----
    log(TASK, "info", "Step 5/6: Collecting tracking statistics")
    all_tracks = tracker.all_tracks
    tracks_created_total = len(all_tracks)
    for t in all_tracks.values():
        track_lengths.append(len(t.detections))

    # ---- Step 6: Generate validation report ----
    log(TASK, "info", "Step 6/6: Generating validation report")

    inf_arr = np.array(inference_times_ms) if inference_times_ms else np.array([0.0])
    det_arr = np.array(detections_per_frame) if detections_per_frame else np.array([0])
    tl_arr = np.array(track_lengths) if track_lengths else np.array([0])

    report: Dict[str, Any] = {
        "validation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": {
            "path": args.model,
            "input_size": img_size,
            "size_mb": round(Path(args.model).stat().st_size / (1024 * 1024), 2),
        },
        "video": {
            "path": video_path,
            "fps": round(video_fps, 2),
            "resolution": f"{video_w}x{video_h}",
            "duration_s": round(total_video_frames / video_fps, 2) if video_fps > 0 else 0,
            "processed_duration_s": args.duration,
        },
        "frames": {
            "total_processed": frames_processed,
            "target_fps": args.fps,
            "confidence_threshold": args.confidence,
        },
        "detections": {
            "total": int(sum(detections_per_frame)),
            "per_frame_avg": round(float(det_arr.mean()), 2),
            "per_frame_min": int(det_arr.min()),
            "per_frame_max": int(det_arr.max()),
            "per_frame_std": round(float(det_arr.std()), 2),
            "frames_with_detections": int(np.sum(det_arr > 0)),
            "detection_rate": round(
                float(np.sum(det_arr > 0) / max(frames_processed, 1)), 4),
        },
        "classes": {
            str(cls_id): count
            for cls_id, count in sorted(class_counts.items())
        },
        "inference_latency": {
            "avg_ms": round(float(inf_arr.mean()), 2),
            "p50_ms": round(float(np.percentile(inf_arr, 50)), 2),
            "p95_ms": round(float(np.percentile(inf_arr, 95)), 2),
            "p99_ms": round(float(np.percentile(inf_arr, 99)), 2),
            "min_ms": round(float(inf_arr.min()), 2),
            "max_ms": round(float(inf_arr.max()), 2),
            "std_ms": round(float(inf_arr.std()), 2),
            "target_50ms_met": bool(float(np.percentile(inf_arr, 50)) < 50.0),
        },
        "tracking": {
            "tracks_created": tracks_created_total,
            "avg_track_length": round(float(tl_arr.mean()), 2),
            "max_track_length": int(tl_arr.max()),
            "min_track_length": int(tl_arr.min()),
        },
        "rally_detection": {
            "total_events": len(rally_events),
            "events": rally_events,
        },
        "outputs": {
            "annotated_video": annotated_video_path if writer is not None else None,
            "report_path": str(output_dir / "validation_report.json"),
        },
    }

    total_ms = int((time.perf_counter() - overall_start) * 1000)
    report["total_elapsed_ms"] = total_ms

    # Save report
    report_path = output_dir / "validation_report.json"
    with open(str(report_path), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log(TASK, "ok", "Pipeline validation complete",
        ms=total_ms,
        frames_processed=frames_processed,
        total_detections=int(sum(detections_per_frame)),
        tracks_created=tracks_created_total,
        rally_events=len(rally_events),
        report=str(report_path))

    # Print human-readable summary
    print("\n" + "=" * 65)
    print("  PIPELINE VALIDATION REPORT")
    print("=" * 65)
    print(f"  Model:                {args.model}")
    print(f"  Video:                {Path(video_path).name}")
    print(f"  Frames processed:     {frames_processed}")
    print(f"  Total detections:     {int(sum(detections_per_frame))}")
    print(f"  Avg detections/frame: {float(det_arr.mean()):.2f}")
    print(f"  Detection rate:       {float(np.sum(det_arr > 0) / max(frames_processed, 1)):.1%}")
    print(f"  Classes detected:     {dict(sorted(class_counts.items()))}")
    print(f"  Inference avg:        {float(inf_arr.mean()):.2f}ms")
    print(f"  Inference p50:        {float(np.percentile(inf_arr, 50)):.2f}ms")
    print(f"  Inference p95:        {float(np.percentile(inf_arr, 95)):.2f}ms")
    print(f"  Inference p99:        {float(np.percentile(inf_arr, 99)):.2f}ms")
    print(f"  Target (<50ms):       {'PASS' if float(np.percentile(inf_arr, 50)) < 50.0 else 'FAIL'}")
    print(f"  Tracks created:       {tracks_created_total}")
    print(f"  Avg track length:     {float(tl_arr.mean()):.1f} frames")
    print(f"  Rally events:         {len(rally_events)}")
    if rally_events:
        for ev in rally_events:
            print(f"    Rally #{ev['rally_number']}: "
                  f"{ev['duration_seconds']}s, reason={ev['end_reason']}")
    print(f"\n  Report:   {report_path}")
    if writer is not None:
        print(f"  Video:    {annotated_video_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
