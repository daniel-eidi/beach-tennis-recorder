"""
Beach Tennis Recorder - Test Gesture Detection on Video
AGENT-02 | Gesture Detection Feature

Runs YOLO + MediaPipe Pose on each frame of a video, looking for
palm-to-racket clap gestures. Generates an annotated video with
skeleton overlay and gesture state, and reports detected gesture events.

Usage:
    python scripts/test_gesture_video.py --video path/to/video.mp4 --model path/to/model.pt
    python scripts/test_gesture_video.py --video test_results/downloads/clip.mp4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure project imports work
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR.parent))

from vision.tracking.byte_tracker import Detection
from vision.tracking.gesture_detector import (
    GestureDetector,
    GestureEvent,
    GestureState,
    RACKET_CLASS_IDS,
)
from vision.tracking.pose_estimator import PoseEstimator
from vision.utils.visualization import draw_gesture_state, draw_skeleton

AGENT = "02"
TASK = "gesture-video-test"

# COCO class IDs relevant for gesture detection
COCO_PERSON = 0
COCO_SPORTS_BALL = 32
COCO_TENNIS_RACKET = 38

COCO_NAMES = {
    0: "person",
    32: "sports_ball",
    38: "tennis_racket",
}


def log_json(task: str, status: str, message: str = "", **kwargs):
    entry = {
        "agent": AGENT,
        "task": task,
        "status": status,
        "message": message,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    entry.update(kwargs)
    print(json.dumps(entry), flush=True)


def yolo_boxes_to_detections(results, classes=None):
    """Convert YOLO results to our Detection dataclass list."""
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            if classes is not None and cls_id not in classes:
                continue

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            detections.append(Detection(cx, cy, w, h, conf, cls_id))
    return detections


def draw_yolo_detections(frame, detections):
    """Draw YOLO detection boxes on frame."""
    for det in detections:
        x1 = int(det.x_center - det.width / 2)
        y1 = int(det.y_center - det.height / 2)
        x2 = int(det.x_center + det.width / 2)
        y2 = int(det.y_center + det.height / 2)

        if det.class_id == COCO_TENNIS_RACKET:
            color = (255, 128, 0)  # Blue-ish
        elif det.class_id == COCO_PERSON:
            color = (0, 255, 0)    # Green
        elif det.class_id == COCO_SPORTS_BALL:
            color = (0, 255, 255)  # Yellow
        else:
            color = (255, 255, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{COCO_NAMES.get(det.class_id, str(det.class_id))} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Test gesture detection (palm-to-racket clap) on video"
    )
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--model", type=str,
                        default=str(PROJECT_DIR / "models" / "yolov8n.pt"),
                        help="Path to YOLO model")
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_DIR / "test_results" / "gesture"),
                        help="Output directory")
    parser.add_argument("--duration", type=int, default=120,
                        help="Max seconds to process")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="YOLO confidence threshold")
    parser.add_argument("--cooldown", type=float, default=5.0,
                        help="Gesture cooldown seconds")
    parser.add_argument("--min-frames", type=int, default=3,
                        help="Minimum frames for gesture state transitions")
    parser.add_argument("--show", action="store_true",
                        help="Show video in OpenCV window during processing")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Validate input
    if not os.path.isfile(args.video):
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)

    # Load YOLO model
    print("=" * 60)
    print("  GESTURE DETECTION TEST")
    print("=" * 60)
    print(f"\nVideo:  {args.video}")
    print(f"Model:  {args.model}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(args.model)
    print("YOLO model loaded.")

    # Initialize Pose Estimator
    try:
        pose_estimator = PoseEstimator(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        )
        print("MediaPipe Pose initialized.")
    except ImportError:
        print("ERROR: mediapipe not installed. Run: pip install mediapipe")
        sys.exit(1)

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video:  {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    # Initialize gesture detector
    gesture_detector = GestureDetector(
        fps=fps,
        cooldown_seconds=args.cooldown,
        min_frames=args.min_frames,
    )

    # Output annotated video
    video_name = Path(args.video).stem
    out_path = os.path.join(args.output_dir, f"{video_name}_gesture_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # Detection classes: person + racket (+ ball for context)
    detect_classes = [COCO_PERSON, COCO_SPORTS_BALL, COCO_TENNIS_RACKET]

    print(f"\nProcessing (conf={args.confidence}, cooldown={args.cooldown}s)...")
    print(f"Max duration: {args.duration}s\n")

    frame_idx = 0
    max_frames = int(args.duration * fps) if args.duration else total_frames
    gesture_events: list = []
    inference_times: list = []
    pose_times: list = []
    state_history: list = []

    start_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break

        # --- YOLO inference ---
        t0 = time.perf_counter()
        results = model(frame, conf=args.confidence, classes=detect_classes, verbose=False)
        yolo_ms = (time.perf_counter() - t0) * 1000
        inference_times.append(yolo_ms)

        detections = yolo_boxes_to_detections(results, classes=set(detect_classes))

        # --- Pose estimation ---
        t1 = time.perf_counter()
        pose_results = pose_estimator.estimate(frame)
        pose_ms = (time.perf_counter() - t1) * 1000
        pose_times.append(pose_ms)

        # --- Gesture detection ---
        event = gesture_detector.update(
            frame, detections,
            pose_results=pose_results,
        )

        if event is not None:
            gesture_events.append({
                "frame": event.frame_index,
                "timestamp_s": round(event.timestamp, 2),
                "confidence": round(event.confidence, 3),
                "player_bbox": [round(v, 1) for v in event.player_bbox],
            })
            print(f"  ** GESTURE DETECTED at frame {event.frame_index} "
                  f"(t={event.timestamp:.2f}s, conf={event.confidence:.3f})")

        state_history.append(gesture_detector.state.value)

        # --- Annotate frame ---
        annotated = frame.copy()
        annotated = draw_yolo_detections(annotated, detections)

        # Draw skeleton
        if pose_results:
            from vision.utils.visualization import draw_skeleton as _draw_skel
            annotated = _draw_skel(annotated, pose_results[0])

        # Draw gesture state
        annotated = draw_gesture_state(annotated, gesture_detector,
                                       pose_result=pose_results[0] if pose_results else None)

        # Info overlay
        info_text = (f"F:{frame_idx} | YOLO:{yolo_ms:.0f}ms Pose:{pose_ms:.0f}ms | "
                     f"State:{gesture_detector.state.value}")
        cv2.putText(annotated, info_text, (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

        rackets = [d for d in detections if d.class_id == COCO_TENNIS_RACKET]
        persons = [d for d in detections if d.class_id == COCO_PERSON]
        det_text = f"Persons:{len(persons)} Rackets:{len(rackets)} Poses:{len(pose_results)}"
        cv2.putText(annotated, det_text, (10, height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

        out.write(annotated)

        if args.show:
            cv2.imshow("Gesture Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

        if frame_idx % 100 == 0:
            elapsed = time.perf_counter() - start_time
            print(f"  Frame {frame_idx}/{max_frames} | "
                  f"YOLO avg: {np.mean(inference_times):.1f}ms | "
                  f"Pose avg: {np.mean(pose_times):.1f}ms | "
                  f"Gestures: {len(gesture_events)} | "
                  f"Elapsed: {elapsed:.1f}s")

    cap.release()
    out.release()
    if args.show:
        cv2.destroyAllWindows()

    pose_estimator.close()

    total_elapsed = time.perf_counter() - start_time

    # --- Report ---
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Frames processed:      {frame_idx}")
    print(f"  Gesture events:        {len(gesture_events)}")

    if gesture_events:
        print(f"\n  Gesture timestamps:")
        for ge in gesture_events:
            print(f"    Frame {ge['frame']} | t={ge['timestamp_s']}s | "
                  f"conf={ge['confidence']}")

    print(f"\n  YOLO latency (avg):    {np.mean(inference_times):.1f}ms")
    print(f"  Pose latency (avg):    {np.mean(pose_times):.1f}ms")
    print(f"  Combined avg:          {np.mean(inference_times) + np.mean(pose_times):.1f}ms")
    print(f"  Total elapsed:         {total_elapsed:.1f}s")
    print(f"  Annotated video:       {out_path}")

    # State distribution
    from collections import Counter
    state_counts = Counter(state_history)
    print(f"\n  State distribution:")
    for state_name, count in state_counts.most_common():
        pct = count / len(state_history) * 100
        print(f"    {state_name:30s} {count:6d} ({pct:.1f}%)")

    print("=" * 60)

    # Save metrics
    metrics = {
        "video": args.video,
        "model": args.model,
        "frames_processed": frame_idx,
        "gesture_events": gesture_events,
        "gesture_count": len(gesture_events),
        "yolo_latency_avg_ms": round(np.mean(inference_times), 2),
        "pose_latency_avg_ms": round(np.mean(pose_times), 2),
        "combined_latency_avg_ms": round(
            np.mean(inference_times) + np.mean(pose_times), 2
        ),
        "state_distribution": dict(state_counts),
        "total_elapsed_s": round(total_elapsed, 2),
        "annotated_video": out_path,
        "config": {
            "cooldown_seconds": args.cooldown,
            "min_frames": args.min_frames,
            "confidence": args.confidence,
        },
    }

    metrics_path = os.path.join(args.output_dir, f"{video_name}_gesture_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics: {metrics_path}\n")

    log_json(TASK, "ok", "Gesture video test complete",
             frames=frame_idx, gestures=len(gesture_events),
             ms=round(total_elapsed * 1000))


if __name__ == "__main__":
    main()
