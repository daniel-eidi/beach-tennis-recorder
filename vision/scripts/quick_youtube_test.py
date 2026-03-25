"""
Quick YouTube test — uses YOLO .pt model directly (no TFLite export needed).
Downloads a short clip from @btcanallive and runs detection + annotated video.

Usage:
    python scripts/quick_youtube_test.py
    python scripts/quick_youtube_test.py --url "https://youtube.com/watch?v=..."
    python scripts/quick_youtube_test.py --duration 30 --confidence 0.3
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project imports work
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR.parent))

from vision.utils.youtube_download import download_video, list_channel_videos

DEFAULT_CHANNEL = "https://www.youtube.com/@btcanallive"


def main():
    parser = argparse.ArgumentParser(description="Quick YouTube test with YOLOv8")
    parser.add_argument("--url", type=str, default=None, help="YouTube video URL")
    parser.add_argument("--channel", type=str, default=DEFAULT_CHANNEL)
    parser.add_argument("--model", type=str, default=str(PROJECT_DIR / "models" / "yolov8n.pt"))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_DIR / "test_results"))
    parser.add_argument("--duration", type=int, default=60, help="Max seconds to process")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--resolution", type=str, default="720p")
    parser.add_argument("--classes", type=int, nargs="+", default=[32, 0, 38],
                        help="COCO class IDs to detect (32=sports ball, 0=person, 38=tennis racket)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    downloads_dir = os.path.join(args.output_dir, "downloads")

    # Step 1: Get video
    print("=" * 60)
    print("  BEACH TENNIS - QUICK YOUTUBE TEST")
    print("=" * 60)

    if args.url:
        video_url = args.url
    else:
        print(f"\nListando vídeos do canal: {args.channel}")
        videos = list_channel_videos(args.channel, max_results=3)
        if not videos:
            print("ERRO: Nenhum vídeo encontrado no canal.")
            sys.exit(1)
        print(f"\nVídeos encontrados:")
        for i, v in enumerate(videos):
            dur = v.get("duration", "?")
            print(f"  [{i+1}] {v['title']} ({dur}s)")
        # Use first video
        video_url = videos[0]["url"]
        print(f"\nUsando primeiro vídeo: {videos[0]['title']}")

    print(f"\nBaixando vídeo ({args.resolution})...")
    video_path = download_video(
        url=video_url,
        output_dir=downloads_dir,
        max_duration=None,  # download full, process only --duration seconds
        resolution=args.resolution,
    )
    if not video_path:
        print("ERRO: Falha ao baixar vídeo.")
        sys.exit(1)
    print(f"Download completo: {video_path}")

    # Step 2: Load YOLO model
    print(f"\nCarregando modelo: {args.model}")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERRO: ultralytics não instalado. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(args.model)
    print("Modelo carregado!")

    # Step 3: Run inference on video
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERRO: Não conseguiu abrir vídeo: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVídeo: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    # Output annotated video
    video_name = Path(video_path).stem
    out_path = os.path.join(args.output_dir, f"{video_name}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    # COCO class names for display
    coco_names = {0: "person", 32: "sports ball", 38: "tennis racket",
                  56: "chair", 60: "dining table", 62: "tv"}

    print(f"\nProcessando com conf={args.confidence}, classes={args.classes}...")
    print(f"Classes: {[coco_names.get(c, f'class_{c}') for c in args.classes]}")

    frame_idx = 0
    total_detections = 0
    inference_times = []
    detections_per_frame = []
    class_counts = {}
    max_frames = int(args.duration * fps) if args.duration else total_frames

    start_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= max_frames:
            break

        # Run YOLO inference
        t0 = time.perf_counter()
        results = model(frame, conf=args.confidence, classes=args.classes, verbose=False)
        inference_ms = (time.perf_counter() - t0) * 1000
        inference_times.append(inference_ms)

        # Draw detections
        annotated_frame = frame.copy()
        frame_dets = 0

        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = coco_names.get(cls_id, f"class_{cls_id}")

                    # Color by class
                    if cls_id == 32:  # ball
                        color = (0, 255, 255)  # yellow
                    elif cls_id == 0:  # person
                        color = (0, 255, 0)  # green
                    elif cls_id == 38:  # racket
                        color = (255, 128, 0)  # blue-ish
                    else:
                        color = (255, 255, 255)

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_name} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    frame_dets += 1
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        # Info overlay
        info_text = f"Frame {frame_idx} | Dets: {frame_dets} | {inference_ms:.0f}ms"
        cv2.putText(annotated_frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        out.write(annotated_frame)
        total_detections += frame_dets
        detections_per_frame.append(frame_dets)
        frame_idx += 1

        # Progress every 100 frames
        if frame_idx % 100 == 0:
            elapsed = time.perf_counter() - start_time
            print(f"  Frame {frame_idx}/{max_frames} | "
                  f"Avg: {np.mean(inference_times):.1f}ms | "
                  f"Dets: {total_detections} | "
                  f"Elapsed: {elapsed:.1f}s")

    cap.release()
    out.release()

    total_elapsed = time.perf_counter() - start_time

    # Metrics
    print("\n" + "=" * 60)
    print("  RESULTADOS")
    print("=" * 60)
    print(f"  Frames processados:   {frame_idx}")
    print(f"  Total detecções:      {total_detections}")
    print(f"  Detecções/frame (avg): {np.mean(detections_per_frame):.2f}")
    print(f"  Frames com detecção:  {sum(1 for d in detections_per_frame if d > 0)} ({sum(1 for d in detections_per_frame if d > 0)/max(frame_idx,1)*100:.1f}%)")
    print(f"\n  Latência inferência:")
    print(f"    Média:   {np.mean(inference_times):.2f}ms")
    print(f"    Mediana: {np.median(inference_times):.2f}ms")
    print(f"    P95:     {np.percentile(inference_times, 95):.2f}ms")
    print(f"    Min:     {np.min(inference_times):.2f}ms")
    print(f"    Max:     {np.max(inference_times):.2f}ms")
    print(f"\n  Detecções por classe:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"    {cls_name}: {count}")
    print(f"\n  Tempo total:          {total_elapsed:.1f}s")
    print(f"  Vídeo anotado:        {out_path}")
    print("=" * 60)

    # Save metrics JSON
    metrics = {
        "video": video_path,
        "frames_processed": frame_idx,
        "total_detections": total_detections,
        "avg_detections_per_frame": round(np.mean(detections_per_frame), 2),
        "detection_rate": round(sum(1 for d in detections_per_frame if d > 0) / max(frame_idx, 1), 4),
        "latency_avg_ms": round(np.mean(inference_times), 2),
        "latency_median_ms": round(np.median(inference_times), 2),
        "latency_p95_ms": round(np.percentile(inference_times, 95), 2),
        "class_counts": class_counts,
        "total_elapsed_s": round(total_elapsed, 2),
        "annotated_video": out_path,
    }
    metrics_path = os.path.join(args.output_dir, f"{video_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Métricas: {metrics_path}\n")


if __name__ == "__main__":
    main()
