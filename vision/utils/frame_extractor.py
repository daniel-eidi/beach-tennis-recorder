"""
Beach Tennis Recorder - Frame Extractor
AGENT-02 | TASK-02-03

Extracts frames from video files at configurable FPS for dataset annotation.
Target: ~3000 images from 30+ min of video at 5fps.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2


AGENT = "02"
TASK = "02-03"


def log(task: str, status: str, message: str = "", **kwargs) -> None:
    """Emit structured JSON log line."""
    entry = {
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
        description="Extract frames from video for dataset annotation"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input video file or directory of videos",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output directory for extracted frames (default: dataset/images/raw/)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Extraction rate in frames per second (default: 5.0)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum number of frames to extract per video (0 = unlimited)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=0,
        help="Resize frames to this width (0 = keep original, height scaled proportionally)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Output image format (default: jpg)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame",
        help="Filename prefix (default: frame)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality (1-100, default: 95)",
    )
    return parser.parse_args()


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    target_fps: float = 5.0,
    max_frames: int = 0,
    resize_width: int = 0,
    img_format: str = "jpg",
    prefix: str = "frame",
    quality: int = 95,
    frame_offset: int = 0,
) -> int:
    """
    Extract frames from a single video file.

    Args:
        video_path: Path to the video file.
        output_dir: Directory to save extracted frames.
        target_fps: Frames per second to extract.
        max_frames: Maximum frames to extract (0 = unlimited).
        resize_width: Target width for resizing (0 = keep original).
        img_format: Output format ('jpg' or 'png').
        prefix: Filename prefix.
        quality: JPEG quality.
        frame_offset: Starting number for frame naming.

    Returns:
        Number of frames extracted.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(TASK, "error", f"Cannot open video: {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / video_fps if video_fps > 0 else 0

    log(TASK, "info", f"Processing: {Path(video_path).name}",
        video_fps=round(video_fps, 2),
        total_frames=total_frames,
        duration_s=round(duration_s, 2),
        target_fps=target_fps)

    # Calculate frame interval
    frame_interval = max(1, int(round(video_fps / target_fps)))

    os.makedirs(output_dir, exist_ok=True)

    extracted = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Resize if requested
            if resize_width > 0:
                h, w = frame.shape[:2]
                scale = resize_width / w
                new_h = int(h * scale)
                frame = cv2.resize(frame, (resize_width, new_h),
                                   interpolation=cv2.INTER_AREA)

            # Save frame
            num = frame_offset + extracted
            filename = f"{prefix}_{num:06d}.{img_format}"
            filepath = os.path.join(output_dir, filename)

            if img_format == "jpg":
                cv2.imwrite(filepath, frame,
                            [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(filepath, frame)

            extracted += 1

            if max_frames > 0 and extracted >= max_frames:
                break

        frame_idx += 1

    cap.release()

    log(TASK, "ok", f"Extracted {extracted} frames from {Path(video_path).name}",
        frames=extracted, output_dir=output_dir)

    return extracted


def get_video_files(input_path: str) -> list:
    """Get list of video files from path (file or directory)."""
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    p = Path(input_path)

    if p.is_file() and p.suffix.lower() in video_extensions:
        return [str(p)]

    if p.is_dir():
        files = []
        for ext in video_extensions:
            files.extend(p.glob(f"*{ext}"))
            files.extend(p.glob(f"*{ext.upper()}"))
        return sorted(str(f) for f in files)

    return []


def main() -> None:
    args = parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    output_dir = args.output or str(project_dir / "dataset" / "images" / "raw")

    video_files = get_video_files(args.input)
    if not video_files:
        log(TASK, "error", f"No video files found at: {args.input}")
        sys.exit(1)

    log(TASK, "start", f"Extracting frames from {len(video_files)} video(s)",
        target_fps=args.fps, output_dir=output_dir)

    start_time = time.time()
    total_extracted = 0

    for video_path in video_files:
        count = extract_frames_from_video(
            video_path=video_path,
            output_dir=output_dir,
            target_fps=args.fps,
            max_frames=args.max_frames,
            resize_width=args.resize,
            img_format=args.format,
            prefix=args.prefix,
            quality=args.quality,
            frame_offset=total_extracted,
        )
        total_extracted += count

    elapsed_ms = int((time.time() - start_time) * 1000)
    log(TASK, "ok", "Frame extraction complete",
        total_frames=total_extracted,
        videos_processed=len(video_files),
        ms=elapsed_ms)

    # Estimate: 30 min video at 30fps = 54000 frames, at 5fps extraction = 9000 frames
    # We target ~3000 for annotation, so user may want to subsample further
    if total_extracted > 5000:
        log(TASK, "info",
            f"Extracted {total_extracted} frames. Consider subsampling to ~3000 for annotation.")


if __name__ == "__main__":
    main()
