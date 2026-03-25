"""
Beach Tennis Recorder - YOLOv8 Training Script
AGENT-02 | TASK-02-06

Fine-tunes YOLOv8n (nano) for beach tennis object detection.
Classes: ball, net, court_line
Target: mAP@0.5 > 0.75
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from ultralytics import YOLO


AGENT = "02"
TASK = "02-06"
PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"


def log(task: str, status: str, message: str = "", **kwargs: Any) -> None:
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
        description="Fine-tune YOLOv8n for beach tennis detection"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_DIR / "dataset" / "data.yaml"),
        help="Path to data.yaml dataset config",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base model to fine-tune (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image size for training (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to train on: '', 'cpu', '0', '0,1' (default: auto)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    return parser.parse_args()


def validate_dataset(config_path: str) -> bool:
    """Check that the dataset config and directories exist."""
    config = Path(config_path)
    if not config.exists():
        log(TASK, "error", f"Dataset config not found: {config_path}")
        return False

    # Check for image directories
    dataset_dir = config.parent
    for split in ["images/train", "images/val"]:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            log(TASK, "warning", f"Split directory missing: {split_dir}. Creating it.")
            split_dir.mkdir(parents=True, exist_ok=True)

    return True


def train(args: argparse.Namespace) -> None:
    """Run YOLOv8 fine-tuning."""
    log(TASK, "start", "Beginning YOLOv8 training", config={
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "device": args.device or "auto",
    })

    if not validate_dataset(args.config):
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Load base model
    model = YOLO(args.model)

    # Train
    results = model.train(
        data=args.config,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        device=args.device if args.device else None,
        project=str(PROJECT_DIR / "runs"),
        name="train",
        exist_ok=True,
        patience=20,
        save=True,
        save_period=10,
        resume=args.resume,
        # Augmentation suited for small-object (ball) detection
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        flipud=0.0,  # No vertical flip for sports
        fliplr=0.5,
    )

    elapsed_ms = int((time.time() - start_time) * 1000)

    # Copy best model to models/
    best_src = PROJECT_DIR / "runs" / "train" / "weights" / "best.pt"
    best_dst = MODELS_DIR / "best.pt"
    if best_src.exists():
        import shutil
        shutil.copy2(str(best_src), str(best_dst))
        log(TASK, "ok", f"Best model saved to {best_dst}", ms=elapsed_ms)
    else:
        log(TASK, "error", "best.pt not found after training", ms=elapsed_ms)
        sys.exit(1)

    # Evaluate and log metrics
    metrics = model.val(data=args.config)
    map50 = float(metrics.box.map50) if hasattr(metrics.box, "map50") else 0.0
    map50_95 = float(metrics.box.map) if hasattr(metrics.box, "map") else 0.0

    log("02-07", "ok", "Validation complete", metrics={
        "mAP50": round(map50, 4),
        "mAP50-95": round(map50_95, 4),
        "target_met": map50 > 0.75,
    }, ms=elapsed_ms)

    if map50 < 0.75:
        log("02-07", "warning",
            f"mAP@0.5 = {map50:.4f} is below target 0.75. "
            "Consider more data or longer training.")


if __name__ == "__main__":
    args = parse_args()
    train(args)
