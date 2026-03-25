"""
Beach Tennis Recorder - Pre-trained Model Setup
AGENT-02 | Sprint 2

Downloads YOLOv8n (nano) base model from ultralytics and exports it to
TFLite and ONNX formats for pipeline validation before a custom-trained
model is available.

The pre-trained model uses COCO classes (80 classes).  "sports ball" is
class 32 which serves as a proxy for "ball" during pipeline validation.

Usage:
    python -m vision.scripts.setup_pretrained --output-dir models/
    python scripts/setup_pretrained.py --output-dir models/
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict

AGENT = "02"
TASK = "setup-pretrained"
PROJECT_DIR = Path(__file__).resolve().parent.parent


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
        description="Download and prepare a pre-trained YOLOv8n model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_DIR / "models"),
        help="Directory to save model files (default: models/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Ultralytics model name to download (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image size for export (default: 640)",
    )
    parser.add_argument(
        "--skip-tflite",
        action="store_true",
        help="Skip TFLite export",
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Skip ONNX export",
    )
    return parser.parse_args()


def download_model(model_name: str, output_dir: Path) -> Path:
    """Download YOLOv8n from ultralytics hub."""
    try:
        from ultralytics import YOLO
    except ImportError:
        log(TASK, "error", "ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    log(TASK, "info", f"Downloading {model_name} from ultralytics...")
    start = time.perf_counter()

    model = YOLO(model_name)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    # Save to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    pt_path = output_dir / model_name
    if not pt_path.exists():
        # ultralytics caches the model; copy it to our output dir
        default_path = Path(model_name)
        if default_path.exists():
            shutil.copy2(str(default_path), str(pt_path))
        else:
            # Model may be cached in ultralytics default location
            # Re-save via export workaround: just save the model info
            model.save(str(pt_path))

    file_size_mb = pt_path.stat().st_size / (1024 * 1024) if pt_path.exists() else 0
    log(TASK, "ok", f"Model downloaded: {pt_path}",
        ms=elapsed_ms, size_mb=round(file_size_mb, 2))

    return pt_path


def export_tflite(model_path: Path, output_dir: Path, img_size: int) -> Path:
    """Export model to TFLite format."""
    from ultralytics import YOLO

    log(TASK, "info", "Exporting to TFLite...", img_size=img_size)
    start = time.perf_counter()

    model = YOLO(str(model_path))
    export_path = model.export(format="tflite", imgsz=img_size)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    if export_path is None:
        log(TASK, "error", "TFLite export failed", ms=elapsed_ms)
        sys.exit(1)

    export_path = Path(export_path)
    dest = output_dir / "yolov8n.tflite"
    if export_path != dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(export_path), str(dest))

    # Also copy as best.tflite for pipeline compatibility
    best_dest = output_dir / "best.tflite"
    shutil.copy2(str(dest), str(best_dest))

    file_size_mb = dest.stat().st_size / (1024 * 1024)
    log(TASK, "ok", f"TFLite model exported: {dest}",
        ms=elapsed_ms, size_mb=round(file_size_mb, 2))
    log(TASK, "info", f"Also copied as {best_dest} for pipeline compatibility")

    return dest


def export_onnx(model_path: Path, output_dir: Path, img_size: int) -> Path:
    """Export model to ONNX format."""
    from ultralytics import YOLO

    log(TASK, "info", "Exporting to ONNX...", img_size=img_size)
    start = time.perf_counter()

    model = YOLO(str(model_path))
    export_path = model.export(format="onnx", imgsz=img_size, simplify=True)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    if export_path is None:
        log(TASK, "error", "ONNX export failed", ms=elapsed_ms)
        sys.exit(1)

    export_path = Path(export_path)
    dest = output_dir / "yolov8n.onnx"
    if export_path != dest:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(export_path), str(dest))

    # Also copy as best.onnx for pipeline compatibility
    best_dest = output_dir / "best.onnx"
    shutil.copy2(str(dest), str(best_dest))

    file_size_mb = dest.stat().st_size / (1024 * 1024)
    log(TASK, "ok", f"ONNX model exported: {dest}",
        ms=elapsed_ms, size_mb=round(file_size_mb, 2))
    log(TASK, "info", f"Also copied as {best_dest} for pipeline compatibility")

    return dest


def print_coco_mapping() -> None:
    """Print relevant COCO class mappings for beach tennis pipeline."""
    relevant = {
        32: "sports ball",
        0: "person",
        38: "tennis racket",
        63: "laptop",  # sometimes confused with net
    }
    log(TASK, "info", "Relevant COCO class mappings for beach tennis",
        mappings=relevant,
        note="Pre-trained model uses COCO classes. "
             "'sports ball' (id=32) is the proxy for 'ball'. "
             "Custom-trained model will use {0: 'ball', 1: 'net', 2: 'court_line'}.")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    log(TASK, "start", "Setting up pre-trained YOLOv8n model",
        output_dir=str(output_dir), model=args.model, img_size=args.img_size)

    overall_start = time.perf_counter()

    # Step 1: Download model
    pt_path = download_model(args.model, output_dir)

    # Step 2: Export to TFLite
    if not args.skip_tflite:
        try:
            export_tflite(pt_path, output_dir, args.img_size)
        except Exception as e:
            log(TASK, "warning", f"TFLite export failed: {e}. "
                "This may require tensorflow to be installed.")

    # Step 3: Export to ONNX
    if not args.skip_onnx:
        try:
            export_onnx(pt_path, output_dir, args.img_size)
        except Exception as e:
            log(TASK, "warning", f"ONNX export failed: {e}. "
                "This may require onnx to be installed.")

    # Step 4: Print class mapping info
    print_coco_mapping()

    total_ms = int((time.perf_counter() - overall_start) * 1000)
    log(TASK, "ok", "Pre-trained model setup complete", ms=total_ms,
        output_dir=str(output_dir))

    # List output files
    print("\n" + "=" * 60)
    print("  PRE-TRAINED MODEL SETUP COMPLETE")
    print("=" * 60)
    for f in sorted(output_dir.glob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name:30s}  {size_mb:7.2f} MB")
    print("=" * 60)
    print("  NOTE: Pre-trained model uses COCO classes.")
    print("  'sports ball' (class 32) is used as proxy for 'ball'.")
    print("  Run 'make validate' to test the full pipeline.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
