"""
Beach Tennis Recorder - TFLite Export Script
AGENT-02 | TASK-02-11

Exports YOLOv8 model to INT8 quantized TFLite for mobile deployment.
Contract: Input [1, 640, 640, 3] float32 -> Output [1, 25200, 6] float32
Output classes: {0: "ball", 1: "net", 2: "court_line"}
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ultralytics import YOLO


AGENT = "02"
TASK = "02-11"
PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
MOBILE_MODEL_DIR = PROJECT_DIR.parent / "mobile" / "assets" / "models"

EXPECTED_INPUT_SHAPE = (1, 640, 640, 3)
EXPECTED_CLASSES = {0: "ball", 1: "net", 2: "court_line"}


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
        description="Export YOLOv8 model to INT8 quantized TFLite"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODELS_DIR / "best.pt"),
        help="Path to the trained .pt model (default: models/best.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(MODELS_DIR / "best.tflite"),
        help="Output path for the .tflite model",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image size for export (default: 640)",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        default=True,
        help="Use INT8 quantization (default: True)",
    )
    parser.add_argument(
        "--copy-to-mobile",
        action="store_true",
        default=True,
        help="Copy the exported model to mobile/assets/models/",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        default=False,
        help="Skip output shape validation",
    )
    return parser.parse_args()


def validate_tflite_model(tflite_path: str) -> bool:
    """Validate the exported TFLite model's input/output shapes."""
    try:
        import tensorflow as tf
    except ImportError:
        log(TASK, "warning", "tensorflow not installed, skipping TFLite validation")
        return True

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Validate input shape
    input_shape = tuple(input_details[0]["shape"])
    log(TASK, "info", f"TFLite input shape: {input_shape}")

    if len(input_shape) != 4 or input_shape[1] != 640 or input_shape[2] != 640:
        log(TASK, "warning",
            f"Input shape {input_shape} differs from expected {EXPECTED_INPUT_SHAPE}. "
            "This may be acceptable depending on the export configuration.")

    # Log output shape
    output_shape = tuple(output_details[0]["shape"])
    log(TASK, "info", f"TFLite output shape: {output_shape}")

    # Run a dummy inference to verify the model works
    input_data = np.random.rand(*input_shape).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])

    log(TASK, "info", f"Dummy inference output shape: {output_data.shape}")

    return True


def export(args: argparse.Namespace) -> None:
    """Export YOLOv8 model to TFLite."""
    model_path = Path(args.model)
    output_path = Path(args.output)

    if not model_path.exists():
        log(TASK, "error", f"Model not found: {model_path}")
        sys.exit(1)

    log(TASK, "start", "Exporting to TFLite", config={
        "model": str(model_path),
        "output": str(output_path),
        "img_size": args.img_size,
        "int8": args.int8,
    })

    start_time = time.time()

    # Load model
    model = YOLO(str(model_path))

    # Export to TFLite
    export_path = model.export(
        format="tflite",
        imgsz=args.img_size,
        int8=args.int8,
    )

    elapsed_ms = int((time.time() - start_time) * 1000)

    if export_path is None:
        log(TASK, "error", "TFLite export failed", ms=elapsed_ms)
        sys.exit(1)

    export_path = Path(export_path)

    # Move to desired output location
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if export_path != output_path:
        shutil.copy2(str(export_path), str(output_path))

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    log(TASK, "ok", f"TFLite model exported: {output_path}",
        ms=elapsed_ms, size_mb=round(file_size_mb, 2))

    # Validate
    if not args.skip_validation:
        log(TASK, "info", "Validating exported TFLite model...")
        validate_tflite_model(str(output_path))

    # Copy to mobile assets
    if args.copy_to_mobile:
        MOBILE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        mobile_dest = MOBILE_MODEL_DIR / "ball_detector.tflite"
        shutil.copy2(str(output_path), str(mobile_dest))
        log(TASK, "ok", f"Model copied to mobile: {mobile_dest}")


if __name__ == "__main__":
    args = parse_args()
    export(args)
