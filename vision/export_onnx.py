"""
Beach Tennis Recorder - ONNX Export Script
AGENT-02 | TASK-02-12

Exports YOLOv8 model to ONNX format as Android fallback.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from ultralytics import YOLO


AGENT = "02"
TASK = "02-12"
PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"


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
        description="Export YOLOv8 model to ONNX format"
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
        default=str(MODELS_DIR / "best.onnx"),
        help="Output path for the .onnx model",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Image size for export (default: 640)",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="Simplify ONNX graph (default: True)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version (default: 12)",
    )
    return parser.parse_args()


def validate_onnx_model(onnx_path: str) -> bool:
    """Validate the exported ONNX model."""
    try:
        import onnx
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        log(TASK, "info", "ONNX model validation passed")

        # Log input/output info
        for inp in model.graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            log(TASK, "info", f"ONNX input: {inp.name}, shape: {shape}")
        for out in model.graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            log(TASK, "info", f"ONNX output: {out.name}, shape: {shape}")

        return True
    except ImportError:
        log(TASK, "warning", "onnx package not installed, skipping validation")
        return True
    except Exception as e:
        log(TASK, "error", f"ONNX validation failed: {e}")
        return False


def validate_onnx_inference(onnx_path: str, img_size: int = 640) -> bool:
    """Run a dummy inference through the ONNX model."""
    try:
        import onnxruntime as ort
        import numpy as np

        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape

        # Create dummy input
        dummy = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
        outputs = session.run(None, {input_name: dummy})

        log(TASK, "info", f"ONNX inference test passed, output shape: {outputs[0].shape}")
        return True
    except ImportError:
        log(TASK, "warning", "onnxruntime not installed, skipping inference test")
        return True
    except Exception as e:
        log(TASK, "error", f"ONNX inference test failed: {e}")
        return False


def export(args: argparse.Namespace) -> None:
    """Export YOLOv8 model to ONNX."""
    model_path = Path(args.model)
    output_path = Path(args.output)

    if not model_path.exists():
        log(TASK, "error", f"Model not found: {model_path}")
        sys.exit(1)

    log(TASK, "start", "Exporting to ONNX", config={
        "model": str(model_path),
        "output": str(output_path),
        "img_size": args.img_size,
        "simplify": args.simplify,
        "opset": args.opset,
    })

    start_time = time.time()

    model = YOLO(str(model_path))

    export_path = model.export(
        format="onnx",
        imgsz=args.img_size,
        simplify=args.simplify,
        opset=args.opset,
    )

    elapsed_ms = int((time.time() - start_time) * 1000)

    if export_path is None:
        log(TASK, "error", "ONNX export failed", ms=elapsed_ms)
        sys.exit(1)

    export_path = Path(export_path)

    # Move to desired output location
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if export_path != output_path:
        shutil.copy2(str(export_path), str(output_path))

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    log(TASK, "ok", f"ONNX model exported: {output_path}",
        ms=elapsed_ms, size_mb=round(file_size_mb, 2))

    # Validate
    validate_onnx_model(str(output_path))
    validate_onnx_inference(str(output_path), args.img_size)


if __name__ == "__main__":
    args = parse_args()
    export(args)
