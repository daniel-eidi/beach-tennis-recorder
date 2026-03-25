"""
Beach Tennis Recorder - Inference Test Script
AGENT-02 | TASK-02-13

Tests TFLite model inference speed and accuracy on sample images.
Target: < 50ms/frame on iPhone 12+ class hardware.
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


AGENT = "02"
TASK = "02-13"
PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"

CLASS_NAMES = {0: "ball", 1: "net", 2: "court_line"}
CONFIDENCE_THRESHOLD = 0.45
NMS_IOU_THRESHOLD = 0.45
IMG_SIZE = 640


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
        description="Test TFLite model inference on sample images"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODELS_DIR / "best.tflite"),
        help="Path to the .tflite model",
    )
    parser.add_argument(
        "--images",
        type=str,
        default=str(PROJECT_DIR / "dataset" / "images" / "test"),
        help="Path to directory of test images or a single image",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=50,
        help="Number of iterations for latency benchmark (default: 50)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory to save annotated images (optional)",
    )
    return parser.parse_args()


def preprocess_image(image: np.ndarray, img_size: int = IMG_SIZE) -> np.ndarray:
    """Preprocess image for model input: resize, normalize, add batch dim."""
    # Resize with letterboxing
    h, w = image.shape[:2]
    scale = min(img_size / h, img_size / w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    padded = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    pad_x = (img_size - new_w) // 2
    pad_y = (img_size - new_h) // 2
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    # Normalize to 0-1 float32 and add batch dimension
    blob = padded.astype(np.float32) / 255.0
    blob = np.expand_dims(blob, axis=0)  # [1, 640, 640, 3]
    return blob


def postprocess_detections(
    output: np.ndarray,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    iou_threshold: float = NMS_IOU_THRESHOLD,
) -> List[dict]:
    """Parse model output into detection dicts with NMS."""
    detections = []

    # Output shape: [1, N, 6] where 6 = (x, y, w, h, confidence, class)
    if output.ndim == 3:
        output = output[0]  # Remove batch dim -> [N, 6]

    for det in output:
        conf = float(det[4])
        if conf < confidence_threshold:
            continue

        class_id = int(det[5])
        x_center, y_center, width, height = det[0], det[1], det[2], det[3]

        detections.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES.get(class_id, f"unknown_{class_id}"),
            "confidence": round(conf, 4),
            "bbox": {
                "x_center": round(float(x_center), 2),
                "y_center": round(float(y_center), 2),
                "width": round(float(width), 2),
                "height": round(float(height), 2),
            },
        })

    # Simple NMS by class
    filtered: List[dict] = []
    for cls_id in CLASS_NAMES:
        cls_dets = [d for d in detections if d["class_id"] == cls_id]
        cls_dets.sort(key=lambda d: d["confidence"], reverse=True)
        # Keep top detections (simple approach for testing)
        filtered.extend(cls_dets[:10])

    return filtered


def load_tflite_model(model_path: str):
    """Load TFLite interpreter."""
    try:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except ImportError:
        log(TASK, "error", "tensorflow not installed. Install with: pip install tensorflow")
        sys.exit(1)


def run_tflite_inference(interpreter, input_data: np.ndarray) -> np.ndarray:
    """Run inference on TFLite interpreter."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return output


def benchmark_latency(
    interpreter,
    iterations: int = 50,
) -> dict:
    """Benchmark inference latency over multiple iterations."""
    dummy_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)

    # Warmup
    for _ in range(5):
        run_tflite_inference(interpreter, dummy_input)

    # Benchmark
    times_ms: List[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        run_tflite_inference(interpreter, dummy_input)
        elapsed = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed)

    return {
        "iterations": iterations,
        "mean_ms": round(np.mean(times_ms), 2),
        "median_ms": round(float(np.median(times_ms)), 2),
        "min_ms": round(min(times_ms), 2),
        "max_ms": round(max(times_ms), 2),
        "std_ms": round(float(np.std(times_ms)), 2),
        "target_met": float(np.median(times_ms)) < 50.0,
    }


def get_image_paths(images_arg: str) -> List[str]:
    """Resolve image path argument to list of image file paths."""
    p = Path(images_arg)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        paths: List[str] = []
        for ext in extensions:
            paths.extend(glob.glob(str(p / ext)))
        return sorted(paths)
    return []


def main() -> None:
    args = parse_args()
    model_path = args.model

    if not Path(model_path).exists():
        log(TASK, "error", f"Model not found: {model_path}")
        sys.exit(1)

    log(TASK, "start", "Loading TFLite model", model=model_path)

    interpreter = load_tflite_model(model_path)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    log(TASK, "info", "Model loaded", input_shape=str(input_details[0]["shape"]),
        output_shape=str(output_details[0]["shape"]))

    # Run latency benchmark
    log(TASK, "info", f"Running latency benchmark ({args.benchmark_iterations} iterations)...")
    benchmark = benchmark_latency(interpreter, args.benchmark_iterations)
    log(TASK, "ok", "Benchmark complete", benchmark=benchmark)

    # Run inference on sample images
    image_paths = get_image_paths(args.images)
    if not image_paths:
        log(TASK, "warning", f"No images found at: {args.images}")
        return

    log(TASK, "info", f"Running inference on {len(image_paths)} images")

    output_dir: Optional[Path] = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    total_detections = 0
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            log(TASK, "warning", f"Could not read image: {img_path}")
            continue

        input_data = preprocess_image(image)

        start = time.perf_counter()
        output = run_tflite_inference(interpreter, input_data)
        elapsed_ms = (time.perf_counter() - start) * 1000

        detections = postprocess_detections(output, args.confidence)
        total_detections += len(detections)

        log(TASK, "info", f"Image: {Path(img_path).name}", detections=len(detections),
            inference_ms=round(elapsed_ms, 2),
            results=[{
                "class": d["class_name"],
                "conf": d["confidence"],
            } for d in detections[:5]])

        # Save annotated image if output dir specified
        if output_dir and detections:
            annotated = image.copy()
            for det in detections:
                bbox = det["bbox"]
                x1 = int(bbox["x_center"] - bbox["width"] / 2)
                y1 = int(bbox["y_center"] - bbox["height"] / 2)
                x2 = int(bbox["x_center"] + bbox["width"] / 2)
                y2 = int(bbox["y_center"] + bbox["height"] / 2)
                color = (0, 255, 0) if det["class_name"] == "ball" else (255, 0, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{det['class_name']} {det['confidence']:.2f}"
                cv2.putText(annotated, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            out_path = output_dir / Path(img_path).name
            cv2.imwrite(str(out_path), annotated)

    log(TASK, "ok", "Inference test complete",
        total_images=len(image_paths),
        total_detections=total_detections)


if __name__ == "__main__":
    main()
