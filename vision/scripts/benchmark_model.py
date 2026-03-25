"""
Beach Tennis Recorder - Model Performance Benchmark
AGENT-02 | Sprint 2

Benchmarks TFLite model inference latency, throughput, and memory usage.
Target: < 50ms/frame for mobile deployment.

Usage:
    python -m vision.scripts.benchmark_model --model models/best.tflite
    python scripts/benchmark_model.py --model models/best.tflite --iterations 200
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

AGENT = "02"
TASK = "benchmark"
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
    parser = argparse.ArgumentParser(
        description="Benchmark TFLite model inference performance"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(MODELS_DIR / "best.tflite"),
        help="Path to TFLite model",
    )
    parser.add_argument(
        "--model-fp32",
        type=str,
        default="",
        help="Path to FP32 TFLite model for comparison (optional)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of inference iterations (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_DIR / "benchmark_results"),
        help="Directory to save benchmark report",
    )
    return parser.parse_args()


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def benchmark_tflite(
    model_path: str,
    iterations: int,
    warmup: int,
    img_size: int,
    label: str = "model",
) -> Dict[str, Any]:
    """
    Benchmark a single TFLite model.

    Returns a dict with latency stats, throughput, and memory info.
    """
    try:
        import tensorflow as tf
    except ImportError:
        log(TASK, "error", "tensorflow not installed")
        sys.exit(1)

    if not Path(model_path).exists():
        log(TASK, "error", f"Model not found: {model_path}")
        return {"error": f"Model not found: {model_path}"}

    model_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
    log(TASK, "info", f"Loading {label}: {model_path}",
        size_mb=round(model_size_mb, 2))

    mem_before = get_memory_usage_mb()

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = tuple(input_details[0]["shape"])
    input_dtype = str(input_details[0]["dtype"])
    output_shape = tuple(output_details[0]["shape"])

    mem_after_load = get_memory_usage_mb()
    mem_model_mb = mem_after_load - mem_before

    log(TASK, "info", f"{label} loaded",
        input_shape=str(input_shape),
        output_shape=str(output_shape),
        input_dtype=input_dtype,
        memory_mb=round(mem_model_mb, 2))

    # Create input tensor
    dummy_input = np.random.rand(*input_shape).astype(np.float32)

    # Warmup
    log(TASK, "info", f"Running {warmup} warmup iterations...")
    for _ in range(warmup):
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()

    # Benchmark
    log(TASK, "info", f"Running {iterations} benchmark iterations...")
    times_ms: List[float] = []
    for i in range(iterations):
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times_ms.append(elapsed_ms)

    mem_after_bench = get_memory_usage_mb()

    arr = np.array(times_ms)
    throughput_fps = 1000.0 / float(arr.mean()) if arr.mean() > 0 else 0.0

    result: Dict[str, Any] = {
        "label": label,
        "model_path": model_path,
        "model_size_mb": round(model_size_mb, 2),
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
        "input_dtype": input_dtype,
        "iterations": iterations,
        "warmup": warmup,
        "latency": {
            "avg_ms": round(float(arr.mean()), 3),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p75_ms": round(float(np.percentile(arr, 75)), 3),
            "p90_ms": round(float(np.percentile(arr, 90)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3),
            "min_ms": round(float(arr.min()), 3),
            "max_ms": round(float(arr.max()), 3),
            "std_ms": round(float(arr.std()), 3),
        },
        "throughput_fps": round(throughput_fps, 2),
        "memory": {
            "model_load_mb": round(mem_model_mb, 2),
            "peak_mb": round(mem_after_bench - mem_before, 2),
        },
        "target_50ms_met": bool(float(np.percentile(arr, 50)) < 50.0),
    }

    log(TASK, "ok", f"{label} benchmark complete",
        avg_ms=result["latency"]["avg_ms"],
        p50_ms=result["latency"]["p50_ms"],
        p95_ms=result["latency"]["p95_ms"],
        throughput_fps=result["throughput_fps"],
        target_met=result["target_50ms_met"])

    return result


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log(TASK, "start", "Model performance benchmark",
        model=args.model, iterations=args.iterations)
    overall_start = time.perf_counter()

    results: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": {
            "iterations": args.iterations,
            "warmup": args.warmup,
            "img_size": args.img_size,
        },
        "benchmarks": [],
    }

    # Benchmark primary model
    primary = benchmark_tflite(
        args.model, args.iterations, args.warmup,
        args.img_size, label="primary",
    )
    results["benchmarks"].append(primary)

    # Benchmark FP32 model if provided
    if args.model_fp32 and Path(args.model_fp32).exists():
        fp32 = benchmark_tflite(
            args.model_fp32, args.iterations, args.warmup,
            args.img_size, label="fp32",
        )
        results["benchmarks"].append(fp32)

        # Compute speedup
        if "error" not in primary and "error" not in fp32:
            speedup = (
                fp32["latency"]["avg_ms"] / primary["latency"]["avg_ms"]
                if primary["latency"]["avg_ms"] > 0 else 0
            )
            results["comparison"] = {
                "int8_vs_fp32_speedup": round(speedup, 2),
                "int8_avg_ms": primary["latency"]["avg_ms"],
                "fp32_avg_ms": fp32["latency"]["avg_ms"],
                "int8_size_mb": primary["model_size_mb"],
                "fp32_size_mb": fp32["model_size_mb"],
                "size_reduction": round(
                    1 - primary["model_size_mb"] / fp32["model_size_mb"], 2
                ) if fp32["model_size_mb"] > 0 else 0,
            }

    total_ms = int((time.perf_counter() - overall_start) * 1000)
    results["total_elapsed_ms"] = total_ms

    # Save report
    report_path = output_dir / "benchmark_report.json"
    with open(str(report_path), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log(TASK, "ok", "Benchmark complete", ms=total_ms, report=str(report_path))

    # Print summary
    print("\n" + "=" * 65)
    print("  MODEL PERFORMANCE BENCHMARK")
    print("=" * 65)

    for bench in results["benchmarks"]:
        if "error" in bench:
            print(f"\n  {bench['label']}: ERROR - {bench['error']}")
            continue
        lat = bench["latency"]
        print(f"\n  [{bench['label'].upper()}] {bench['model_path']}")
        print(f"    Model size:     {bench['model_size_mb']:.2f} MB")
        print(f"    Input:          {bench['input_shape']} ({bench['input_dtype']})")
        print(f"    Iterations:     {bench['iterations']} (+{bench['warmup']} warmup)")
        print(f"    Avg latency:    {lat['avg_ms']:.3f}ms")
        print(f"    P50 latency:    {lat['p50_ms']:.3f}ms")
        print(f"    P95 latency:    {lat['p95_ms']:.3f}ms")
        print(f"    P99 latency:    {lat['p99_ms']:.3f}ms")
        print(f"    Min/Max:        {lat['min_ms']:.3f}ms / {lat['max_ms']:.3f}ms")
        print(f"    Throughput:     {bench['throughput_fps']:.2f} FPS")
        print(f"    Memory (model): {bench['memory']['model_load_mb']:.2f} MB")
        print(f"    Target (<50ms): {'PASS' if bench['target_50ms_met'] else 'FAIL'}")

    if "comparison" in results:
        comp = results["comparison"]
        print(f"\n  COMPARISON (INT8 vs FP32):")
        print(f"    Speedup:        {comp['int8_vs_fp32_speedup']:.2f}x")
        print(f"    Size reduction: {comp['size_reduction']:.0%}")

    print(f"\n  Report: {report_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
