"""
Beach Tennis Recorder - Dataset Quality Analysis
AGENT-02 | Sprint 2

Analyzes the annotated dataset for quality issues:
- Class distribution balance
- Image quality (resolution, brightness, contrast)
- Annotation quality (overlapping boxes, tiny boxes, missing classes)
- Train/val/test split validation

Usage:
    python -m vision.utils.dataset_analyzer --data-yaml dataset/data.yaml
    python utils/dataset_analyzer.py --data-yaml dataset/data.yaml --output-dir dataset_analysis/
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

AGENT = "02"
TASK = "dataset-analysis"
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
        description="Analyze dataset quality for beach tennis detection"
    )
    parser.add_argument(
        "--data-yaml",
        type=str,
        default=str(PROJECT_DIR / "dataset" / "data.yaml"),
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_DIR / "dataset_analysis"),
        help="Directory to save analysis report",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=500,
        help="Max images to analyze for quality stats (default: 500)",
    )
    return parser.parse_args()


def parse_data_yaml(yaml_path: str) -> Dict[str, Any]:
    """Parse YOLO data.yaml configuration."""
    try:
        import yaml
    except ImportError:
        # Minimal parser for simple YAML
        config: Dict[str, Any] = {}
        names: Dict[int, str] = {}
        with open(yaml_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                if ":" in line and not line.startswith(" "):
                    key, val = line.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    if val:
                        config[key] = val
                elif ":" in line and line.startswith(" "):
                    # Names mapping
                    parts = line.strip().split(":")
                    if len(parts) == 2:
                        try:
                            names[int(parts[0].strip())] = parts[1].strip()
                        except ValueError:
                            pass
        config["names"] = names
        return config

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def find_split_files(
    dataset_dir: Path,
    images_subdir: str,
    labels_subdir: str,
) -> Tuple[List[Path], List[Path]]:
    """Find image and label files for a given split."""
    img_dir = dataset_dir / images_subdir
    lbl_dir = dataset_dir / labels_subdir

    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images: List[Path] = []
    labels: List[Path] = []

    if img_dir.exists():
        images = sorted([
            f for f in img_dir.iterdir()
            if f.is_file() and f.suffix.lower() in img_extensions
        ])

    if lbl_dir.exists():
        labels = sorted([
            f for f in lbl_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".txt"
        ])

    return images, labels


def parse_yolo_label(label_path: Path) -> List[Dict[str, Any]]:
    """Parse a YOLO format label file."""
    annotations: List[Dict[str, Any]] = []
    try:
        with open(str(label_path), "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    annotations.append({
                        "class_id": int(parts[0]),
                        "x_center": float(parts[1]),
                        "y_center": float(parts[2]),
                        "width": float(parts[3]),
                        "height": float(parts[4]),
                    })
    except Exception:
        pass
    return annotations


def analyze_annotations(
    label_files: List[Path],
    class_names: Dict[int, str],
) -> Dict[str, Any]:
    """Analyze annotation quality across all label files."""
    class_counter: Counter = Counter()
    total_boxes = 0
    boxes_per_image: List[int] = []
    tiny_boxes = 0
    large_boxes = 0
    overlapping_pairs = 0
    images_missing_classes: Dict[str, int] = {}
    box_widths: List[float] = []
    box_heights: List[float] = []
    box_areas: List[float] = []
    empty_labels = 0

    expected_classes = set(class_names.keys())

    for lbl_path in label_files:
        annots = parse_yolo_label(lbl_path)
        boxes_per_image.append(len(annots))

        if not annots:
            empty_labels += 1
            continue

        present_classes = set()
        total_boxes += len(annots)

        for ann in annots:
            cls_id = ann["class_id"]
            class_counter[cls_id] += 1
            present_classes.add(cls_id)

            w = ann["width"]
            h = ann["height"]
            area = w * h
            box_widths.append(w)
            box_heights.append(h)
            box_areas.append(area)

            # Tiny box: area < 0.001 (very small in normalized coords)
            if area < 0.001:
                tiny_boxes += 1

            # Large box: area > 0.5 (more than half the image)
            if area > 0.5:
                large_boxes += 1

        # Check for missing classes
        missing = expected_classes - present_classes
        for m in missing:
            name = class_names.get(m, str(m))
            images_missing_classes[name] = images_missing_classes.get(name, 0) + 1

        # Check for overlapping boxes (IoU > 0.8 within same image)
        for i in range(len(annots)):
            for j in range(i + 1, len(annots)):
                if annots[i]["class_id"] == annots[j]["class_id"]:
                    iou = _compute_iou_normalized(annots[i], annots[j])
                    if iou > 0.8:
                        overlapping_pairs += 1

    # Build class distribution
    class_distribution: Dict[str, int] = {}
    for cls_id, name in sorted(class_names.items()):
        class_distribution[name] = class_counter.get(cls_id, 0)

    bpi = np.array(boxes_per_image) if boxes_per_image else np.array([0])
    bw = np.array(box_widths) if box_widths else np.array([0.0])
    bh = np.array(box_heights) if box_heights else np.array([0.0])
    ba = np.array(box_areas) if box_areas else np.array([0.0])

    return {
        "total_annotations": total_boxes,
        "total_label_files": len(label_files),
        "empty_label_files": empty_labels,
        "class_distribution": class_distribution,
        "boxes_per_image": {
            "avg": round(float(bpi.mean()), 2),
            "min": int(bpi.min()),
            "max": int(bpi.max()),
            "std": round(float(bpi.std()), 2),
        },
        "box_size": {
            "avg_width": round(float(bw.mean()), 4),
            "avg_height": round(float(bh.mean()), 4),
            "avg_area": round(float(ba.mean()), 6),
            "min_area": round(float(ba.min()), 6),
            "max_area": round(float(ba.max()), 6),
        },
        "quality_issues": {
            "tiny_boxes_count": tiny_boxes,
            "large_boxes_count": large_boxes,
            "overlapping_pairs": overlapping_pairs,
            "images_missing_classes": images_missing_classes,
        },
    }


def _compute_iou_normalized(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """Compute IoU between two normalized YOLO bounding boxes."""
    ax1 = a["x_center"] - a["width"] / 2
    ay1 = a["y_center"] - a["height"] / 2
    ax2 = a["x_center"] + a["width"] / 2
    ay2 = a["y_center"] + a["height"] / 2

    bx1 = b["x_center"] - b["width"] / 2
    by1 = b["y_center"] - b["height"] / 2
    bx2 = b["x_center"] + b["width"] / 2
    by2 = b["y_center"] + b["height"] / 2

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = a["width"] * a["height"]
    area_b = b["width"] * b["height"]
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def analyze_image_quality(
    image_files: List[Path],
    sample_limit: int = 500,
) -> Dict[str, Any]:
    """Analyze image quality statistics."""
    if not image_files:
        return {"error": "No images to analyze"}

    sample = image_files[:sample_limit]
    widths: List[int] = []
    heights: List[int] = []
    brightnesses: List[float] = []
    contrasts: List[float] = []
    unreadable = 0

    for img_path in sample:
        img = cv2.imread(str(img_path))
        if img is None:
            unreadable += 1
            continue

        h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)

        # Brightness: mean of grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightnesses.append(float(gray.mean()))

        # Contrast: std of grayscale
        contrasts.append(float(gray.std()))

    if not widths:
        return {"error": "No readable images found", "unreadable": unreadable}

    w_arr = np.array(widths)
    h_arr = np.array(heights)
    b_arr = np.array(brightnesses)
    c_arr = np.array(contrasts)

    # Find unique resolutions
    resolutions = Counter(zip(widths, heights))
    top_resolutions = [
        {"width": w, "height": h, "count": c}
        for (w, h), c in resolutions.most_common(5)
    ]

    return {
        "images_analyzed": len(sample),
        "unreadable_images": unreadable,
        "resolution": {
            "avg_width": round(float(w_arr.mean()), 0),
            "avg_height": round(float(h_arr.mean()), 0),
            "min_width": int(w_arr.min()),
            "max_width": int(w_arr.max()),
            "top_resolutions": top_resolutions,
        },
        "brightness": {
            "avg": round(float(b_arr.mean()), 2),
            "min": round(float(b_arr.min()), 2),
            "max": round(float(b_arr.max()), 2),
            "std": round(float(b_arr.std()), 2),
            "dark_images": int(np.sum(b_arr < 50)),
            "bright_images": int(np.sum(b_arr > 200)),
        },
        "contrast": {
            "avg": round(float(c_arr.mean()), 2),
            "min": round(float(c_arr.min()), 2),
            "max": round(float(c_arr.max()), 2),
            "low_contrast_images": int(np.sum(c_arr < 20)),
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = args.data_yaml
    if not Path(yaml_path).exists():
        log(TASK, "error", f"data.yaml not found: {yaml_path}")
        sys.exit(1)

    log(TASK, "start", "Dataset quality analysis", data_yaml=yaml_path)
    overall_start = time.perf_counter()

    config = parse_data_yaml(yaml_path)
    dataset_dir = Path(yaml_path).parent
    class_names: Dict[int, str] = config.get("names", {})
    nc = int(config.get("nc", len(class_names)))

    log(TASK, "info", "Dataset config loaded",
        classes=nc, names=class_names,
        path=str(dataset_dir))

    report: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset_path": str(dataset_dir),
        "classes": {"count": nc, "names": class_names},
        "splits": {},
    }

    # Analyze each split
    split_configs = {
        "train": (config.get("train", "images/train"), "labels/train"),
        "val": (config.get("val", "images/val"), "labels/val"),
        "test": (config.get("test", "images/test"), "labels/test"),
    }

    total_images = 0
    total_labels = 0

    for split_name, (img_sub, lbl_sub) in split_configs.items():
        # Derive label path from image path
        lbl_sub_derived = str(img_sub).replace("images", "labels")

        images, labels = find_split_files(dataset_dir, img_sub, lbl_sub_derived)

        split_report: Dict[str, Any] = {
            "image_count": len(images),
            "label_count": len(labels),
            "images_without_labels": max(0, len(images) - len(labels)),
        }

        total_images += len(images)
        total_labels += len(labels)

        if labels:
            log(TASK, "info", f"Analyzing {split_name} annotations ({len(labels)} files)")
            split_report["annotations"] = analyze_annotations(labels, class_names)

        if images:
            log(TASK, "info", f"Analyzing {split_name} image quality ({min(len(images), args.sample_limit)} samples)")
            split_report["image_quality"] = analyze_image_quality(
                images, args.sample_limit)

        report["splits"][split_name] = split_report

    # Split balance
    if total_images > 0:
        split_pcts: Dict[str, float] = {}
        for s in ["train", "val", "test"]:
            count = report["splits"].get(s, {}).get("image_count", 0)
            split_pcts[s] = round(count / total_images * 100, 1)

        report["split_balance"] = {
            "total_images": total_images,
            "total_labels": total_labels,
            "percentages": split_pcts,
            "target": {"train": 70, "val": 20, "test": 10},
            "balanced": (
                abs(split_pcts.get("train", 0) - 70) < 10 and
                abs(split_pcts.get("val", 0) - 20) < 10
            ),
        }
    else:
        report["split_balance"] = {
            "total_images": 0,
            "total_labels": 0,
            "note": "No images found. Dataset may not be populated yet.",
        }

    # Recommendations
    recommendations: List[str] = []
    if total_images == 0:
        recommendations.append("Dataset is empty. Collect and annotate images (TASK-02-02 to 02-04).")
    elif total_images < 500:
        recommendations.append(f"Only {total_images} images. Target is ~3000 for adequate training.")

    for split_name, split_data in report.get("splits", {}).items():
        annot = split_data.get("annotations", {})
        issues = annot.get("quality_issues", {})
        if issues.get("tiny_boxes_count", 0) > 10:
            recommendations.append(
                f"{split_name}: {issues['tiny_boxes_count']} tiny boxes detected. "
                "Review annotations for very small bounding boxes.")
        if issues.get("overlapping_pairs", 0) > 5:
            recommendations.append(
                f"{split_name}: {issues['overlapping_pairs']} overlapping box pairs. "
                "Check for duplicate annotations.")

    report["recommendations"] = recommendations

    total_ms = int((time.perf_counter() - overall_start) * 1000)
    report["elapsed_ms"] = total_ms

    # Save report
    report_path = output_dir / "dataset_analysis_report.json"
    with open(str(report_path), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log(TASK, "ok", "Dataset analysis complete",
        ms=total_ms,
        total_images=total_images,
        total_labels=total_labels,
        report=str(report_path))

    # Print summary
    print("\n" + "=" * 65)
    print("  DATASET QUALITY ANALYSIS")
    print("=" * 65)
    print(f"  Dataset:    {dataset_dir}")
    print(f"  Classes:    {nc} - {class_names}")
    print(f"  Total imgs: {total_images}")
    print(f"  Total lbls: {total_labels}")
    for s_name, s_data in report.get("splits", {}).items():
        img_c = s_data.get("image_count", 0)
        lbl_c = s_data.get("label_count", 0)
        pct = report.get("split_balance", {}).get("percentages", {}).get(s_name, 0)
        print(f"\n  [{s_name.upper()}] {img_c} images, {lbl_c} labels ({pct}%)")
        annot = s_data.get("annotations", {})
        if annot:
            print(f"    Annotations:  {annot.get('total_annotations', 0)}")
            print(f"    Distribution: {annot.get('class_distribution', {})}")
            issues = annot.get("quality_issues", {})
            if any(v for v in issues.values() if v):
                print(f"    Issues:       {issues}")

    if recommendations:
        print(f"\n  RECOMMENDATIONS:")
        for r in recommendations:
            print(f"    - {r}")

    print(f"\n  Report: {report_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
