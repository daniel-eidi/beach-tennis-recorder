"""
Quick fine-tune pipeline: download Roboflow dataset -> train YOLOv8n -> test.

Usage:
    python scripts/finetune_roboflow.py
    python scripts/finetune_roboflow.py --epochs 50 --batch-size 8
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

VISION_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VISION_DIR.parent))

MODELS_DIR = VISION_DIR / "models"
DATASET_DIR = VISION_DIR / "dataset" / "roboflow"


def log(msg: str) -> None:
    print(f"[FINETUNE] {msg}", flush=True)


def download_dataset(output_dir: Path, api_key: str = "") -> Path:
    """Download Padel-WPT dataset from Roboflow."""
    log("Baixando dataset Padel-WPT-10videos do Roboflow...")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from roboflow import Roboflow

        # Try with API key first, fall back to public access
        if api_key:
            rf = Roboflow(api_key=api_key)
        else:
            rf = Roboflow(api_key="a]")  # placeholder triggers public access

        project = rf.workspace("yolo-data-labeling").project("padel-wpt-10videos")
        version = project.version(1)
        dataset = version.download("yolov8", location=str(output_dir))
        log(f"Dataset baixado em: {output_dir}")
        return output_dir

    except Exception as e:
        log(f"Erro ao baixar via Roboflow SDK: {e}")
        log("Tentando download alternativo via URL pública...")
        return download_dataset_manual(output_dir)


def download_dataset_manual(output_dir: Path) -> Path:
    """Fallback: download dataset via direct URL."""
    import urllib.request
    import zipfile

    url = "https://universe.roboflow.com/ds/padel-wpt-10videos?key=yolov8"
    zip_path = output_dir / "dataset.zip"

    log(f"Baixando de {url}...")
    try:
        urllib.request.urlretrieve(url, str(zip_path))
        with zipfile.ZipFile(str(zip_path), 'r') as z:
            z.extractall(str(output_dir))
        zip_path.unlink()
        log("Download manual completo.")
    except Exception as e:
        log(f"Download manual falhou: {e}")
        log("")
        log("=" * 60)
        log("INSTRUÇÃO MANUAL:")
        log("1. Acesse: https://universe.roboflow.com/yolo-data-labeling/padel-wpt-10videos")
        log("2. Clique em 'Download Dataset' -> formato YOLOv8")
        log(f"3. Extraia o ZIP em: {output_dir}")
        log("4. Rode este script novamente com --skip-download")
        log("=" * 60)
        sys.exit(1)

    return output_dir


def find_data_yaml(dataset_dir: Path) -> Path:
    """Find data.yaml in the dataset directory."""
    # Search recursively
    for yaml_file in dataset_dir.rglob("data.yaml"):
        return yaml_file

    # If not found, create one
    log("data.yaml não encontrado, criando...")
    data_yaml = dataset_dir / "data.yaml"

    # Detect directory structure
    train_imgs = None
    val_imgs = None
    for d in dataset_dir.rglob("train"):
        if (d / "images").exists():
            train_imgs = str(d / "images")
        elif d.name == "train" and any(d.glob("*.jpg")):
            train_imgs = str(d)
    for d in dataset_dir.rglob("valid"):
        if (d / "images").exists():
            val_imgs = str(d / "images")
        elif d.name == "valid" and any(d.glob("*.jpg")):
            val_imgs = str(d)
    if not val_imgs:
        for d in dataset_dir.rglob("val"):
            if (d / "images").exists():
                val_imgs = str(d / "images")

    content = f"""train: {train_imgs or str(dataset_dir / 'train' / 'images')}
val: {val_imgs or str(dataset_dir / 'valid' / 'images')}

nc: 3
names: ['ball', 'net', 'court_line']
"""
    data_yaml.write_text(content)
    log(f"Criado: {data_yaml}")
    return data_yaml


def remap_classes(dataset_dir: Path) -> None:
    """
    Remap Padel-WPT classes to our target classes.
    Padel-WPT: 0=ball, 1=net, 2=player, 3=racket, 4=serve_line
    Our target: 0=ball, 1=net, 2=court_line

    Strategy: keep ball(0), net(1), map serve_line(4)->court_line(2),
    remove player(2) and racket(3).
    """
    log("Remapeando classes para nosso formato (ball, net, court_line)...")

    # Map: old_id -> new_id (None = remove)
    class_map = {
        0: 0,     # ball -> ball
        1: 1,     # net -> net
        2: None,  # player -> remove
        3: None,  # racket -> remove
        4: 2,     # serve_line -> court_line
    }

    label_dirs = list(dataset_dir.rglob("labels"))
    total_files = 0
    total_kept = 0
    total_removed = 0

    for label_dir in label_dirs:
        if not label_dir.is_dir():
            continue
        for label_file in label_dir.glob("*.txt"):
            if label_file.name == "classes.txt":
                continue
            total_files += 1
            new_lines = []
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    old_class = int(parts[0])
                    new_class = class_map.get(old_class)
                    if new_class is not None:
                        parts[0] = str(new_class)
                        new_lines.append(" ".join(parts))
                        total_kept += 1
                    else:
                        total_removed += 1

            with open(label_file, "w") as f:
                f.write("\n".join(new_lines) + "\n" if new_lines else "")

    log(f"Remapeamento: {total_files} arquivos, {total_kept} anotações mantidas, {total_removed} removidas")


def update_data_yaml(data_yaml: Path) -> None:
    """Update data.yaml to reflect our 3 classes."""
    content = data_yaml.read_text()

    # Replace class info
    import re
    content = re.sub(r'nc:\s*\d+', 'nc: 3', content)
    content = re.sub(
        r"names:\s*\[.*?\]",
        "names: ['ball', 'net', 'court_line']",
        content,
        flags=re.DOTALL
    )
    # Also handle dict-style names
    content = re.sub(
        r"names:\s*\{.*?\}",
        "names: ['ball', 'net', 'court_line']",
        content,
        flags=re.DOTALL
    )

    data_yaml.write_text(content)
    log(f"data.yaml atualizado: 3 classes (ball, net, court_line)")


def train_model(data_yaml: Path, epochs: int, batch_size: int, device: str) -> Path:
    """Fine-tune YOLOv8n."""
    from ultralytics import YOLO

    log(f"Iniciando fine-tune: epochs={epochs}, batch={batch_size}, device={device or 'auto'}")

    # Use pretrained model if available, otherwise download
    base_model = MODELS_DIR / "yolov8n.pt"
    if not base_model.exists():
        base_model = Path("yolov8n.pt")

    model = YOLO(str(base_model))

    start = time.perf_counter()

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        device=device if device else None,
        project=str(VISION_DIR / "runs"),
        name="finetune_padel",
        exist_ok=True,
        patience=20,
        save=True,
        save_period=10,
        # Augmentation for small-object (ball) detection
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        flipud=0.0,
        fliplr=0.5,
    )

    elapsed = time.perf_counter() - start
    log(f"Treino completo em {elapsed:.0f}s")

    # Copy best model
    best_src = VISION_DIR / "runs" / "finetune_padel" / "weights" / "best.pt"
    best_dst = MODELS_DIR / "best.pt"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if best_src.exists():
        shutil.copy2(str(best_src), str(best_dst))
        log(f"Melhor modelo salvo em: {best_dst}")
    else:
        log("ERRO: best.pt não encontrado após treino")
        sys.exit(1)

    # Validate
    metrics = model.val(data=str(data_yaml))
    map50 = float(metrics.box.map50) if hasattr(metrics.box, "map50") else 0.0
    map50_95 = float(metrics.box.map) if hasattr(metrics.box, "map") else 0.0

    log(f"mAP@0.5: {map50:.4f} | mAP@0.5-0.95: {map50_95:.4f} | Target: >0.75")

    return best_dst


def test_on_video(model_path: Path, video_dir: Path, output_dir: Path, confidence: float = 0.25) -> None:
    """Test the fine-tuned model on downloaded video."""
    from ultralytics import YOLO
    import cv2
    import numpy as np

    # Find existing test video
    video_path = None
    if video_dir.exists():
        for f in video_dir.iterdir():
            if f.suffix.lower() in {".mp4", ".mkv", ".webm"}:
                video_path = f
                break

    if not video_path:
        log("Nenhum vídeo de teste encontrado. Pulando teste.")
        return

    log(f"Testando modelo em: {video_path.name}")

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(output_dir / f"{video_path.stem}_finetuned.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    class_names = {0: "ball", 1: "net", 2: "court_line"}
    class_colors = {0: (0, 255, 255), 1: (255, 0, 0), 2: (0, 165, 255)}

    frame_idx = 0
    max_frames = int(30 * fps)  # 30 seconds
    total_dets = {name: 0 for name in class_names.values()}
    inference_times = []

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        results = model(frame, conf=confidence, verbose=False)
        inference_ms = (time.perf_counter() - t0) * 1000
        inference_times.append(inference_ms)

        annotated = frame.copy()
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = class_names.get(cls_id, f"cls_{cls_id}")
                    color = class_colors.get(cls_id, (255, 255, 255))

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_name} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
                    cv2.putText(annotated, label, (x1, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    total_dets[cls_name] = total_dets.get(cls_name, 0) + 1

        info = f"Frame {frame_idx} | {inference_ms:.0f}ms"
        cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        out.write(annotated)
        frame_idx += 1

        if frame_idx % 100 == 0:
            log(f"  Frame {frame_idx}/{max_frames} | Avg: {np.mean(inference_times):.1f}ms")

    cap.release()
    out.release()

    log("")
    log("=" * 60)
    log("  RESULTADOS - MODELO FINE-TUNED")
    log("=" * 60)
    log(f"  Frames processados: {frame_idx}")
    log(f"  Latência média:     {np.mean(inference_times):.1f}ms")
    log(f"  Latência mediana:   {np.median(inference_times):.1f}ms")
    log(f"  Detecções por classe:")
    for cls_name, count in total_dets.items():
        log(f"    {cls_name}: {count}")
    log(f"  Vídeo anotado: {out_path}")
    log("=" * 60)

    # Save comparison metrics
    metrics = {
        "model": str(model_path),
        "video": str(video_path),
        "frames": frame_idx,
        "detections": total_dets,
        "latency_avg_ms": round(float(np.mean(inference_times)), 2),
        "latency_median_ms": round(float(np.median(inference_times)), 2),
        "output_video": out_path,
    }
    metrics_path = output_dir / f"{video_path.stem}_finetuned_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n with Roboflow dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--device", type=str, default="", help="Device: '', 'cpu', '0'")
    parser.add_argument("--api-key", type=str, default="", help="Roboflow API key (optional)")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, just test")
    parser.add_argument("--confidence", type=float, default=0.25, help="Test confidence threshold")
    args = parser.parse_args()

    log("=" * 60)
    log("  PIPELINE: Roboflow Dataset -> Fine-tune -> Test")
    log("=" * 60)

    # Step 1: Download dataset
    if not args.skip_download:
        download_dataset(DATASET_DIR, args.api_key)
    else:
        log(f"Pulando download, usando dataset em: {DATASET_DIR}")

    # Step 2: Find and update data.yaml
    data_yaml = find_data_yaml(DATASET_DIR)
    log(f"data.yaml encontrado: {data_yaml}")

    # Step 3: Remap classes
    if not args.skip_download:
        remap_classes(DATASET_DIR)
        update_data_yaml(data_yaml)

    # Step 4: Train
    if not args.skip_train:
        model_path = train_model(data_yaml, args.epochs, args.batch_size, args.device)
    else:
        model_path = MODELS_DIR / "best.pt"
        if not model_path.exists():
            log(f"ERRO: Modelo não encontrado: {model_path}")
            sys.exit(1)

    # Step 5: Test on video
    test_videos_dir = VISION_DIR / "test_results" / "downloads"
    test_output_dir = VISION_DIR / "test_results"
    test_on_video(model_path, test_videos_dir, test_output_dir, args.confidence)

    log("\nPipeline completo!")


if __name__ == "__main__":
    main()
