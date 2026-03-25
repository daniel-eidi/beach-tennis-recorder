"""
Prepare dataset from YouTube video for Roboflow annotation.
Downloads a specific segment, extracts frames, and optionally uploads to Roboflow.

Usage:
    python scripts/prepare_dataset.py
    python scripts/prepare_dataset.py --api-key YOUR_ROBOFLOW_KEY
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR.parent))

from vision.utils.youtube_download import download_video


def log(msg):
    print(f"[DATASET] {msg}", flush=True)


def download_segment(url, output_dir, start_seconds, end_seconds, resolution="720p"):
    """Download video segment using yt-dlp."""
    import yt_dlp

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "source_video.mp4")

    if os.path.exists(output_path):
        log(f"Video ja existe: {output_path}")
        return output_path

    log(f"Baixando video: {url}")
    log(f"Segmento: {start_seconds}s - {end_seconds}s ({(end_seconds-start_seconds)/60:.1f} min)")

    height_map = {"480p": 480, "720p": 720, "1080p": 1080}
    target_height = height_map.get(resolution, 720)

    format_str = (
        f"bestvideo[height<={target_height}][ext=mp4]+bestaudio[ext=m4a]"
        f"/bestvideo[height<={target_height}]+bestaudio"
        f"/best[height<={target_height}]"
        f"/best"
    )

    ydl_opts = {
        "format": format_str,
        "outtmpl": output_path,
        "merge_output_format": "mp4",
        "quiet": False,
        "noplaylist": True,
        "download_ranges": yt_dlp.utils.download_range_func(
            None, [(start_seconds, end_seconds)]
        ),
        "force_keyframes_at_cuts": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        log(f"Erro no download com recorte: {e}")
        log("Tentando download completo + recorte local via ffmpeg...")
        return download_and_cut_ffmpeg(url, output_dir, start_seconds, end_seconds, resolution)

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        log(f"Download completo: {output_path} ({size_mb:.1f} MB)")
        return output_path

    # yt-dlp may add extension
    for ext in [".mp4", ".mkv", ".webm"]:
        p = output_path.replace(".mp4", ext)
        if os.path.exists(p):
            return p

    log("ERRO: arquivo nao encontrado apos download")
    return None


def download_and_cut_ffmpeg(url, output_dir, start_seconds, end_seconds, resolution):
    """Fallback: download full video then cut with ffmpeg."""
    import subprocess

    full_path = os.path.join(output_dir, "full_video.mp4")
    output_path = os.path.join(output_dir, "source_video.mp4")

    if not os.path.exists(full_path):
        log("Baixando video completo...")
        path = download_video(
            url=url,
            output_dir=output_dir,
            max_duration=None,
            resolution=resolution,
        )
        if path:
            os.rename(path, full_path)
        else:
            log("ERRO: falha no download")
            return None

    # Cut with ffmpeg
    log(f"Recortando com ffmpeg: {start_seconds}s - {end_seconds}s")
    duration = end_seconds - start_seconds
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_seconds),
        "-i", full_path,
        "-t", str(duration),
        "-c", "copy",
        output_path
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        log(f"Recorte completo: {output_path} ({size_mb:.1f} MB)")
        return output_path

    return None


def extract_frames(video_path, output_dir, target_frames=500, min_interval_ms=500):
    """
    Extract frames from video at regular intervals.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        target_frames: Target number of frames to extract
        min_interval_ms: Minimum interval between frames in milliseconds
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"ERRO: nao conseguiu abrir {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    log(f"Video: {width}x{height} @ {fps:.1f}fps, {duration_s:.0f}s, {total_frames} frames")

    # Calculate frame interval
    interval_frames = max(1, total_frames // target_frames)
    min_interval_frames = max(1, int(min_interval_ms / 1000 * fps))
    interval_frames = max(interval_frames, min_interval_frames)

    log(f"Extraindo 1 frame a cada {interval_frames} frames ({interval_frames/fps:.1f}s)")

    extracted = []
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval_frames == 0 and saved_count < target_frames:
            filename = f"frame_{saved_count:05d}_t{frame_idx/fps:.1f}s.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted.append({
                "filename": filename,
                "filepath": filepath,
                "frame_index": frame_idx,
                "timestamp_s": round(frame_idx / fps, 2),
            })
            saved_count += 1

            if saved_count % 50 == 0:
                log(f"  Extraidos: {saved_count}/{target_frames} frames")

        frame_idx += 1

    cap.release()

    log(f"Total extraido: {saved_count} frames em {output_dir}")
    return extracted


def run_pre_annotation(frames_dir, model_path, output_dir):
    """
    Run YOLO inference on frames to generate pre-annotations.
    Helps speed up manual annotation in Roboflow.
    """
    if not os.path.exists(model_path):
        log(f"Modelo nao encontrado: {model_path}. Pulando pre-anotacao.")
        return

    from ultralytics import YOLO
    model = YOLO(model_path)

    os.makedirs(output_dir, exist_ok=True)

    frames = sorted(Path(frames_dir).glob("*.jpg"))
    log(f"Gerando pre-anotacoes para {len(frames)} frames...")

    class_map = model.names  # {0: 'ball', 1: 'net', ...}
    # Map to our target classes
    target_classes = {"ball": 0, "net": 1, "serve line": 2, "serve_line": 2}

    for i, frame_path in enumerate(frames):
        results = model(str(frame_path), conf=0.2, verbose=False)

        # Generate YOLO format annotation
        label_name = frame_path.stem + ".txt"
        label_path = os.path.join(output_dir, label_name)

        lines = []
        img = cv2.imread(str(frame_path))
        h, w = img.shape[:2]

        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = class_map.get(cls_id, "")

                    # Map to target class
                    target_id = target_classes.get(cls_name)
                    if target_id is None:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    # Convert to YOLO format (center_x, center_y, width, height) normalized
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    conf = float(box.conf[0])

                    lines.append(f"{target_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        if (i + 1) % 50 == 0:
            log(f"  Pre-anotados: {i+1}/{len(frames)}")

    log(f"Pre-anotacoes salvas em: {output_dir}")


def upload_to_roboflow(frames_dir, api_key, project_name="beach-tennis-bt", workspace=None):
    """Upload frames to Roboflow for annotation."""
    if not api_key:
        log("API key nao fornecida. Pulando upload.")
        log("")
        log("Para fazer upload manual:")
        log(f"  1. Acesse https://app.roboflow.com")
        log(f"  2. Crie projeto '{project_name}' com classes: ball, net, court_line")
        log(f"  3. Faca upload das imagens de: {frames_dir}")
        return

    try:
        from roboflow import Roboflow

        log(f"Fazendo upload para Roboflow projeto '{project_name}'...")
        rf = Roboflow(api_key=api_key)

        if workspace:
            ws = rf.workspace(workspace)
        else:
            ws = rf.workspace()

        try:
            project = ws.project(project_name)
        except Exception:
            log(f"Projeto '{project_name}' nao encontrado. Crie manualmente no Roboflow.")
            log("Classes recomendadas: ball, net, court_line")
            return

        frames = sorted(Path(frames_dir).glob("*.jpg"))
        log(f"Uploading {len(frames)} frames...")

        for i, frame_path in enumerate(frames):
            project.upload(str(frame_path))
            if (i + 1) % 50 == 0:
                log(f"  Uploaded: {i+1}/{len(frames)}")

        log("Upload completo!")

    except Exception as e:
        log(f"Erro no upload: {e}")
        log(f"Faca upload manual das imagens de: {frames_dir}")


def create_annotation_guide(output_dir):
    """Create a guide for annotating frames."""
    guide = """# Guia de Anotacao - Beach Tennis Dataset

## Classes para anotar:
1. **ball** - Bola de beach tennis (pequena, geralmente branca/amarela)
2. **net** - Rede no centro da quadra
3. **court_line** - Linhas da quadra (todas as linhas visiveis)

## Dicas:
- A bola e PEQUENA - faca bounding boxes bem ajustados
- Anote a bola mesmo quando parcialmente oclusa
- A rede deve ter um bbox que cubra toda a extensao visivel
- Linhas da quadra: anote cada segmento visivel separadamente
- Se a bola nao estiver visivel no frame, nao anote (frame sem bola e ok)
- Priorize frames onde a bola esta em jogo (durante rallies)

## Apos anotar:
1. Exporte do Roboflow em formato YOLOv8
2. Extraia em: vision/dataset/btcanal/
3. Rode: python scripts/finetune_roboflow.py --skip-download --dataset-dir dataset/btcanal --epochs 30
"""
    guide_path = os.path.join(output_dir, "ANNOTATION_GUIDE.md")
    with open(guide_path, "w", encoding="utf-8") as f:
        f.write(guide)
    log(f"Guia de anotacao salvo em: {guide_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset from YouTube video")
    parser.add_argument("--url", type=str,
        default="https://www.youtube.com/watch?v=BGJknXQXuJE",
        help="YouTube video URL")
    parser.add_argument("--start", type=str, default="4:49:00",
        help="Start time (H:MM:SS or seconds)")
    parser.add_argument("--end", type=str, default="5:12:00",
        help="End time (H:MM:SS or seconds)")
    parser.add_argument("--frames", type=int, default=500,
        help="Number of frames to extract (default: 500)")
    parser.add_argument("--output-dir", type=str,
        default=str(PROJECT_DIR / "dataset" / "btcanal"),
        help="Output directory")
    parser.add_argument("--model", type=str,
        default=str(PROJECT_DIR / "models" / "best.pt"),
        help="YOLO model for pre-annotation")
    parser.add_argument("--api-key", type=str, default="",
        help="Roboflow API key for upload")
    parser.add_argument("--project", type=str, default="beach-tennis-bt",
        help="Roboflow project name")
    parser.add_argument("--resolution", type=str, default="720p")
    parser.add_argument("--skip-download", action="store_true",
        help="Skip video download")
    parser.add_argument("--skip-extract", action="store_true",
        help="Skip frame extraction")
    args = parser.parse_args()

    # Parse time strings to seconds
    def parse_time(t):
        parts = t.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return int(t)

    start_s = parse_time(args.start)
    end_s = parse_time(args.end)

    output_dir = args.output_dir
    frames_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    video_dir = os.path.join(output_dir, "video")

    log("=" * 60)
    log("  PREPARACAO DE DATASET - Beach Tennis")
    log("=" * 60)
    log(f"  URL: {args.url}")
    log(f"  Segmento: {args.start} - {args.end} ({(end_s-start_s)/60:.0f} min)")
    log(f"  Frames alvo: {args.frames}")
    log(f"  Output: {output_dir}")
    log("=" * 60)

    # Step 1: Download video segment
    video_path = None
    if not args.skip_download:
        log("\n[1/4] Baixando segmento do video...")
        video_path = download_segment(
            args.url, video_dir, start_s, end_s, args.resolution
        )
        if not video_path:
            log("ERRO: falha no download")
            sys.exit(1)
    else:
        # Find existing video
        for f in Path(video_dir).glob("*"):
            if f.suffix.lower() in {".mp4", ".mkv", ".webm"}:
                video_path = str(f)
                break
        if not video_path:
            log(f"ERRO: nenhum video encontrado em {video_dir}")
            sys.exit(1)
        log(f"Usando video existente: {video_path}")

    # Step 2: Extract frames
    if not args.skip_extract:
        log("\n[2/4] Extraindo frames...")
        frames = extract_frames(video_path, frames_dir, args.frames)
        log(f"  {len(frames)} frames extraidos")

        # Save frame manifest
        manifest_path = os.path.join(output_dir, "frames_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(frames, f, indent=2)
    else:
        log("Pulando extracao de frames")

    # Step 3: Pre-annotate with existing model
    log("\n[3/4] Gerando pre-anotacoes com modelo existente...")
    run_pre_annotation(frames_dir, args.model, labels_dir)

    # Step 4: Upload to Roboflow or save instructions
    log("\n[4/4] Preparando para anotacao...")
    create_annotation_guide(output_dir)

    if args.api_key:
        upload_to_roboflow(frames_dir, args.api_key, args.project)
    else:
        frame_count = len(list(Path(frames_dir).glob("*.jpg")))
        log("")
        log("=" * 60)
        log("  DATASET PRONTO PARA ANOTACAO!")
        log("=" * 60)
        log(f"  Frames: {frame_count} imagens em {frames_dir}")
        log(f"  Pre-anotacoes: {labels_dir}")
        log(f"  Guia: {output_dir}/ANNOTATION_GUIDE.md")
        log("")
        log("  Proximos passos:")
        log("  1. Crie projeto no Roboflow: https://app.roboflow.com")
        log("     Classes: ball, net, court_line")
        log(f"  2. Upload das imagens de: {frames_dir}")
        log(f"  3. Use pre-anotacoes de: {labels_dir}")
        log("  4. Revise e corrija as anotacoes no Roboflow")
        log("  5. Exporte em formato YOLOv8")
        log(f"  6. Extraia em: {output_dir}/exported/")
        log("  7. Retreine: python scripts/finetune_roboflow.py --skip-download --epochs 30")
        log("=" * 60)


if __name__ == "__main__":
    main()
