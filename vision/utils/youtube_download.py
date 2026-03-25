"""
Beach Tennis Recorder - YouTube Video Downloader
AGENT-02 | TASK-02-14

Standalone utility to download videos from YouTube for model validation.
Uses yt-dlp Python API to download beach tennis match videos.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yt_dlp


AGENT = "02"
TASK = "02-14"


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


def list_channel_videos(
    channel_url: str,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    List recent videos from a YouTube channel.

    Args:
        channel_url: YouTube channel URL (e.g. https://www.youtube.com/@btcanallive).
        max_results: Maximum number of videos to return.

    Returns:
        List of dicts with video info: {url, title, duration, upload_date, id}.
    """
    log(TASK, "info", f"Listing videos from channel: {channel_url}",
        max_results=max_results)

    # Append /videos to get the uploads page if not already specified
    videos_url = channel_url.rstrip("/")
    if not videos_url.endswith("/videos"):
        videos_url += "/videos"

    ydl_opts: Dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "playlistend": max_results,
    }

    start = time.perf_counter()
    videos: List[Dict[str, Any]] = []

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(videos_url, download=False)
            if info is None:
                log(TASK, "error", "Failed to extract channel info")
                return []

            entries = info.get("entries", [])
            for entry in entries:
                if entry is None:
                    continue
                video_info: Dict[str, Any] = {
                    "id": entry.get("id", ""),
                    "url": f"https://www.youtube.com/watch?v={entry.get('id', '')}",
                    "title": entry.get("title", "Unknown"),
                    "duration": entry.get("duration"),
                    "upload_date": entry.get("upload_date"),
                }
                videos.append(video_info)
                if len(videos) >= max_results:
                    break

    except Exception as exc:
        log(TASK, "error", f"Failed to list channel videos: {exc}")
        return []

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    log(TASK, "ok", f"Found {len(videos)} videos from channel",
        count=len(videos), ms=elapsed_ms)

    return videos


def download_video(
    url: str,
    output_dir: str,
    max_duration: Optional[int] = None,
    resolution: str = "720p",
) -> Optional[str]:
    """
    Download a single YouTube video.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save the downloaded video.
        max_duration: Maximum duration in seconds to download (None = full video).
        resolution: Target resolution (e.g. "720p", "480p", "1080p").

    Returns:
        Path to the downloaded video file, or None on failure.
    """
    os.makedirs(output_dir, exist_ok=True)

    log(TASK, "info", f"Downloading video: {url}",
        output_dir=output_dir, resolution=resolution,
        max_duration=max_duration)

    # Map resolution string to height
    height_map = {"480p": 480, "720p": 720, "1080p": 1080, "360p": 360}
    target_height = height_map.get(resolution, 720)

    # Build format selector: best video up to target height + best audio, merged
    format_str = (
        f"bestvideo[height<={target_height}][ext=mp4]+bestaudio[ext=m4a]"
        f"/bestvideo[height<={target_height}]+bestaudio"
        f"/best[height<={target_height}]"
        f"/best"
    )

    outtmpl = os.path.join(output_dir, "%(id)s_%(title).50s.%(ext)s")

    ydl_opts: Dict[str, Any] = {
        "format": format_str,
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "restrictfilenames": True,
        "noplaylist": True,
    }

    # If max_duration is set, use download_ranges via postprocessor
    if max_duration is not None and max_duration > 0:
        ydl_opts["download_ranges"] = yt_dlp.utils.download_range_func(
            None, [(0, max_duration)]
        )
        ydl_opts["force_keyframes_at_cuts"] = True

    start = time.perf_counter()
    downloaded_path: Optional[str] = None

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                log(TASK, "error", "Failed to extract video info")
                return None

            # Determine the actual downloaded file path
            if info.get("requested_downloads"):
                downloaded_path = info["requested_downloads"][0].get("filepath")
            else:
                # Fallback: construct path from template
                video_id = info.get("id", "unknown")
                title = info.get("title", "video")
                # yt-dlp with restrictfilenames sanitizes the title
                downloaded_path = ydl.prepare_filename(info)

            # Verify file exists
            if downloaded_path and not os.path.exists(downloaded_path):
                # Try with .mp4 extension
                mp4_path = str(Path(downloaded_path).with_suffix(".mp4"))
                if os.path.exists(mp4_path):
                    downloaded_path = mp4_path
                else:
                    log(TASK, "error",
                        f"Downloaded file not found at expected path: {downloaded_path}")
                    return None

    except Exception as exc:
        log(TASK, "error", f"Download failed: {exc}")
        return None

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    if downloaded_path and os.path.exists(downloaded_path):
        file_size_mb = os.path.getsize(downloaded_path) / (1024 * 1024)
        log(TASK, "ok", f"Video downloaded: {Path(downloaded_path).name}",
            path=downloaded_path, size_mb=round(file_size_mb, 2), ms=elapsed_ms)
        return downloaded_path

    log(TASK, "error", "Download completed but file not found")
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download YouTube videos for beach tennis model testing"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Specific YouTube video URL to download",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="https://www.youtube.com/@btcanallive",
        help="YouTube channel URL to list videos from",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=3,
        help="Max videos to list from channel (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_results/downloads",
        help="Output directory for downloaded videos",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="720p",
        choices=["360p", "480p", "720p", "1080p"],
        help="Download resolution (default: 720p)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Max duration in seconds to download (default: full video)",
    )

    args = parser.parse_args()

    if args.url:
        path = download_video(
            url=args.url,
            output_dir=args.output_dir,
            max_duration=args.duration,
            resolution=args.resolution,
        )
        if path:
            print(f"\nDownloaded: {path}")
    else:
        videos = list_channel_videos(args.channel, args.max_videos)
        print(f"\nFound {len(videos)} videos:")
        for v in videos:
            print(f"  [{v['id']}] {v['title']} (duration: {v.get('duration', '?')}s)")
            print(f"    URL: {v['url']}")
