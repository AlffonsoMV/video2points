"""I/O utilities: directories, images, videos."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_input_image(
    data_dir: str | Path = "data",
    stem: str = "image_01",
    allowed_suffixes: Iterable[str] = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"),
) -> Path:
    data_dir = Path(data_dir)
    for suffix in allowed_suffixes:
        p = data_dir / f"{stem}{suffix}"
        if p.exists():
            return p
    existing = sorted([p.name for p in data_dir.glob("*") if p.is_file()]) if data_dir.exists() else []
    raise FileNotFoundError(f"Could not find {stem} with known suffixes in {data_dir}. Found: {existing}")


def list_data_images(
    data_dir: str | Path = "data",
    exts: Iterable[str] = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"),
) -> list[Path]:
    """Return sorted paths to image files in data_dir."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    exts_set = set(exts)
    return sorted([p for p in data_dir.glob("*") if p.is_file() and p.suffix in exts_set])


def list_videos(
    data_dir: str | Path = "data",
    exts: Iterable[str] = (".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"),
) -> list[Path]:
    """Return sorted paths to video files in data_dir."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    exts_set = set(exts)
    return sorted([p for p in data_dir.glob("*") if p.is_file() and p.suffix in exts_set])


def extract_frame_from_video(
    video_path: str | Path,
    frame_index: int | str = "middle",
) -> Image.Image:
    """
    Extract a single frame from a video.
    frame_index: int (0-based), or "first", "middle", "last"
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        raise RuntimeError(f"Video has no frames: {video_path}")

    if frame_index == "first":
        idx = 0
    elif frame_index == "middle":
        idx = total // 2
    elif frame_index == "last":
        idx = total - 1
    else:
        idx = int(frame_index)
    idx = max(0, min(idx, total - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {idx} from {video_path}")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame, mode="RGB")


def extract_frames_from_videos(
    data_dir: str | Path = "data",
    frame_index: int | str = "middle",
    output_stem: str = "image",
    overwrite: bool = True,
) -> list[Path]:
    """
    Extract one frame from each video in data_dir, save as image_01.png, image_02.png, etc.
    Returns list of saved image paths.
    """
    videos = list_videos(data_dir)
    if not videos:
        raise FileNotFoundError(f"No video files found in {data_dir}")
    data_dir = Path(data_dir)
    saved: list[Path] = []
    for i, v in enumerate(videos, start=1):
        out_path = data_dir / f"{output_stem}_{i:02d}.png"
        if out_path.exists() and not overwrite:
            saved.append(out_path)
            continue
        img = extract_frame_from_video(v, frame_index=frame_index)
        save_pil(img, out_path)
        saved.append(out_path)
    return saved


def extract_n_frames_from_video(
    video_path: str | Path,
    n_frames: int = 7,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """
    Extract n_frames evenly-spaced frames from a single video and save as PNGs.
    Returns list of saved image paths.
    """
    video_path = Path(video_path)
    if output_dir is None:
        output_dir = video_path.parent
    output_dir = ensure_dir(output_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        raise RuntimeError(f"Video has no frames: {video_path}")

    n_frames = min(n_frames, total)
    indices = np.linspace(0, total - 1, n_frames, dtype=int)

    def read_frame_near(target_idx: int, search_radius: int = 2) -> np.ndarray | None:
        candidate_indices = [int(np.clip(target_idx, 0, total - 1))]
        for offset in range(1, search_radius + 1):
            candidate_indices.append(int(np.clip(target_idx - offset, 0, total - 1)))
            candidate_indices.append(int(np.clip(target_idx + offset, 0, total - 1)))

        # Some videos expose a nominal frame count that includes a tail of unreadable frames.
        # If the local neighborhood fails, walk backward through the clip so the caller still
        # gets a stable set of samples instead of silently dropping the last view.
        candidate_indices.extend(range(max(target_idx - search_radius - 1, -1), -1, -1))

        tried: set[int] = set()
        for idx in candidate_indices:
            if idx in tried:
                continue
            tried.add(idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                return frame
        return None

    saved: list[Path] = []
    stem = video_path.stem
    for i, frame_idx in enumerate(indices):
        frame = read_frame_near(int(frame_idx))
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out_path = output_dir / f"{stem}_frame_{i:03d}.png"
        Image.fromarray(frame, mode="RGB").save(out_path)
        saved.append(out_path)

    cap.release()
    if not saved:
        raise RuntimeError(f"Could not extract any frames from {video_path}")
    return saved


def ensure_png_copy(image_path: str | Path, target_name: str | None = None) -> Path:
    src = Path(image_path)
    target = src.with_suffix(".png") if target_name is None else src.parent / target_name
    if target.exists():
        return target
    img = Image.open(src).convert("RGB")
    img.save(target)
    return target


def load_pil_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_pil(image: Image.Image, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    try:
        image.save(path)
    except OSError as e:
        if e.errno == 28:  # No space left on device
            raise RuntimeError(
                "No space left on device. Free disk space and retry:\n"
                "  python -m scripts.clear_hf_cache           # removes SDXL (~14GB)\n"
                "  python -m scripts.clear_hf_cache --remove-partial  # removes partial FLUX (~21GB)"
            ) from e
        raise
    return path


def check_disk_space(path: str | Path, min_gb: float = 5.0) -> tuple[float, bool]:
    """Return (free_gb, ok). Warn if free < min_gb."""
    path = Path(path).resolve()
    if not path.exists():
        path = path.parent if path.parent.exists() else Path.cwd()
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)
    return free_gb, free_gb >= min_gb


def pil_to_np_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def np_to_pil_rgb(image: np.ndarray) -> Image.Image:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def coerce_image_rgb(image: Image.Image | str | Path) -> Image.Image:
    """Load or convert image to RGB PIL Image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")
