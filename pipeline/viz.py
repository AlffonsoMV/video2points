"""Visualization utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pipeline.io import ensure_dir, pil_to_np_rgb, np_to_pil_rgb
from pipeline.rendering import _set_axes_equal_3d
from pipeline.vggt import run_vggt_reconstruction


def plot_point_cloud_3d(
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray | None = None,
    title: str = "3D Point Cloud",
    figsize: tuple[int, int] = (8, 6),
    point_size: float = 0.2,
    elev: float = 20.0,
    azim: float = -60.0,
) -> tuple[plt.Figure, Any]:
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_xyz must have shape (N, 3), got {pts.shape}")

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    if colors_rgb is not None:
        c = np.asarray(colors_rgb, dtype=np.float32)
        if c.max() > 1.0:
            c = c / 255.0
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=c, s=point_size, depthshade=False)
    else:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=point_size, depthshade=False)

    _set_axes_equal_3d(ax, pts)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.tight_layout()
    return fig, ax


def save_point_cloud_ply(
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray | None,
    path: str | Path,
) -> Path:
    pts = np.asarray(points_xyz, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_xyz must have shape (N, 3), got {pts.shape}")

    if colors_rgb is None:
        cols = np.full((len(pts), 3), 255, dtype=np.uint8)
    else:
        cols = np.asarray(colors_rgb)
        if cols.shape != pts.shape:
            raise ValueError(f"colors_rgb must have shape {pts.shape}, got {cols.shape}")
        if cols.dtype != np.uint8:
            scale = 255.0 if cols.size and float(np.nanmax(cols)) <= 1.0 else 1.0
            cols = np.clip(cols * scale, 0, 255).astype(np.uint8)

    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertex_data = np.empty(len(pts), dtype=vertex_dtype)
    vertex_data["x"] = pts[:, 0]
    vertex_data["y"] = pts[:, 1]
    vertex_data["z"] = pts[:, 2]
    vertex_data["red"] = cols[:, 0]
    vertex_data["green"] = cols[:, 1]
    vertex_data["blue"] = cols[:, 2]

    path = Path(path)
    ensure_dir(path.parent)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertex_data)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        vertex_data.tofile(f)
    return path


def overlay_mask_on_image(
    image: Image.Image | np.ndarray,
    mask: Image.Image | np.ndarray,
    color: tuple[int, int, int] = (255, 40, 40),
    alpha: float = 0.45,
) -> Image.Image:
    img = pil_to_np_rgb(image) if isinstance(image, Image.Image) else np.asarray(image).copy()
    m = np.asarray(mask.convert("L") if isinstance(mask, Image.Image) else mask)
    m_bool = m > 0
    overlay = np.zeros_like(img, dtype=np.uint8)
    overlay[:, :] = np.array(color, dtype=np.uint8)
    out = img.copy()
    out[m_bool] = np.clip((1 - alpha) * img[m_bool] + alpha * overlay[m_bool], 0, 255).astype(np.uint8)
    return np_to_pil_rgb(out)


def to_pil_mask(mask: np.ndarray | Image.Image) -> Image.Image:
    if isinstance(mask, Image.Image):
        return mask.convert("L")
    m = np.asarray(mask)
    if m.dtype == bool:
        m = m.astype(np.uint8) * 255
    elif m.dtype != np.uint8:
        m = np.clip(m, 0, 255).astype(np.uint8)
    return Image.fromarray(m, mode="L")


def save_matplotlib_figure(fig: plt.Figure, path: str | Path, dpi: int = 150, close: bool = True) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path


def plot_image_grid(images: list[Image.Image], titles: list[str] | None = None, figsize: tuple[int, int] = (16, 5)) -> tuple[plt.Figure, np.ndarray]:
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = np.array([axes])
    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        ax.axis("off")
        if titles and i < len(titles):
            ax.set_title(titles[i])
    fig.tight_layout()
    return fig, axes


def describe_scene(scene: dict[str, Any]) -> dict[str, Any]:
    info = {
        "device": scene.get("device"),
        "image_paths": [str(p) for p in scene.get("image_paths", [])],
        "image_hw": tuple(scene.get("image_hw", ())),
        "preprocessed_shape": tuple(scene.get("preprocessed_shape", ())),
    }
    for key in ["depth", "depth_conf", "world_points", "world_points_conf", "world_points_from_depth", "extrinsic", "intrinsic"]:
        if key in scene:
            info[f"{key}_shape"] = tuple(np.asarray(scene[key]).shape)
    return info


def build_two_view_reconstruction(
    original_image_path: str | Path,
    novel_inpainted_image_path: str | Path,
    model: Any = None,
    device: str | None = None,
    preprocess_mode: str = "crop",
    model_id: str = "facebook/VGGT-1B",
) -> dict[str, Any]:
    return run_vggt_reconstruction(
        [original_image_path, novel_inpainted_image_path],
        model=model,
        device=device,
        preprocess_mode=preprocess_mode,
        model_id=model_id,
    )
