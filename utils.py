from __future__ import annotations

import base64
import gc
import io
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


_VGGT_MODEL_CACHE: dict[tuple[str, str], VGGT] = {}
_INPAINT_PIPE_CACHE: dict[tuple[str, str, str], Any] = {}


@dataclass
class PipelineConfig:
    data_dir: str = "data"
    output_dir: str = "outputs"
    input_image_stem: str = "image_01"
    vggt_model_id: str = "facebook/VGGT-1B"
    inpaint_model_id: str = "black-forest-labs/FLUX.2-klein-4B"
    inpaint_backend: str = "auto"  # auto | diffusers | flux2_klein_bfl
    bfl_api_base_url: str = "https://api.bfl.ai"
    bfl_api_key_env: str = "BFL_API_KEY"
    bfl_poll_interval_seconds: float = 1.0
    bfl_timeout_seconds: int = 600
    bfl_safety_tolerance: int = 2
    bfl_output_format: str = "png"
    preprocess_mode: str = "crop"
    vggt_conf_percentile: float = 50.0
    max_points_plot: int = 30000
    max_points_render: int = 180000
    render_point_radius: int = 1
    novel_shift_right: float = 0.12
    novel_yaw_deg: float = -4.0
    novel_pitch_deg: float = 0.0
    novel_roll_deg: float = 0.0
    mask_dilate_px: int = 4
    mask_close_px: int = 3
    inpaint_steps: int = 6
    inpaint_guidance_scale: float = 1.0
    inpaint_prompt: str = (
        "Photorealistic continuation of the same scene from a slightly shifted viewpoint, "
        "fill only missing regions and preserve visible content, lighting, geometry, and textures."
    )
    inpaint_negative_prompt: str = (
        "changes to visible regions, duplicated objects, warped geometry, blurry details, oversmoothing"
    )
    seed: int = 0


@dataclass
class RenderResult:
    image: Image.Image
    image_np: np.ndarray  # HxWx3 uint8
    valid_mask: np.ndarray  # HxW bool
    depth_map: np.ndarray  # HxW float32, inf where invalid
    projected_count: int


@dataclass
class InpaintResult:
    raw_generated: Image.Image
    composited: Image.Image
    mask_used: Image.Image
    backend: str
    resized_for_model: bool
    metadata: dict[str, Any] | None = None


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clear_torch_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def clear_loaded_model_caches() -> None:
    _VGGT_MODEL_CACHE.clear()
    _INPAINT_PIPE_CACHE.clear()
    clear_torch_cache()


def get_device(prefer_mps: bool = True) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
    image.save(path)
    return path


def pil_to_np_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def np_to_pil_rgb(image: np.ndarray) -> Image.Image:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


def _cuda_autocast_context(device: str) -> Any:
    if device == "cuda":
        dtype = torch.float16
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                dtype = torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def load_vggt_model(
    model_id: str = "facebook/VGGT-1B",
    device: str | None = None,
    use_cache: bool = True,
) -> VGGT:
    device = device or get_device()
    cache_key = (model_id, device)
    if use_cache and cache_key in _VGGT_MODEL_CACHE:
        return _VGGT_MODEL_CACHE[cache_key]

    model = VGGT.from_pretrained(model_id)
    model.eval()
    model.to(device)

    if use_cache:
        _VGGT_MODEL_CACHE[cache_key] = model
    return model


def _squeeze_batch_tensors(predictions: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in predictions.items():
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            if tensor.ndim > 0 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            out[key] = tensor
        else:
            out[key] = value
    return out


def _tensor_image_stack_to_nhwc(images_tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(images_tensor):
        images_tensor = images_tensor.detach().cpu().numpy()
    arr = np.asarray(images_tensor)
    if arr.ndim == 4 and arr.shape[1] == 3:  # S,C,H,W
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected image stack in NCHW or NHWC, got shape {arr.shape}")
    return arr


def run_vggt_reconstruction(
    image_paths: list[str | Path],
    model: VGGT | None = None,
    device: str | None = None,
    model_id: str = "facebook/VGGT-1B",
    preprocess_mode: str = "crop",
) -> dict[str, Any]:
    device = device or get_device()
    image_paths = [Path(p) for p in image_paths]
    if not image_paths:
        raise ValueError("image_paths cannot be empty")

    if model is None:
        model = load_vggt_model(model_id=model_id, device=device)

    images = load_and_preprocess_images([str(p) for p in image_paths], mode=preprocess_mode)
    image_hw = tuple(int(x) for x in images.shape[-2:])
    images_dev = images.to(device)

    with torch.no_grad():
        with _cuda_autocast_context(device):
            predictions = model(images_dev)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], image_hw)

    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    squeezed = _squeeze_batch_tensors(predictions)
    result: dict[str, Any] = {
        "device": device,
        "image_paths": image_paths,
        "image_hw": image_hw,
        "preprocessed_shape": tuple(int(x) for x in images.shape),
        "preprocess_mode": preprocess_mode,
    }

    for key, value in squeezed.items():
        if torch.is_tensor(value):
            result[key] = value.numpy()
        else:
            result[key] = value

    if "images" in result:
        result["images_nhwc"] = _tensor_image_stack_to_nhwc(result["images"])

    if "depth" in result and "extrinsic" in result and "intrinsic" in result:
        result["world_points_from_depth"] = unproject_depth_map_to_point_map(
            result["depth"], result["extrinsic"], result["intrinsic"]
        )

    return result


def resize_intrinsics(intrinsic: np.ndarray, old_hw: tuple[int, int], new_hw: tuple[int, int]) -> np.ndarray:
    old_h, old_w = old_hw
    new_h, new_w = new_hw
    sx = new_w / float(old_w)
    sy = new_h / float(old_h)
    k = intrinsic.copy().astype(np.float32)
    k[0, 0] *= sx
    k[1, 1] *= sy
    k[0, 2] *= sx
    k[1, 2] *= sy
    return k


def build_point_cloud_from_scene(
    scene: dict[str, Any],
    view_idx: int = 0,
    prefer_depth_unprojection: bool = True,
    conf_percentile: float = 50.0,
    max_points: int | None = None,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if prefer_depth_unprojection and "world_points_from_depth" in scene:
        world_points = np.asarray(scene["world_points_from_depth"][view_idx], dtype=np.float32)
        conf = np.asarray(scene.get("depth_conf", np.ones(world_points.shape[:2], dtype=np.float32))[view_idx], dtype=np.float32)
    else:
        world_points = np.asarray(scene["world_points"][view_idx], dtype=np.float32)
        conf = np.asarray(scene.get("world_points_conf", np.ones(world_points.shape[:2], dtype=np.float32))[view_idx], dtype=np.float32)

    images_nhwc = np.asarray(scene["images_nhwc"], dtype=np.float32)
    colors = images_nhwc[view_idx]

    pts = world_points.reshape(-1, 3)
    cols = colors.reshape(-1, 3)
    conf_flat = conf.reshape(-1)

    valid = np.isfinite(pts).all(axis=1)
    valid &= conf_flat > 1e-6
    if conf_percentile is not None:
        threshold = np.percentile(conf_flat[valid], conf_percentile) if np.any(valid) else 0.0
        valid &= conf_flat >= threshold

    pts = pts[valid]
    cols = cols[valid]
    conf_kept = conf_flat[valid]

    if max_points is not None and len(pts) > max_points:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]
        cols = cols[idx]
        conf_kept = conf_kept[idx]

    return pts.astype(np.float32), cols.astype(np.float32), conf_kept.astype(np.float32)


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


def _set_axes_equal_3d(ax: Any, pts: np.ndarray) -> None:
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0 + 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def camera_center_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    r = extrinsic[:, :3]
    t = extrinsic[:, 3]
    return (-r.T @ t).astype(np.float32)


def extrinsic_from_camera_pose(r_world_from_cam: np.ndarray, cam_center_world: np.ndarray) -> np.ndarray:
    r_cam_from_world = r_world_from_cam.T
    t = -r_cam_from_world @ cam_center_world
    return np.concatenate([r_cam_from_world, t[:, None]], axis=1).astype(np.float32)


def perturb_camera_extrinsic(
    extrinsic: np.ndarray,
    shift_right: float = 0.2,
    shift_down: float = 0.0,
    shift_forward: float = 0.0,
    yaw_deg: float = -8.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """
    Perturb a world->camera OpenCV extrinsic by a local camera-space translation + rotation.
    Local axes use OpenCV convention: x-right, y-down, z-forward.
    """
    extrinsic = np.asarray(extrinsic, dtype=np.float32)
    r_cam_from_world = extrinsic[:, :3]
    r_world_from_cam = r_cam_from_world.T
    c_world = camera_center_from_extrinsic(extrinsic)

    delta_local = np.array([shift_right, shift_down, shift_forward], dtype=np.float32)
    delta_world = r_world_from_cam @ delta_local
    c_new = c_world + delta_world

    r_local = Rotation.from_euler("yxz", [yaw_deg, pitch_deg, roll_deg], degrees=True).as_matrix().astype(np.float32)
    r_world_from_cam_new = r_world_from_cam @ r_local
    return extrinsic_from_camera_pose(r_world_from_cam_new, c_new)


def project_world_points(
    points_xyz: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points_xyz, dtype=np.float32)
    r = extrinsic[:, :3].astype(np.float32)
    t = extrinsic[:, 3].astype(np.float32)
    pts_cam = (r @ pts.T).T + t[None, :]
    z = pts_cam[:, 2]
    valid = np.isfinite(pts_cam).all(axis=1) & (z > eps)
    uv = np.full((len(pts), 2), np.nan, dtype=np.float32)
    if np.any(valid):
        p = (intrinsic @ pts_cam[valid].T).T
        uv_valid = p[:, :2] / p[:, 2:3]
        uv[valid] = uv_valid
    return uv, z.astype(np.float32), valid


def _expand_splats(
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    colors: np.ndarray,
    h: int,
    w: int,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if radius <= 0:
        valid = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        return u[valid], v[valid], z[valid], colors[valid]

    uu_all, vv_all, zz_all, cc_all = [], [], [], []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            uu = u + dx
            vv = v + dy
            valid = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            if not np.any(valid):
                continue
            uu_all.append(uu[valid])
            vv_all.append(vv[valid])
            zz_all.append(z[valid])
            cc_all.append(colors[valid])
    if not uu_all:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
        )
    return (
        np.concatenate(uu_all).astype(np.int32),
        np.concatenate(vv_all).astype(np.int32),
        np.concatenate(zz_all).astype(np.float32),
        np.concatenate(cc_all).astype(np.uint8),
    )


def render_projected_point_cloud(
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_hw: tuple[int, int],
    point_radius: int = 1,
    background_color: tuple[int, int, int] = (255, 255, 255),
) -> RenderResult:
    h, w = image_hw
    pts = np.asarray(points_xyz, dtype=np.float32)
    colors = np.asarray(colors_rgb)
    if colors.dtype != np.uint8:
        colors = np.clip(colors * (255.0 if colors.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)

    uv, z, valid3d = project_world_points(pts, extrinsic, intrinsic)
    valid = valid3d & np.isfinite(uv).all(axis=1)
    uv = uv[valid]
    z = z[valid]
    colors = colors[valid]

    if len(uv) == 0:
        canvas = np.full((h, w, 3), background_color, dtype=np.uint8)
        depth_map = np.full((h, w), np.inf, dtype=np.float32)
        return RenderResult(np_to_pil_rgb(canvas), canvas, np.zeros((h, w), dtype=bool), depth_map, 0)

    u = np.rint(uv[:, 0]).astype(np.int32)
    v = np.rint(uv[:, 1]).astype(np.int32)
    u, v, z, colors = _expand_splats(u, v, z, colors, h, w, radius=point_radius)

    canvas = np.full((h, w, 3), background_color, dtype=np.uint8)
    depth_map = np.full((h, w), np.inf, dtype=np.float32)
    if len(u) == 0:
        return RenderResult(np_to_pil_rgb(canvas), canvas, np.zeros((h, w), dtype=bool), depth_map, 0)

    pix = v.astype(np.int64) * w + u.astype(np.int64)
    order = np.argsort(z, kind="stable")  # nearest first
    pix_sorted = pix[order]
    _, first_idx = np.unique(pix_sorted, return_index=True)
    chosen = order[first_idx]

    u_c = u[chosen]
    v_c = v[chosen]
    z_c = z[chosen]
    c_c = colors[chosen]

    canvas[v_c, u_c] = c_c
    depth_map[v_c, u_c] = z_c
    valid_mask = np.isfinite(depth_map)
    return RenderResult(np_to_pil_rgb(canvas), canvas, valid_mask, depth_map, int(len(chosen)))


def render_scene_view(
    scene: dict[str, Any],
    view_idx: int = 0,
    conf_percentile: float = 50.0,
    max_points: int | None = None,
    point_radius: int = 1,
    prefer_depth_unprojection: bool = True,
    output_hw: tuple[int, int] | None = None,
) -> RenderResult:
    pts, cols, _ = build_point_cloud_from_scene(
        scene,
        view_idx=view_idx,
        prefer_depth_unprojection=prefer_depth_unprojection,
        conf_percentile=conf_percentile,
        max_points=max_points,
    )
    old_hw = tuple(int(x) for x in scene["image_hw"])
    image_hw = output_hw or old_hw
    extrinsic = np.asarray(scene["extrinsic"][view_idx], dtype=np.float32)
    intrinsic = np.asarray(scene["intrinsic"][view_idx], dtype=np.float32)
    if image_hw != old_hw:
        intrinsic = resize_intrinsics(intrinsic, old_hw, image_hw)
    return render_projected_point_cloud(
        pts, cols, extrinsic, intrinsic, image_hw=image_hw, point_radius=point_radius
    )


def render_scene_from_custom_camera(
    scene: dict[str, Any],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    conf_percentile: float = 50.0,
    max_points: int | None = None,
    point_radius: int = 1,
    prefer_depth_unprojection: bool = True,
    source_view_idx_for_colors: int = 0,
) -> RenderResult:
    pts, cols, _ = build_point_cloud_from_scene(
        scene,
        view_idx=source_view_idx_for_colors,
        prefer_depth_unprojection=prefer_depth_unprojection,
        conf_percentile=conf_percentile,
        max_points=max_points,
    )
    h = int(intrinsic[1, 2] * 2)
    w = int(intrinsic[0, 2] * 2)
    if h <= 0 or w <= 0:
        h, w = tuple(int(x) for x in scene["image_hw"])
    return render_projected_point_cloud(
        pts, cols, np.asarray(extrinsic, dtype=np.float32), np.asarray(intrinsic, dtype=np.float32), (h, w), point_radius
    )


def build_novel_view_render(
    scene: dict[str, Any],
    base_view_idx: int = 0,
    shift_right: float = 0.2,
    yaw_deg: float = -8.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    conf_percentile: float = 50.0,
    max_points: int | None = None,
    point_radius: int = 1,
) -> tuple[RenderResult, np.ndarray, np.ndarray]:
    extr = np.asarray(scene["extrinsic"][base_view_idx], dtype=np.float32)
    intr = np.asarray(scene["intrinsic"][base_view_idx], dtype=np.float32)
    novel_extr = perturb_camera_extrinsic(
        extr,
        shift_right=shift_right,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
    )
    render = render_scene_from_custom_camera(
        scene,
        extrinsic=novel_extr,
        intrinsic=intr,
        conf_percentile=conf_percentile,
        max_points=max_points,
        point_radius=point_radius,
        source_view_idx_for_colors=base_view_idx,
    )
    return render, novel_extr, intr


def build_hole_mask_from_valid_mask(
    valid_mask: np.ndarray,
    dilate_px: int = 4,
    close_px: int = 3,
    min_area_px: int = 16,
) -> np.ndarray:
    hole_mask = (~np.asarray(valid_mask, dtype=bool)).astype(np.uint8) * 255
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
        hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_CLOSE, k)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        hole_mask = cv2.dilate(hole_mask, k)

    if min_area_px > 1:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((hole_mask > 0).astype(np.uint8), connectivity=8)
        cleaned = np.zeros_like(hole_mask)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_area_px:
                cleaned[labels == label] = 255
        hole_mask = cleaned
    return hole_mask


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


def _resize_to_multiple(image: Image.Image, mask: Image.Image, multiple: int = 8) -> tuple[Image.Image, Image.Image, bool]:
    w, h = image.size
    new_w = max(multiple, (w // multiple) * multiple)
    new_h = max(multiple, (h // multiple) * multiple)
    if new_w == w and new_h == h:
        return image, mask, False
    return (
        image.resize((new_w, new_h), Image.Resampling.LANCZOS),
        mask.resize((new_w, new_h), Image.Resampling.NEAREST),
        True,
    )


def _resize_to_multiple_of_8(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image, bool]:
    return _resize_to_multiple(image, mask, multiple=8)


def _composite_preserve_unmasked(
    original: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    orig = pil_to_np_rgb(original)
    gen = pil_to_np_rgb(generated)
    m = np.asarray(mask.convert("L")) > 0
    out = orig.copy()
    out[m] = gen[m]
    return np_to_pil_rgb(out)


def _pil_to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _looks_like_bfl_flux2_klein(model_id: str) -> bool:
    mid = (model_id or "").strip().lower()
    return (
        mid.startswith("bfl:flux-2-klein-")
        or mid.startswith("flux-2-klein-")
        or "flux.2-klein" in mid
    )


def _normalize_bfl_flux2_klein_endpoint(model_id: str) -> str:
    mid = (model_id or "").strip()
    if mid.startswith("bfl:"):
        mid = mid[len("bfl:") :]
    if not mid.startswith("flux-2-klein-"):
        raise ValueError(f"Unsupported FLUX2-klein model id '{model_id}'. Expected e.g. 'bfl:flux-2-klein-9b'.")
    return mid


def opencv_inpaint_fallback(image: Image.Image, mask: Image.Image, radius: int = 3) -> Image.Image:
    img_np = cv2.cvtColor(pil_to_np_rgb(image), cv2.COLOR_RGB2BGR)
    mask_np = np.asarray(mask.convert("L"))
    out = cv2.inpaint(img_np, mask_np, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return np_to_pil_rgb(out)


def _masked_region_near_black(image: Image.Image, mask: Image.Image, threshold_mean: float = 2.0, threshold_std: float = 2.0) -> bool:
    mask_np = np.asarray(mask.convert("L")) > 0
    if not np.any(mask_np):
        return False
    img_np = np.asarray(image.convert("RGB"), dtype=np.float32)
    region = img_np[mask_np]
    if region.size == 0:
        return False
    return float(region.mean()) < threshold_mean and float(region.std()) < threshold_std


def _load_reference_images(reference_images: Iterable[Image.Image | str | Path] | None) -> list[Image.Image]:
    if reference_images is None:
        return []
    refs: list[Image.Image] = []
    for ref in reference_images:
        if isinstance(ref, Image.Image):
            refs.append(ref.convert("RGB"))
        else:
            refs.append(Image.open(ref).convert("RGB"))
    return refs


def load_flux2_klein_pipeline(
    model_id: str = "black-forest-labs/FLUX.2-klein-4B",
    device: str | None = None,
    use_cache: bool = True,
) -> Any:
    import diffusers

    device = device or get_device()
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    cache_key = (f"flux2_klein::{model_id}", device, str(dtype))
    if use_cache and cache_key in _INPAINT_PIPE_CACHE:
        return _INPAINT_PIPE_CACHE[cache_key]

    Flux2KleinPipeline = getattr(diffusers, "Flux2KleinPipeline", None)
    if Flux2KleinPipeline is None:
        raise RuntimeError(
            "Local FLUX.2-klein requires diffusers main branch (Flux2KleinPipeline is missing). "
            "Install with: pip install --upgrade --no-cache-dir git+https://github.com/huggingface/diffusers.git@main"
        )

    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dtype)

    # GPU-only offload helpers are not useful on MPS/CPU; only enable on CUDA.
    if device == "cuda":
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe.to(device)
    else:
        pipe.to(device)

    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    if use_cache:
        _INPAINT_PIPE_CACHE[cache_key] = pipe
    return pipe


def edit_with_flux2_klein_local(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str | None = None,  # kept for API symmetry; may be unsupported by local Flux2Klein pipeline
    model_id: str = "black-forest-labs/FLUX.2-klein-4B",
    device: str | None = None,
    reference_images: Iterable[Image.Image | str | Path] | None = None,
    num_inference_steps: int = 4,
    guidance_scale: float = 1.0,
    seed: int = 0,
) -> InpaintResult:
    del negative_prompt

    device = device or get_device()
    image = image.convert("RGB")
    mask = mask.convert("L")
    refs = _load_reference_images(reference_images)

    # FLUX2 docs/BFL docs mention image inputs are adjusted to multiples of 16. Pre-align ourselves.
    image_for_pipe, mask_for_pipe, resized = _resize_to_multiple(image, mask, multiple=16)
    # Encourage the edit model to focus on the holes by zeroing them in the base input while preserving context elsewhere.
    masked_base = _composite_preserve_unmasked(
        image_for_pipe,
        Image.new("RGB", image_for_pipe.size, (255, 255, 255)),
        mask_for_pipe,
    )
    refs_for_pipe = [
        ref.resize(image_for_pipe.size, Image.Resampling.LANCZOS) if ref.size != image_for_pipe.size else ref
        for ref in refs
    ]

    pipe = load_flux2_klein_pipeline(model_id=model_id, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    image_arg: list[Image.Image] | Image.Image
    if refs_for_pipe:
        # Flux2KleinPipeline supports multi-reference editing by passing a list of images.
        image_arg = [masked_base, *refs_for_pipe]
    else:
        image_arg = masked_base

    result = pipe(
        prompt=prompt,
        image=image_arg,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
    )
    raw = result.images[0].convert("RGB")
    if resized:
        raw = raw.resize(image.size, Image.Resampling.LANCZOS)
    composited = _composite_preserve_unmasked(image, raw, mask)
    return InpaintResult(
        raw_generated=raw,
        composited=composited,
        mask_used=mask,
        backend="flux2_klein_local_diffusers",
        resized_for_model=resized,
        metadata={"reference_count": len(refs_for_pipe), "model_id": model_id},
    )


def inpaint_with_flux2_klein_bfl(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str | None = None,  # kept for API symmetry; currently not used by BFL FLUX2-klein API
    model_id: str = "bfl:flux-2-klein-9b",
    reference_images: Iterable[Image.Image | str | Path] | None = None,
    seed: int = 0,
    num_inference_steps: int = 20,
    guidance_scale: float = 6.5,
    bfl_api_key: str | None = None,
    bfl_api_key_env: str = "BFL_API_KEY",
    bfl_api_base_url: str = "https://api.bfl.ai",
    bfl_poll_interval_seconds: float = 1.0,
    bfl_timeout_seconds: int = 600,
    bfl_safety_tolerance: int = 2,
    bfl_output_format: str = "png",
) -> InpaintResult:
    import requests

    del negative_prompt  # not supported in the BFL FLUX2-klein editing endpoint (as documented)

    image = image.convert("RGB")
    mask = mask.convert("L")
    # BFL notes input images may be cropped/resized to multiples of 16; pre-align ourselves.
    image_for_api, mask_for_api, resized = _resize_to_multiple(image, mask, multiple=16)

    endpoint = _normalize_bfl_flux2_klein_endpoint(model_id)
    api_key = bfl_api_key or os.getenv(bfl_api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing BFL API key. Set {bfl_api_key_env} or pass bfl_api_key to use {endpoint}."
        )

    refs = _load_reference_images(reference_images)
    # BFL docs note FLUX2-klein supports up to 4 reference images via API.
    # We use input_image as the editable base image, then add up to 3 extra references.
    max_extra_refs = 3
    if len(refs) > max_extra_refs:
        print(f"[inpaint_with_flux2_klein_bfl] Truncating references from {len(refs)} to {max_extra_refs} for FLUX2-klein API.")
        refs = refs[:max_extra_refs]

    # Resize references to match the edited image resolution for more stable alignment/style transfer.
    refs_for_api = [ref.resize(image_for_api.size, Image.Resampling.LANCZOS) if ref.size != image_for_api.size else ref for ref in refs]

    payload: dict[str, Any] = {
        "prompt": prompt,
        "input_image": _pil_to_base64_png(image_for_api),
        "seed": int(seed),
        "steps": int(num_inference_steps),
        "guidance": float(guidance_scale),
        "safety_tolerance": int(bfl_safety_tolerance),
        "output_format": bfl_output_format,
        "prompt_upsampling": False,
    }
    for idx, ref in enumerate(refs_for_api, start=2):
        payload[f"input_image_{idx}"] = _pil_to_base64_png(ref)

    base_url = bfl_api_base_url.rstrip("/")
    url = f"{base_url}/v1/{endpoint}"
    headers = {"x-key": api_key, "accept": "application/json", "content-type": "application/json"}

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    start_data = resp.json()
    polling_url = start_data.get("polling_url")
    if not polling_url:
        raise RuntimeError(f"BFL FLUX2-klein start response missing polling_url: {start_data}")
    if polling_url.startswith("/"):
        polling_url = f"{base_url}{polling_url}"

    t0 = time.time()
    last_data: dict[str, Any] | None = None
    while True:
        if time.time() - t0 > float(bfl_timeout_seconds):
            raise TimeoutError(f"BFL FLUX2-klein request timed out after {bfl_timeout_seconds}s ({polling_url})")

        poll = requests.get(polling_url, headers={"x-key": api_key, "accept": "application/json"}, timeout=60)
        poll.raise_for_status()
        last_data = poll.json()
        status = str(last_data.get("status", "")).strip().lower()

        if status == "ready":
            break
        if status in {"error", "failed", "request_moderated", "content_moderated", "moderated"}:
            raise RuntimeError(f"BFL FLUX2-klein failed with status='{last_data.get('status')}'. Payload: {last_data}")
        if status in {"pending", "processing", "running"}:
            time.sleep(float(bfl_poll_interval_seconds))
            continue

        # Unknown status: keep polling unless a result already exists.
        if "result" in last_data and isinstance(last_data["result"], dict) and last_data["result"].get("sample"):
            break
        time.sleep(float(bfl_poll_interval_seconds))

    result_data = (last_data or {}).get("result", {})
    sample_url = result_data.get("sample")
    if not sample_url:
        raise RuntimeError(f"BFL FLUX2-klein result missing result.sample. Payload: {last_data}")

    img_resp = requests.get(sample_url, timeout=120)
    img_resp.raise_for_status()
    raw = Image.open(io.BytesIO(img_resp.content)).convert("RGB")

    if resized:
        raw = raw.resize(image.size, Image.Resampling.LANCZOS)
    composited = _composite_preserve_unmasked(image, raw, mask)
    return InpaintResult(
        raw_generated=raw,
        composited=composited,
        mask_used=mask,
        backend=f"bfl_{endpoint}_api",
        resized_for_model=resized,
        metadata={
            "endpoint": endpoint,
            "polling_url": polling_url,
            "result": last_data,
            "sample_url": sample_url,
            "reference_count": len(refs_for_api),
        },
    )


def load_inpainting_pipeline(
    model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    device: str | None = None,
    use_cache: bool = True,
) -> Any:
    from diffusers import AutoPipelineForInpainting

    device = device or get_device()
    # MPS is often more stable with float32 for SD inpainting than float16.
    dtype = torch.float16 if device == "cuda" else torch.float32
    cache_key = (model_id, device, str(dtype))
    if use_cache and cache_key in _INPAINT_PIPE_CACHE:
        return _INPAINT_PIPE_CACHE[cache_key]

    def _load_pipe(**kwargs: Any) -> Any:
        return AutoPipelineForInpainting.from_pretrained(model_id, **kwargs)

    pipe = None
    load_errors: list[str] = []
    base_kwargs = {"torch_dtype": dtype, "safety_checker": None, "requires_safety_checker": False}
    if dtype == torch.float16:
        try:
            pipe = _load_pipe(**base_kwargs, variant="fp16")
        except Exception as exc:
            load_errors.append(f"fp16 variant load failed: {exc}")
    if pipe is None:
        try:
            pipe = _load_pipe(**base_kwargs)
        except Exception as exc:
            load_errors.append(f"default dtype load failed: {exc}")
            pipe = _load_pipe(torch_dtype=torch.float32)
            dtype = torch.float32

    pipe = pipe.to(device)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    if use_cache:
        _INPAINT_PIPE_CACHE[cache_key] = pipe
    if load_errors:
        print("[load_inpainting_pipeline] warnings:")
        for err in load_errors:
            print(" -", err)
    return pipe


def inpaint_with_diffusion(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str | None = "blurry, distorted, duplicated structures, changed visible regions",
    model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    device: str | None = None,
    reference_images: Iterable[Image.Image | str | Path] | None = None,
    backend: str = "auto",
    num_inference_steps: int = 20,
    guidance_scale: float = 6.5,
    seed: int = 0,
    allow_fallback_to_opencv: bool = True,
    bfl_api_key: str | None = None,
    bfl_api_key_env: str = "BFL_API_KEY",
    bfl_api_base_url: str = "https://api.bfl.ai",
    bfl_poll_interval_seconds: float = 1.0,
    bfl_timeout_seconds: int = 600,
    bfl_safety_tolerance: int = 2,
    bfl_output_format: str = "png",
) -> InpaintResult:
    device = device or get_device()
    if backend == "auto":
        if _looks_like_bfl_flux2_klein(model_id):
            if str(model_id).startswith("bfl:"):
                backend = "flux2_klein_bfl"
            else:
                backend = "flux2_klein_local"
        else:
            backend = "diffusers"

    if backend == "flux2_klein_bfl":
        return inpaint_with_flux2_klein_bfl(
            image=image,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_id=model_id,
            reference_images=reference_images,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            bfl_api_key=bfl_api_key,
            bfl_api_key_env=bfl_api_key_env,
            bfl_api_base_url=bfl_api_base_url,
            bfl_poll_interval_seconds=bfl_poll_interval_seconds,
            bfl_timeout_seconds=bfl_timeout_seconds,
            bfl_safety_tolerance=bfl_safety_tolerance,
            bfl_output_format=bfl_output_format,
        )

    if backend == "flux2_klein_local":
        try:
            return edit_with_flux2_klein_local(
                image=image,
                mask=mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                model_id=model_id,
                device=device,
                reference_images=reference_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
        except Exception as exc:
            if not allow_fallback_to_opencv:
                raise
            print(f"[inpaint_with_diffusion] Local FLUX2-klein failed, falling back to OpenCV Telea: {exc}")
            image = image.convert("RGB")
            mask = mask.convert("L")
            raw = opencv_inpaint_fallback(image, mask)
            composited = _composite_preserve_unmasked(image, raw, mask)
            return InpaintResult(raw_generated=raw, composited=composited, mask_used=mask, backend="opencv_telea_fallback", resized_for_model=False, metadata={"flux2_error": str(exc)})

    image = image.convert("RGB")
    mask = mask.convert("L")
    image_for_pipe, mask_for_pipe, resized = _resize_to_multiple_of_8(image, mask)

    try:
        pipe = load_inpainting_pipeline(model_id=model_id, device=device)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image_for_pipe,
            mask_image=mask_for_pipe,
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        )
        raw = result.images[0].convert("RGB")
        if resized:
            raw = raw.resize(image.size, Image.Resampling.LANCZOS)
        if device == "mps" and _masked_region_near_black(raw, mask):
            print("[inpaint_with_diffusion] Degenerate MPS inpainting output detected, retrying on CPU.")
            return inpaint_with_diffusion(
                image=image,
                mask=mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                model_id=model_id,
                device="cpu",
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                allow_fallback_to_opencv=allow_fallback_to_opencv,
            )
        composited = _composite_preserve_unmasked(image, raw, mask)
        return InpaintResult(raw_generated=raw, composited=composited, mask_used=mask, backend="diffusers", resized_for_model=resized)
    except Exception as exc:
        if not allow_fallback_to_opencv:
            raise
        print(f"[inpaint_with_diffusion] Diffusion inpainting failed, falling back to OpenCV Telea: {exc}")
        raw = opencv_inpaint_fallback(image, mask)
        composited = _composite_preserve_unmasked(image, raw, mask)
        return InpaintResult(raw_generated=raw, composited=composited, mask_used=mask, backend="opencv_telea_fallback", resized_for_model=resized)


def prepare_novel_view_inpainting_inputs(
    render: RenderResult,
    dilate_px: int = 4,
    close_px: int = 3,
    min_area_px: int = 16,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    novel_img = render.image.convert("RGB")
    hole_mask_np = build_hole_mask_from_valid_mask(render.valid_mask, dilate_px=dilate_px, close_px=close_px, min_area_px=min_area_px)
    hole_mask = to_pil_mask(hole_mask_np)
    overlay = overlay_mask_on_image(novel_img, hole_mask)
    return novel_img, hole_mask, overlay


def save_matplotlib_figure(fig: plt.Figure, path: str | Path, dpi: int = 150, close: bool = True) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path


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


def build_two_view_reconstruction(
    original_image_path: str | Path,
    novel_inpainted_image_path: str | Path,
    model: VGGT | None = None,
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
