"""Point cloud rendering utilities."""
from __future__ import annotations

from typing import Any

import numpy as np

from pipeline.config import RenderResult
from pipeline.geometry import project_world_points, perturb_camera_extrinsic, resize_intrinsics
from pipeline.io import np_to_pil_rgb
from pipeline.point_cloud import build_point_cloud_from_scene


def _set_axes_equal_3d(ax: Any, pts: np.ndarray) -> None:
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0 + 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


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
