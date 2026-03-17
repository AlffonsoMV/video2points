"""Point cloud building from scene dict."""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


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


def merge_scene_point_cloud(
    scene: dict[str, Any],
    view_indices: Iterable[int] | None = None,
    prefer_depth_unprojection: bool = True,
    conf_percentile: float = 50.0,
    max_points: int | None = None,
    rng_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    total_views = int(np.asarray(scene["extrinsic"]).shape[0])
    indices = list(view_indices) if view_indices is not None else list(range(total_views))

    pts_all, cols_all = [], []
    for idx in indices:
        pts, cols, _ = build_point_cloud_from_scene(
            scene,
            view_idx=idx,
            prefer_depth_unprojection=prefer_depth_unprojection,
            conf_percentile=conf_percentile,
            max_points=None,
            rng_seed=rng_seed,
        )
        if len(pts):
            pts_all.append(pts)
            cols_all.append(cols)

    if not pts_all:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    pts = np.concatenate(pts_all, axis=0).astype(np.float32)
    cols = np.concatenate(cols_all, axis=0).astype(np.float32)
    if max_points is not None and len(pts) > max_points:
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]
        cols = cols[idx]
    return pts, cols
