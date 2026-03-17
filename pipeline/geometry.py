"""Camera and geometry utilities."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


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


def camera_baseline_scale(extrinsics: np.ndarray, idx_a: int = 0, idx_b: int = 1) -> float:
    """Distance between camera centers at idx_a and idx_b."""
    c_a = camera_center_from_extrinsic(np.asarray(extrinsics[idx_a], dtype=np.float32))
    c_b = camera_center_from_extrinsic(np.asarray(extrinsics[idx_b], dtype=np.float32))
    return float(np.linalg.norm(c_b - c_a))


def relative_camera_pose_from_reference(
    reference_extrinsic: np.ndarray,
    target_extrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (relative_rotation, delta_in_ref_camera) from reference to target."""
    ref = np.asarray(reference_extrinsic, dtype=np.float32)
    tgt = np.asarray(target_extrinsic, dtype=np.float32)
    c_ref = camera_center_from_extrinsic(ref)
    c_tgt = camera_center_from_extrinsic(tgt)
    delta_world = c_tgt - c_ref
    delta_in_ref_camera = ref[:, :3] @ delta_world
    relative_rotation = tgt[:, :3] @ ref[:, :3].T
    return relative_rotation.astype(np.float32), delta_in_ref_camera.astype(np.float32)


def rotation_error_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    """Rotation difference in degrees (geodesic distance on SO(3))."""
    delta = np.asarray(rotation_a, dtype=np.float64) @ np.asarray(rotation_b, dtype=np.float64).T
    trace_value = float(np.trace(delta))
    cos_angle = np.clip((trace_value - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def vector_angle_deg(vector_a: np.ndarray, vector_b: np.ndarray, eps: float = 1e-8) -> float:
    """Angle between two vectors in degrees."""
    a = np.asarray(vector_a, dtype=np.float64)
    b = np.asarray(vector_b, dtype=np.float64)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < eps or norm_b < eps:
        return 0.0
    cos_angle = np.clip(float(np.dot(a, b)) / (norm_a * norm_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def relative_pose_comparison(
    target_rotation: np.ndarray,
    target_translation: np.ndarray,
    predicted_rotation: np.ndarray,
    predicted_translation: np.ndarray,
) -> dict[str, float]:
    """Compare predicted pose to target pose; returns error metrics."""
    target_translation = np.asarray(target_translation, dtype=np.float32)
    predicted_translation = np.asarray(predicted_translation, dtype=np.float32)
    target_norm = float(np.linalg.norm(target_translation))
    predicted_norm = float(np.linalg.norm(predicted_translation))
    return {
        "rotation_error_deg": rotation_error_deg(predicted_rotation, target_rotation),
        "translation_l2_normalized": float(np.linalg.norm(predicted_translation - target_translation)),
        "translation_direction_error_deg": vector_angle_deg(predicted_translation, target_translation),
        "target_translation_norm": target_norm,
        "predicted_translation_norm": predicted_norm,
        "translation_norm_ratio_pred_over_target": float(predicted_norm / target_norm) if target_norm > 1e-8 else 0.0,
    }
