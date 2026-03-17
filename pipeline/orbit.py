"""Orbit estimation and novel-camera generation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline.geometry import (
    camera_center_from_extrinsic,
    extrinsic_from_camera_pose,
)


@dataclass
class OrbitInfo:
    """Parameters describing the camera orbit around an object."""
    centroid: np.ndarray        # (3,) object centroid in world coords
    normal: np.ndarray          # (3,) orbit-plane normal (consistent with cam "up")
    radius: float               # median distance from centroid to cameras (in-plane)
    u: np.ndarray               # (3,) first basis vector in the orbit plane
    v: np.ndarray               # (3,) second basis vector in the orbit plane
    angles: np.ndarray          # (N,) sorted angles of existing cameras on the orbit
    gap_start: float            # angle (rad) where the largest gap begins
    gap_end: float              # angle (rad) where the largest gap ends
    gap_size: float             # size of the largest gap in radians


def _rotation_matrix_around_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation: 3x3 matrix rotating *angle* radians around *axis*."""
    axis = axis.astype(np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ], dtype=np.float64)
    return (np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)).astype(np.float32)


def _average_camera_up(extrinsics: list[np.ndarray]) -> np.ndarray:
    """Return the average world-space "up" direction across cameras (OpenCV convention)."""
    avg_up = np.zeros(3, dtype=np.float32)
    for e in extrinsics:
        R_wfc = np.asarray(e, dtype=np.float32)[:, :3].T
        avg_up -= R_wfc[:, 1]  # -Y_cam is world up in OpenCV
    norm = float(np.linalg.norm(avg_up))
    if norm < 1e-8:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return (avg_up / norm).astype(np.float32)


def camera_forward_world(extrinsic: np.ndarray) -> np.ndarray:
    """Return the camera's forward (Z) direction in world coordinates."""
    R_wfc = np.asarray(extrinsic, dtype=np.float32)[:, :3].T
    forward = R_wfc[:, 2]
    norm = float(np.linalg.norm(forward))
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (forward / norm).astype(np.float32)


def _estimate_lookat_point(
    extrinsics: list[np.ndarray],
    fallback_centroid: np.ndarray,
) -> np.ndarray:
    """Find the 3D point that best lies on all camera optical axes.

    Uses a least-squares intersection of camera rays.  Falls back to
    *fallback_centroid* if the system is degenerate.
    """
    a = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    for extrinsic in extrinsics:
        cam_center = camera_center_from_extrinsic(
            np.asarray(extrinsic, dtype=np.float32),
        ).astype(np.float64)
        forward = camera_forward_world(extrinsic).astype(np.float64)
        projector = np.eye(3, dtype=np.float64) - np.outer(forward, forward)
        a += projector
        b += projector @ cam_center

    if not np.isfinite(a).all() or np.linalg.norm(a) < 1e-8:
        return np.asarray(fallback_centroid, dtype=np.float32)

    lookat, *_ = np.linalg.lstsq(a, b, rcond=None)
    if not np.isfinite(lookat).all():
        return np.asarray(fallback_centroid, dtype=np.float32)
    return np.asarray(lookat, dtype=np.float32)


def _estimate_world_up_from_points(
    pts: np.ndarray,
    cam_centers: np.ndarray,
) -> np.ndarray:
    """Estimate true world-up from point cloud geometry via PCA.

    For aerial / drone footage the point cloud is roughly planar
    (the scene is viewed from above), so the direction of least
    variance is the true vertical axis.  The sign is chosen so
    that the vector points from the scene toward the cameras
    (i.e. *upward*).
    """
    centered = pts - pts.mean(axis=0)
    cov = (centered.T @ centered) / max(len(centered), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    up = eigenvectors[:, 0].astype(np.float32)

    avg_cam = cam_centers.mean(axis=0)
    centroid = pts.mean(axis=0)
    if np.dot(up, avg_cam - centroid) < 0:
        up = -up
    return up


def estimate_orbit(
    extrinsics: list[np.ndarray] | np.ndarray,
    centroid: np.ndarray,
) -> OrbitInfo:
    """Fit a circular orbit to the camera positions around *centroid*.

    Uses PCA to find the orbit plane, then converts each camera centre to an
    angle on that plane.  Returns an ``OrbitInfo`` describing the orbit and the
    largest angular gap (= the part of the 360 arc not covered by the video).
    """
    cam_centers = np.array(
        [camera_center_from_extrinsic(np.asarray(e, dtype=np.float32)) for e in extrinsics],
        dtype=np.float32,
    )
    centroid = np.asarray(centroid, dtype=np.float32)

    centered = cam_centers - centroid
    _, s, Vt = np.linalg.svd(centered, full_matrices=True)

    u_basis = Vt[0].astype(np.float32)
    v_basis = Vt[1].astype(np.float32)
    normal = Vt[2].astype(np.float32)

    # Ensure the normal is consistent with the average camera "up" direction.
    avg_up = np.zeros(3, dtype=np.float32)
    for e in extrinsics:
        R_wfc = np.asarray(e, dtype=np.float32)[:, :3].T
        avg_up -= R_wfc[:, 1]  # -Y_cam is "up" in OpenCV convention
    avg_up /= max(np.linalg.norm(avg_up), 1e-8)
    if np.dot(normal, avg_up) < 0:
        normal = -normal

    projected = centered - np.outer(centered @ normal, normal)
    radii = np.linalg.norm(projected, axis=1)
    radius = float(np.median(radii))

    angles = np.arctan2(projected @ v_basis, projected @ u_basis).astype(np.float32)
    order = np.argsort(angles)
    angles_sorted = angles[order]

    # Find the largest angular gap (including the wrap-around gap).
    diffs = np.diff(angles_sorted)
    wrap_gap = float(2 * np.pi - (angles_sorted[-1] - angles_sorted[0]))
    all_gaps = np.append(diffs, wrap_gap)
    gap_idx = int(np.argmax(all_gaps))

    if gap_idx < len(diffs):
        gap_start = float(angles_sorted[gap_idx])
        gap_end = float(angles_sorted[gap_idx + 1])
    else:
        gap_start = float(angles_sorted[-1])
        gap_end = float(angles_sorted[0] + 2 * np.pi)

    return OrbitInfo(
        centroid=centroid,
        normal=normal,
        radius=radius,
        u=u_basis,
        v=v_basis,
        angles=angles_sorted,
        gap_start=gap_start,
        gap_end=gap_end,
        gap_size=float(all_gaps[gap_idx]),
    )


def estimate_horizontal_orbit(
    extrinsics: list[np.ndarray] | np.ndarray,
    centroid: np.ndarray,
    point_cloud: np.ndarray | None = None,
) -> tuple[OrbitInfo, float]:
    """Like ``estimate_orbit`` but forces a horizontal orbit plane.

    When *point_cloud* is provided the world-up direction is estimated
    via PCA on the point cloud (the "thin" direction for aerial footage
    is vertical).  Otherwise falls back to the average camera-up heuristic.

    Returns ``(OrbitInfo, camera_height_offset)``.
    """
    extrinsics = [np.asarray(e, dtype=np.float32) for e in extrinsics]
    centroid = _estimate_lookat_point(extrinsics, np.asarray(centroid, dtype=np.float32))

    cam_centers = np.array(
        [camera_center_from_extrinsic(e) for e in extrinsics],
        dtype=np.float32,
    )

    if point_cloud is not None and len(point_cloud) >= 10:
        up_axis = _estimate_world_up_from_points(point_cloud, cam_centers)
    else:
        up_axis = _average_camera_up(extrinsics)
    centered = cam_centers - centroid
    height_offsets = centered @ up_axis
    horizontal = centered - np.outer(height_offsets, up_axis)

    u_basis = horizontal[0].copy()
    u_norm = float(np.linalg.norm(u_basis))
    if u_norm < 1e-6:
        for candidate in horizontal[1:]:
            u_basis = candidate.copy()
            u_norm = float(np.linalg.norm(u_basis))
            if u_norm >= 1e-6:
                break
    if u_norm < 1e-6:
        arb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(arb, up_axis))) > 0.9:
            arb = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        u_basis = arb - np.dot(arb, up_axis) * up_axis
        u_norm = float(np.linalg.norm(u_basis))
    u_basis = (u_basis / max(u_norm, 1e-8)).astype(np.float32)
    v_basis = np.cross(up_axis, u_basis).astype(np.float32)
    v_basis /= max(float(np.linalg.norm(v_basis)), 1e-8)

    radii = np.linalg.norm(horizontal, axis=1)
    radius = float(np.median(radii))

    angles = np.arctan2(horizontal @ v_basis, horizontal @ u_basis).astype(np.float32)
    order = np.argsort(angles)
    angles_sorted = angles[order]

    diffs = np.diff(angles_sorted)
    wrap_gap = float(2 * np.pi - (angles_sorted[-1] - angles_sorted[0]))
    all_gaps = np.append(diffs, wrap_gap)
    gap_idx = int(np.argmax(all_gaps))

    if gap_idx < len(diffs):
        gap_start = float(angles_sorted[gap_idx])
        gap_end = float(angles_sorted[gap_idx + 1])
    else:
        gap_start = float(angles_sorted[-1])
        gap_end = float(angles_sorted[0] + 2 * np.pi)

    orbit = OrbitInfo(
        centroid=centroid,
        normal=up_axis,
        radius=radius,
        u=u_basis,
        v=v_basis,
        angles=angles_sorted,
        gap_start=gap_start,
        gap_end=gap_end,
        gap_size=float(all_gaps[gap_idx]),
    )
    return orbit, float(np.median(height_offsets))


def angular_position_rad(extrinsic: np.ndarray, orbit: OrbitInfo) -> float:
    """Return the angular position of a camera on the orbit plane (radians)."""
    cam_center = camera_center_from_extrinsic(np.asarray(extrinsic, dtype=np.float32))
    centered = cam_center - orbit.centroid
    projected = centered - np.dot(centered, orbit.normal) * orbit.normal
    return float(np.arctan2(float(np.dot(projected, orbit.v)), float(np.dot(projected, orbit.u))))


def angular_distance_rad(a: float, b: float) -> float:
    """Angular distance between two angles in radians."""
    return float(abs(np.arctan2(np.sin(a - b), np.cos(a - b))))


def camera_at_angle(
    orbit: OrbitInfo,
    target_angle: float,
    cam_height_offset: float,
    ref_intr: np.ndarray,
    reference_extrinsics: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Build novel extrinsic/intrinsic at *target_angle* on the orbit.

    Orientation is obtained by rotating the nearest reference camera around the
    orbit normal. Returns (extrinsic, intrinsic).
    """
    pos = (
        orbit.centroid
        + cam_height_offset * orbit.normal
        + orbit.radius * (np.cos(target_angle) * orbit.u + np.sin(target_angle) * orbit.v)
    )
    ref_angles = []
    for ext in reference_extrinsics:
        ext = np.asarray(ext, dtype=np.float32)
        center = camera_center_from_extrinsic(ext)
        horiz = (center - orbit.centroid) - np.dot(center - orbit.centroid, orbit.normal) * orbit.normal
        a = float(np.arctan2(np.dot(horiz, orbit.v), np.dot(horiz, orbit.u)))
        ref_angles.append(a)
    ref_angles_arr = np.array(ref_angles)
    diffs = np.abs(np.arctan2(
        np.sin(ref_angles_arr - target_angle),
        np.cos(ref_angles_arr - target_angle),
    ))
    closest_idx = int(np.argmin(diffs))
    closest_ext = np.asarray(reference_extrinsics[closest_idx], dtype=np.float32)
    R_wfc_ref = closest_ext[:, :3].T
    delta = float(target_angle - ref_angles[closest_idx])
    R_delta = _rotation_matrix_around_axis(orbit.normal, delta)
    R_wfc_new = (R_delta @ R_wfc_ref).astype(np.float32)
    novel_extr = extrinsic_from_camera_pose(R_wfc_new, pos.astype(np.float32))
    return novel_extr, np.asarray(ref_intr, dtype=np.float32).copy()


def pick_next_angles_bilateral(
    orbit: OrbitInfo,
    step_deg: float,
) -> list[float]:
    """
    Pick up to two target angles: one step from gap start (left) and one from gap end (right).
    When the gap is large enough, both views are added in one iteration.
    When the gap is small (< 2*step + margin), only one view at the midpoint is added.
    Returns empty list when the gap is too small for any step.
    """
    gap_deg = np.degrees(orbit.gap_size)
    margin_deg = 1.0
    if gap_deg < margin_deg:
        return []
    step_actual_deg = min(step_deg, (gap_deg - margin_deg) / 2.0)
    if step_actual_deg < 0.5:
        return []
    step_actual_rad = np.radians(step_actual_deg)
    margin_rad = np.radians(margin_deg)

    angle_left = float(orbit.gap_start + step_actual_rad)
    angle_right = float(orbit.gap_end - step_actual_rad)

    if angle_left < angle_right - margin_rad:
        return [angle_left, angle_right]
    mid = float((orbit.gap_start + orbit.gap_end) / 2.0)
    return [mid]


def generate_orbit_cameras(
    orbit: OrbitInfo,
    n_views: int,
    reference_intrinsic: np.ndarray,
    camera_height_offset: float = 0.0,
    reference_extrinsics: list[np.ndarray] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Place *n_views* cameras evenly inside the orbit's largest angular gap.

    When *reference_extrinsics* are provided the orientation of each novel
    camera is obtained by rotating the nearest original camera around the
    orbit normal — this preserves the original pitch and roll exactly,
    which is critical for drone / aerial footage.

    Returns a list of ``(extrinsic, intrinsic)`` pairs.
    """
    if n_views <= 0:
        return []

    new_angles = np.linspace(orbit.gap_start, orbit.gap_end, n_views + 2)[1:-1]

    ref_intr = np.asarray(reference_intrinsic, dtype=np.float32)
    cameras: list[tuple[np.ndarray, np.ndarray]] = []

    ref_data: list[tuple[float, np.ndarray]] | None = None
    if reference_extrinsics:
        ref_data = []
        for ext in reference_extrinsics:
            ext = np.asarray(ext, dtype=np.float32)
            center = camera_center_from_extrinsic(ext)
            horiz = (center - orbit.centroid) - np.dot(center - orbit.centroid, orbit.normal) * orbit.normal
            a = float(np.arctan2(np.dot(horiz, orbit.v), np.dot(horiz, orbit.u)))
            ref_data.append((a, ext))

    for angle in new_angles:
        pos = (
            orbit.centroid
            + camera_height_offset * orbit.normal
            + orbit.radius * (np.cos(angle) * orbit.u + np.sin(angle) * orbit.v)
        )

        if ref_data is not None:
            ref_angles_arr = np.array([rd[0] for rd in ref_data])
            diffs = np.abs(np.arctan2(
                np.sin(ref_angles_arr - angle),
                np.cos(ref_angles_arr - angle),
            ))
            closest_idx = int(np.argmin(diffs))
            closest_angle, closest_ext = ref_data[closest_idx]

            R_wfc_ref = closest_ext[:, :3].T
            delta = float(angle - closest_angle)
            R_delta = _rotation_matrix_around_axis(orbit.normal, delta)
            R_wfc_new = (R_delta @ R_wfc_ref).astype(np.float32)

            extr = extrinsic_from_camera_pose(R_wfc_new, pos.astype(np.float32))
        else:
            z_cam = orbit.centroid - pos
            z_cam = z_cam / max(np.linalg.norm(z_cam), 1e-8)

            x_cam = np.cross(orbit.normal, z_cam)
            x_norm = np.linalg.norm(x_cam)
            if x_norm < 1e-6:
                arb = np.array([1, 0, 0], dtype=np.float32)
                if abs(np.dot(z_cam, arb)) > 0.9:
                    arb = np.array([0, 1, 0], dtype=np.float32)
                x_cam = np.cross(arb, z_cam)
                x_norm = np.linalg.norm(x_cam)
            x_cam = x_cam / x_norm

            y_cam = np.cross(z_cam, x_cam)

            R_wfc = np.column_stack([x_cam, y_cam, z_cam]).astype(np.float32)
            extr = extrinsic_from_camera_pose(R_wfc, pos.astype(np.float32))

        cameras.append((extr, ref_intr.copy()))

    return cameras


# Public aliases for orbit helpers used by run_iterative_loop and others
average_camera_up = _average_camera_up
