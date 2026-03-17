from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import utils


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
DEFAULT_IMAGE_OUTPUT_ROOT = "outputs/iterative"
PLANT_INPAINT_PROMPT = (
    "Photorealistic continuation of the same indoor palm tree scene from the target sparse render viewpoint. "
    "Fill missing regions only, preserve the visible rendered geometry, keep the camera pose consistent with the target view, "
    "and keep the surrounding planter, walls, lighting, and indoor environment strictly consistent with the original anchor reference photos."
)
CHAIR_INPAINT_PROMPT = (
    "Photorealistic continuation of the same chair from the target sparse render viewpoint. "
    "Fill missing regions only, preserve the visible rendered geometry, keep the camera pose consistent with the target view, "
    "and keep the exact same chair design, silhouette, backrest, seat, legs, material, color, lighting, floor contact, and surrounding scene consistent with the reference frames."
)
PYRAMID_INPAINT_PROMPT = (
    "Photorealistic continuation of the same single Great Pyramid of Giza scene from the target sparse render viewpoint. "
    "The visible sparse render is incomplete only because of missing reconstructed points, not because the pyramid is damaged, broken, collapsed, stepped, hollow, or split. "
    "Fill missing regions only, preserve the visible rendered geometry, keep the camera pose consistent with the target view, "
    "and reconstruct one intact, continuous Egyptian pyramid with a clean apex, straight edges, continuous limestone faces, a stable base, and consistent desert terrain, sky, sunlight, and Giza surroundings. "
    "Do not create ruins, openings, cavities, missing chunks, duplicated pyramids, extra monuments, mirrored structures, or fragmented geometry."
)
GENERIC_OBJECT_INPAINT_PROMPT = (
    "Photorealistic continuation of the same object scene from the target sparse render viewpoint. "
    "Fill missing regions only, preserve the visible rendered geometry, keep the camera pose consistent with the target view, "
    "and keep the object identity, proportions, material, lighting, and background consistent with the reference frames."
)


@dataclass
class IterativeLoopConfig:
    max_total_views: int = 30
    start_view_count: int = 0
    max_reference_images: int = 0
    max_iterations: int | None = None
    environment_anchor_count: int = 4
    environment_anchor_start_iteration: int = 1
    step_scale: float = 0.6
    min_step_deg: float = 0.0
    output_root: str = "outputs/iterative"


@dataclass
class OrbitIterationPlan:
    orbit: utils.OrbitInfo
    centroid: np.ndarray
    render_points: np.ndarray
    render_colors: np.ndarray
    camera_height_offset: float
    source_view_index: int
    novel_extrinsic: np.ndarray
    novel_intrinsic: np.ndarray
    target_angle_rad: float
    target_gap_deg: float
    observed_coverage_deg: float
    observed_spacing_deg: float
    max_safe_step_deg: float
    largest_gap_deg: float
    step_from_boundary_deg: float
    boundary_side: str


@dataclass
class BackgroundColorFilter:
    color_rgb: np.ndarray
    distance_threshold: float
    border_px: int
    source_image_count: int


def infer_subject_from_video(video_path: str | Path | None) -> str | None:
    if video_path is None:
        return None
    stem = Path(video_path).stem.lower()
    if "chair" in stem:
        return "chair"
    if "plant" in stem or "palm" in stem:
        return "plant"
    if "pyramid" in stem or "giza" in stem:
        return "pyramid"
    if "colosseum" in stem or "coliseum" in stem:
        return "colosseum"
    return None


def default_output_root(video_path: str | Path | None) -> str:
    if video_path is None:
        return DEFAULT_IMAGE_OUTPUT_ROOT
    return f"outputs/iterative/{Path(video_path).stem}"


def default_inpaint_prompt(subject: str | None, *, video_mode: bool) -> str:
    normalized = (subject or "").strip().lower()
    if normalized == "chair":
        return CHAIR_INPAINT_PROMPT
    if normalized == "plant":
        return PLANT_INPAINT_PROMPT
    if normalized == "pyramid":
        return PYRAMID_INPAINT_PROMPT
    if video_mode:
        return GENERIC_OBJECT_INPAINT_PROMPT
    return PLANT_INPAINT_PROMPT


def resolve_initial_images(
    *,
    data_dir: str | Path,
    video_path: str | Path | None,
    video_frame_count: int,
    output_root: str | Path,
) -> tuple[list[Path], Path | None]:
    if video_path is None:
        return utils.list_data_images(data_dir), None

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    frame_output_dir = utils.ensure_dir(Path(output_root) / "00_seed_frames")
    frame_paths = utils.extract_n_frames_from_video(
        video_path,
        n_frames=video_frame_count,
        output_dir=frame_output_dir,
    )
    return frame_paths, video_path


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    utils.ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2))
    return path


def estimate_background_color_filter(
    image_paths: list[Path],
    distance_threshold: float | None,
    border_px: int,
    max_images: int = 4,
) -> BackgroundColorFilter | None:
    if distance_threshold is None or distance_threshold <= 0 or not image_paths:
        return None

    border_samples: list[np.ndarray] = []
    for path in image_paths[:max_images]:
        image = np.asarray(utils.load_pil_rgb(path), dtype=np.float32)
        h, w = image.shape[:2]
        border = int(max(1, min(border_px, h // 4, w // 4)))
        border_samples.extend(
            [
                image[:border].reshape(-1, 3),
                image[-border:].reshape(-1, 3),
                image[:, :border].reshape(-1, 3),
                image[:, -border:].reshape(-1, 3),
            ]
        )

    if not border_samples:
        return None

    background_rgb = np.median(np.concatenate(border_samples, axis=0), axis=0).astype(np.float32)
    return BackgroundColorFilter(
        color_rgb=background_rgb,
        distance_threshold=float(distance_threshold),
        border_px=int(border_px),
        source_image_count=min(len(image_paths), max_images),
    )


def filter_background_colored_points(
    points: np.ndarray,
    colors: np.ndarray,
    background_filter: BackgroundColorFilter | None,
) -> tuple[np.ndarray, np.ndarray]:
    if background_filter is None or len(points) == 0:
        return points, colors

    colors_arr = np.asarray(colors, dtype=np.float32)
    colors_rgb = colors_arr * 255.0 if colors_arr.size and float(np.nanmax(colors_arr)) <= 1.0 else colors_arr
    distances = np.linalg.norm(colors_rgb - background_filter.color_rgb[None, :], axis=1)
    keep_mask = distances > background_filter.distance_threshold
    if not np.any(keep_mask):
        return points, colors
    return points[keep_mask], colors[keep_mask]


def build_foreground_mask_from_background(
    image: Image.Image,
    background_filter: BackgroundColorFilter | None,
    distance_threshold: float | None = None,
) -> Image.Image:
    image_np = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = image_np.shape[:2]
    if background_filter is None or h == 0 or w == 0:
        return Image.new("L", (w, h), 255)

    threshold = float(
        distance_threshold
        if distance_threshold is not None
        else max(background_filter.distance_threshold + 10.0, 40.0)
    )
    distances = np.linalg.norm(image_np - background_filter.color_rgb[None, None, :], axis=-1)
    background_candidate = (distances <= threshold).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(background_candidate, connectivity=8)
    if num_labels <= 1:
        foreground = np.ones((h, w), dtype=np.uint8) * 255
        return Image.fromarray(foreground, mode="L")

    border_labels = np.unique(
        np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    )
    border_labels = border_labels[border_labels != 0]
    background_connected_to_border = np.isin(labels, border_labels)
    foreground = (~background_connected_to_border).astype(np.uint8) * 255

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, open_kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, close_kernel)

    component_count, component_labels, stats, _ = cv2.connectedComponentsWithStats(
        (foreground > 0).astype(np.uint8),
        connectivity=8,
    )
    if component_count > 1:
        largest_component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        foreground = (component_labels == largest_component).astype(np.uint8) * 255

    foreground = cv2.dilate(foreground, dilate_kernel, iterations=1)
    return Image.fromarray(foreground, mode="L")


def apply_foreground_mask(
    image: Image.Image,
    mask: Image.Image,
    background_fill_rgb: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    image_np = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask_np = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    output = np.full_like(image_np, np.asarray(background_fill_rgb, dtype=np.uint8))
    output[mask_np] = image_np[mask_np]
    return Image.fromarray(output, mode="RGB")


def save_background_masked_image(
    image_path: Path,
    output_path: Path,
    mask_path: Path,
    background_filter: BackgroundColorFilter | None,
    distance_threshold: float | None = None,
) -> tuple[Path, Path]:
    image = utils.load_pil_rgb(image_path)
    mask = build_foreground_mask_from_background(
        image,
        background_filter=background_filter,
        distance_threshold=distance_threshold,
    )
    masked = apply_foreground_mask(image, mask)
    utils.save_pil(masked, output_path)
    utils.save_pil(mask, mask_path)
    return output_path, mask_path


def save_background_masked_images(
    image_paths: list[Path],
    output_dir: Path,
    background_filter: BackgroundColorFilter | None,
    distance_threshold: float | None = None,
) -> tuple[list[Path], list[Path]]:
    output_dir = utils.ensure_dir(output_dir)
    masked_paths: list[Path] = []
    mask_paths: list[Path] = []
    for path in image_paths:
        masked_path, mask_path = save_background_masked_image(
            image_path=Path(path),
            output_path=output_dir / Path(path).name,
            mask_path=output_dir / f"{Path(path).stem}_mask.png",
            background_filter=background_filter,
            distance_threshold=distance_threshold,
        )
        masked_paths.append(masked_path)
        mask_paths.append(mask_path)
    return masked_paths, mask_paths


def build_environment_anchor_sheet(
    image_paths: list[Path],
    output_size: tuple[int, int],
    max_images: int,
) -> Image.Image | None:
    if max_images <= 0 or not image_paths:
        return None

    selected_paths = image_paths[:max_images]
    width, height = output_size
    columns = min(2, len(selected_paths))
    rows = ceil(len(selected_paths) / columns)
    short_side = min(width, height)
    outer_margin = max(short_side // 12, 20)
    gutter = max(short_side // 12, 24)
    border = max(gutter // 3, 4)
    usable_width = max(width - 2 * outer_margin - (columns - 1) * gutter, columns)
    usable_height = max(height - 2 * outer_margin - (rows - 1) * gutter, rows)
    tile_width = max(usable_width // columns, 1)
    tile_height = max(usable_height // rows, 1)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for idx, path in enumerate(selected_paths):
        image = utils.load_pil_rgb(path)
        inner_size = (max(tile_width - 2 * border, 1), max(tile_height - 2 * border, 1))
        tile = ImageOps.fit(image, inner_size, method=Image.Resampling.LANCZOS)
        x = outer_margin + (idx % columns) * (tile_width + gutter)
        y = outer_margin + (idx // columns) * (tile_height + gutter)
        draw.rectangle([x, y, x + tile_width - 1, y + tile_height - 1], fill=(245, 245, 245), outline=(0, 0, 0), width=border)
        canvas.paste(tile, (x + border, y + border))

    label = "REFERENCE ONLY"
    font_size = max(short_side // 18, 14)
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (width - tw) // 2
    ty = height - outer_margin // 2 - th
    draw.text((tx, ty), label, fill=(160, 160, 160), font=font)
    return canvas


def observed_spacing_statistics_deg(sorted_angles_rad: np.ndarray) -> tuple[float, float]:
    if len(sorted_angles_rad) <= 1:
        return 0.0, 0.0
    local_gaps_deg = np.degrees(np.diff(np.asarray(sorted_angles_rad, dtype=np.float32)))
    if len(local_gaps_deg) == 0:
        return 0.0, 0.0
    return float(np.median(local_gaps_deg)), float(np.max(local_gaps_deg))


def choose_reference_images(
    image_paths: list[Path],
    scene: dict[str, Any],
    orbit: utils.OrbitInfo,
    target_angle_rad: float,
    max_reference_images: int,
) -> list[Path]:
    if max_reference_images <= 0:
        return []
    if len(image_paths) <= max_reference_images:
        return list(image_paths)

    scored = []
    for idx, path in enumerate(image_paths):
        angle = utils.angular_position_rad(np.asarray(scene["extrinsic"][idx], dtype=np.float32), orbit)
        scored.append((utils.angular_distance_rad(angle, target_angle_rad), idx, path))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [path for _, _, path in scored[:max_reference_images]]


def scene_summary_payload(scene: dict[str, Any], image_paths: list[Path]) -> dict[str, Any]:
    payload = utils.describe_scene(scene)
    payload["image_paths"] = [str(p) for p in image_paths]
    return payload


def build_iteration_point_cloud(
    scene: dict[str, Any],
    cfg: utils.PipelineConfig,
    rng_seed: int,
    background_filter: BackgroundColorFilter | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    points, colors = utils.merge_scene_point_cloud(
        scene,
        conf_percentile=cfg.vggt_conf_percentile,
        max_points=cfg.max_points_render,
        prefer_depth_unprojection=True,
        rng_seed=rng_seed,
    )
    return filter_background_colored_points(points, colors, background_filter)


def build_camera_on_orbit(
    orbit: utils.OrbitInfo,
    angle: float,
    reference_intrinsic: np.ndarray,
    camera_height_offset: float,
) -> tuple[np.ndarray, np.ndarray]:
    pos = orbit.centroid + camera_height_offset * orbit.normal + orbit.radius * (
        np.cos(angle) * orbit.u + np.sin(angle) * orbit.v
    )
    z_cam = orbit.centroid - pos
    z_cam = z_cam / max(np.linalg.norm(z_cam), 1e-8)

    x_cam = np.cross(z_cam, orbit.normal)
    x_norm = np.linalg.norm(x_cam)
    if x_norm < 1e-6:
        arb = np.array([1, 0, 0], dtype=np.float32)
        if abs(np.dot(z_cam, arb)) > 0.9:
            arb = np.array([0, 1, 0], dtype=np.float32)
        x_cam = np.cross(z_cam, arb)
        x_norm = np.linalg.norm(x_cam)
    x_cam = x_cam / x_norm
    y_cam = np.cross(z_cam, x_cam)

    r_world_from_cam = np.column_stack([x_cam, y_cam, z_cam]).astype(np.float32)
    extrinsic = utils.extrinsic_from_camera_pose(r_world_from_cam, pos.astype(np.float32))
    intrinsic = np.asarray(reference_intrinsic, dtype=np.float32).copy()
    return extrinsic, intrinsic


def plan_next_orbit_view(
    scene: dict[str, Any],
    cfg: utils.PipelineConfig,
    loop_cfg: IterativeLoopConfig,
    iteration_index: int,
    rng_seed: int,
    background_filter: BackgroundColorFilter | None = None,
) -> OrbitIterationPlan:
    pts_render, cols_render = build_iteration_point_cloud(
        scene,
        cfg,
        rng_seed=rng_seed,
        background_filter=background_filter,
    )
    if len(pts_render) == 0:
        raise RuntimeError("Merged point cloud is empty; cannot plan the next camera.")

    centroid = pts_render.mean(axis=0).astype(np.float32)
    extrinsics = [np.asarray(extr, dtype=np.float32) for extr in scene["extrinsic"]]
    orbit, camera_height_offset = utils.estimate_horizontal_orbit(extrinsics, centroid, point_cloud=None)
    raw_angles = [camera_angle_on_orbit(extr, orbit) for extr in extrinsics]

    target_gap_deg = 360.0 / float(loop_cfg.max_total_views)
    largest_gap_deg = float(np.degrees(orbit.gap_size))
    observed_coverage_deg = max(0.0, 360.0 - largest_gap_deg)
    observed_spacing_deg, max_local_spacing_deg = observed_spacing_statistics_deg(orbit.angles)
    max_safe_step_deg = max(1.0, max(observed_spacing_deg, max_local_spacing_deg) * 2.0 * max(loop_cfg.step_scale, 0.1))
    step_cap_deg = min(
        target_gap_deg,
        max(largest_gap_deg / 2.0 - 1.0, 1.0),
    )
    step_from_boundary_deg = min(
        step_cap_deg,
        max(loop_cfg.min_step_deg, min(max_safe_step_deg, step_cap_deg)),
    )
    step_from_boundary_rad = np.radians(step_from_boundary_deg)
    boundary_side = "gap_start" if iteration_index % 2 == 1 else "gap_end"
    boundary_angle = float(orbit.gap_start if boundary_side == "gap_start" else orbit.gap_end)
    if boundary_side == "gap_start":
        target_angle_rad = float(orbit.gap_start + step_from_boundary_rad)
    else:
        target_angle_rad = float(orbit.gap_end - step_from_boundary_rad)

    source_view_index = min(
        range(len(raw_angles)),
        key=lambda idx: angular_distance_rad(raw_angles[idx], boundary_angle),
    )
    novel_extr, novel_intr = build_camera_on_orbit(
        orbit,
        angle=target_angle_rad,
        reference_intrinsic=np.asarray(scene["intrinsic"][0], dtype=np.float32),
        camera_height_offset=camera_height_offset,
    )
    return OrbitIterationPlan(
        orbit=orbit,
        centroid=centroid,
        render_points=pts_render,
        render_colors=cols_render,
        camera_height_offset=camera_height_offset,
        source_view_index=source_view_index,
        novel_extrinsic=np.asarray(novel_extr, dtype=np.float32),
        novel_intrinsic=np.asarray(novel_intr, dtype=np.float32),
        target_angle_rad=target_angle_rad,
        target_gap_deg=target_gap_deg,
        observed_coverage_deg=observed_coverage_deg,
        observed_spacing_deg=observed_spacing_deg,
        max_safe_step_deg=max_safe_step_deg,
        largest_gap_deg=largest_gap_deg,
        step_from_boundary_deg=step_from_boundary_deg,
        boundary_side=boundary_side,
    )


def evaluate_augmented_scene(
    scene_before: dict[str, Any],
    scene_after: dict[str, Any],
    target_extrinsic: np.ndarray,
    previous_orbit: utils.OrbitInfo,
    cfg: utils.PipelineConfig,
    rng_seed: int,
    background_filter: BackgroundColorFilter | None = None,
) -> dict[str, Any]:
    base_extrinsics = np.asarray(scene_before["extrinsic"], dtype=np.float32)
    augmented_extrinsics = np.asarray(scene_after["extrinsic"], dtype=np.float32)
    base_scale = utils.camera_baseline_scale(base_extrinsics, idx_a=0, idx_b=1)
    augmented_scale = utils.camera_baseline_scale(augmented_extrinsics, idx_a=0, idx_b=1)

    target_rotation, target_translation = utils.relative_camera_pose_from_reference(base_extrinsics[0], target_extrinsic)
    predicted_rotation, predicted_translation = utils.relative_camera_pose_from_reference(
        augmented_extrinsics[0],
        augmented_extrinsics[-1],
    )
    target_translation_normalized = target_translation / max(base_scale, 1e-8)
    predicted_translation_normalized = predicted_translation / max(augmented_scale, 1e-8)

    pts_after, _ = build_iteration_point_cloud(
        scene_after,
        cfg,
        rng_seed=rng_seed,
        background_filter=background_filter,
    )
    centroid_after = pts_after.mean(axis=0).astype(np.float32)
    orbit_after, _ = utils.estimate_horizontal_orbit(list(augmented_extrinsics), centroid_after, point_cloud=None)
    predicted_angle = camera_angle_on_orbit(augmented_extrinsics[-1], orbit_after)
    existing_angles = [
        camera_angle_on_orbit(np.asarray(extr, dtype=np.float32), orbit_after)
        for extr in augmented_extrinsics[:-1]
    ]
    min_angle_to_existing_deg = min(
        [float(np.degrees(angular_distance_rad(predicted_angle, angle))) for angle in existing_angles],
        default=180.0,
    )

    largest_gap_after_deg = float(np.degrees(orbit_after.gap_size))
    gap_improvement_deg = float(np.degrees(previous_orbit.gap_size) - largest_gap_after_deg)
    return {
        "largest_gap_after_deg": largest_gap_after_deg,
        "gap_improvement_deg": gap_improvement_deg,
        "predicted_angle_deg": float(np.degrees(predicted_angle)),
        "min_angle_to_existing_deg": min_angle_to_existing_deg,
        "target_pose_error": utils.relative_pose_comparison(
            target_rotation,
            target_translation_normalized,
            predicted_rotation,
            predicted_translation_normalized,
        ),
    }


def save_scene_bundle(
    scene: dict[str, Any],
    image_paths: list[Path],
    out_dir: Path,
    cfg: utils.PipelineConfig,
    rng_seed: int,
    title: str,
    background_filter: BackgroundColorFilter | None = None,
) -> None:
    out_dir = utils.ensure_dir(out_dir)
    pts_plot, cols_plot = utils.merge_scene_point_cloud(
        scene,
        conf_percentile=cfg.vggt_conf_percentile,
        max_points=cfg.max_points_plot,
        prefer_depth_unprojection=True,
        rng_seed=rng_seed,
    )
    pts_plot, cols_plot = filter_background_colored_points(pts_plot, cols_plot, background_filter)
    fig, _ = utils.plot_point_cloud_3d(pts_plot, cols_plot, title=title, point_size=0.25)
    utils.save_matplotlib_figure(fig, out_dir / "point_cloud.png")
    utils.save_point_cloud_ply(pts_plot, cols_plot, out_dir / "point_cloud.ply")
    save_json(out_dir / "scene_summary.json", scene_summary_payload(scene, image_paths))


def save_flux_inputs(
    out_dir: Path,
    base_image: Any,
    reference_images: list[Any],
) -> list[str]:
    out_dir = utils.ensure_dir(out_dir)
    saved_paths: list[str] = []
    base_path = utils.save_pil(base_image, out_dir / "00_pos2_flux_base.png")
    saved_paths.append(str(base_path))
    for idx, ref in enumerate(reference_images):
        if hasattr(ref, "save"):
            img = ref.convert("RGB")
            name = f"{idx + 1:02d}_reference_generated_context.png"
        else:
            img = utils.load_pil_rgb(ref)
            name = f"{idx + 1:02d}_reference_{Path(ref).name}"
        saved_paths.append(str(utils.save_pil(img, out_dir / name)))
    return saved_paths


def render_scene_at_camera(
    points: np.ndarray,
    colors: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_hw: tuple[int, int],
    point_radius: int,
) -> utils.RenderResult:
    return utils.render_projected_point_cloud(
        points,
        colors,
        extrinsic=np.asarray(extrinsic, dtype=np.float32),
        intrinsic=np.asarray(intrinsic, dtype=np.float32),
        image_hw=image_hw,
        point_radius=point_radius,
    )


def generate_iteration(
    scene_before: dict[str, Any],
    image_paths: list[Path],
    anchor_paths: list[Path],
    iteration_dir: Path,
    iteration_index: int,
    pipeline_cfg: utils.PipelineConfig,
    loop_cfg: IterativeLoopConfig,
    orbit_plan: OrbitIterationPlan,
    background_filter: BackgroundColorFilter | None = None,
    mask_inputs_for_vggt: bool = False,
    input_mask_distance_threshold: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    iteration_dir = utils.ensure_dir(iteration_dir)
    inputs_dir = utils.ensure_dir(iteration_dir / "00_inputs")
    before_dir = utils.ensure_dir(iteration_dir / "01_before")
    target_dir = utils.ensure_dir(iteration_dir / "02_target_pos2")
    flux_dir = utils.ensure_dir(iteration_dir / "03_flux")
    after_dir = utils.ensure_dir(iteration_dir / "04_after")

    step_seed = pipeline_cfg.seed + iteration_index
    image_hw = tuple(int(x) for x in scene_before["image_hw"])

    save_scene_bundle(
        scene_before,
        image_paths,
        before_dir,
        pipeline_cfg,
        rng_seed=step_seed,
        title=f"Iteration {iteration_index} Before",
        background_filter=background_filter,
    )

    source_view_path = utils.save_pil(
        utils.load_pil_rgb(image_paths[orbit_plan.source_view_index]),
        before_dir / "source_view_input.png",
    )
    source_render = render_scene_at_camera(
        orbit_plan.render_points,
        orbit_plan.render_colors,
        extrinsic=np.asarray(scene_before["extrinsic"][orbit_plan.source_view_index], dtype=np.float32),
        intrinsic=np.asarray(scene_before["intrinsic"][orbit_plan.source_view_index], dtype=np.float32),
        image_hw=image_hw,
        point_radius=pipeline_cfg.render_point_radius,
    )
    source_render_path = utils.save_pil(source_render.image, before_dir / "source_view_render.png")

    novel_render = render_scene_at_camera(
        orbit_plan.render_points,
        orbit_plan.render_colors,
        extrinsic=orbit_plan.novel_extrinsic,
        intrinsic=orbit_plan.novel_intrinsic,
        image_hw=image_hw,
        point_radius=pipeline_cfg.render_point_radius,
    )
    novel_projection_path = utils.save_pil(novel_render.image, target_dir / "pos2_render_before.png")
    np.save(target_dir / "pos2_extrinsic.npy", orbit_plan.novel_extrinsic)
    np.save(target_dir / "pos2_intrinsic.npy", orbit_plan.novel_intrinsic)

    novel_input, hole_mask, hole_overlay = utils.prepare_novel_view_inpainting_inputs(
        novel_render,
        close_px=pipeline_cfg.mask_close_px,
        dilate_px=pipeline_cfg.mask_dilate_px,
        min_area_px=pipeline_cfg.mask_min_area_px,
        shrink_px=pipeline_cfg.mask_shrink_px,
        exterior_only=pipeline_cfg.mask_exterior_only,
        fill_mask_rgb=pipeline_cfg.mask_fill_rgb,
    )
    novel_input_path = utils.save_pil(novel_input, target_dir / "pos2_flux_base.png")
    hole_mask_path = utils.save_pil(hole_mask, target_dir / "hole_mask.png")
    hole_overlay_path = utils.save_pil(hole_overlay, target_dir / "hole_mask_overlay.png")
    environment_anchor_sheet = build_environment_anchor_sheet(
        anchor_paths,
        output_size=(novel_input.size[0], novel_input.size[1]),
        max_images=loop_cfg.environment_anchor_count,
    )
    environment_anchor_sheet_path = (
        utils.save_pil(environment_anchor_sheet, inputs_dir / "01_environment_anchor_sheet.png")
        if environment_anchor_sheet is not None
        else None
    )

    extra_reference_images = choose_reference_images(
        image_paths=image_paths,
        scene=scene_before,
        orbit=orbit_plan.orbit,
        target_angle_rad=orbit_plan.target_angle_rad,
        max_reference_images=loop_cfg.max_reference_images,
    )
    diagnostic_reference_images: list[Any] = [novel_render.image]
    if environment_anchor_sheet is not None:
        diagnostic_reference_images.append(environment_anchor_sheet)
    diagnostic_reference_images.extend(extra_reference_images)
    flux_input_paths = save_flux_inputs(inputs_dir, novel_input, diagnostic_reference_images)

    hole_ratio = float((np.asarray(hole_mask) > 0).mean())
    # Diffusion inpainting produces degenerate output when hole mask covers too much.
    if hole_ratio > 0.6:
        print(f"  Hole ratio {hole_ratio:.2%} too high for diffusion; using OpenCV inpainting")
        raw = utils.opencv_inpaint_fallback(novel_input, hole_mask, radius=5)
        composited = raw  # OpenCV returns full image; compositing is a no-op
        inpaint = utils.InpaintResult(
            raw_generated=raw,
            composited=composited,
            mask_used=hole_mask,
            backend="opencv_telea_fallback",
            resized_for_model=False,
            metadata={"hole_ratio": hole_ratio, "reason": "high_hole_ratio"},
        )
    else:
        inpaint = utils.inpaint_with_diffusion(
            image=novel_input,
            mask=hole_mask,
            prompt=pipeline_cfg.inpaint_prompt,
            negative_prompt=pipeline_cfg.inpaint_negative_prompt,
            model_id=pipeline_cfg.inpaint_model_id,
            device=utils.get_device(),
            num_inference_steps=pipeline_cfg.inpaint_steps,
            guidance_scale=pipeline_cfg.inpaint_guidance_scale,
            seed=step_seed,
            allow_fallback_to_opencv=False,
        )
    generated_raw_path = utils.save_pil(inpaint.raw_generated, flux_dir / "generated_raw.png")
    generated_composited_path = utils.save_pil(inpaint.composited, flux_dir / "generated_composited.png")

    generated_vggt_input_path = generated_raw_path
    generated_vggt_mask_path: Path | None = None
    if mask_inputs_for_vggt:
        generated_vggt_input_path, generated_vggt_mask_path = save_background_masked_image(
            image_path=generated_raw_path,
            output_path=flux_dir / "generated_masked_for_vggt.png",
            mask_path=flux_dir / "generated_mask_for_vggt.png",
            background_filter=background_filter,
            distance_threshold=input_mask_distance_threshold,
        )

    augmented_inputs = [*image_paths, generated_vggt_input_path]
    utils.clear_loaded_model_caches(clear_vggt=True, clear_inpaint=True, clear_lpips=False)
    model = utils.load_vggt_model(pipeline_cfg.vggt_model_id, device=utils.get_device())
    scene_after = utils.run_vggt_reconstruction(
        augmented_inputs,
        model=model,
        device=utils.get_device(),
        preprocess_mode=pipeline_cfg.preprocess_mode,
        model_id=pipeline_cfg.vggt_model_id,
    )
    del model
    utils.clear_loaded_model_caches(clear_vggt=True, clear_inpaint=True, clear_lpips=False)

    save_scene_bundle(
        scene_after,
        augmented_inputs,
        after_dir,
        pipeline_cfg,
        rng_seed=step_seed,
        title=f"Iteration {iteration_index} After",
        background_filter=background_filter,
    )
    after_points, after_colors = build_iteration_point_cloud(
        scene_after,
        pipeline_cfg,
        rng_seed=step_seed,
        background_filter=background_filter,
    )
    rerender_pos2 = render_scene_at_camera(
        after_points,
        after_colors,
        extrinsic=orbit_plan.novel_extrinsic,
        intrinsic=orbit_plan.novel_intrinsic,
        image_hw=image_hw,
        point_radius=pipeline_cfg.render_point_radius,
    )
    rerender_pos2_path = utils.save_pil(rerender_pos2.image, after_dir / "pos2_render_after.png")

    evaluation = evaluate_augmented_scene(
        scene_before=scene_before,
        scene_after=scene_after,
        target_extrinsic=orbit_plan.novel_extrinsic,
        previous_orbit=orbit_plan.orbit,
        cfg=pipeline_cfg,
        rng_seed=step_seed,
        background_filter=background_filter,
    )
    stats = {
        "iteration_index": iteration_index,
        "seed": step_seed,
        "device": utils.get_device(),
        "backend": inpaint.backend,
        "resized_for_model": inpaint.resized_for_model,
        "input_count_before_generation": len(image_paths),
        "input_count_after_generation": len(augmented_inputs),
        "source_view": {
            "index": orbit_plan.source_view_index,
            "path": str(image_paths[orbit_plan.source_view_index]),
            "saved_input_path": str(source_view_path),
            "saved_render_path": str(source_render_path),
        },
        "coverage_before_generation": {
            "largest_gap_deg": orbit_plan.largest_gap_deg,
            "observed_coverage_deg": orbit_plan.observed_coverage_deg,
            "observed_spacing_deg": orbit_plan.observed_spacing_deg,
            "max_safe_step_deg": orbit_plan.max_safe_step_deg,
            "target_gap_deg": orbit_plan.target_gap_deg,
            "step_from_boundary_deg": orbit_plan.step_from_boundary_deg,
            "boundary_side": orbit_plan.boundary_side,
            "target_angle_deg": float(np.degrees(orbit_plan.target_angle_rad)),
            "camera_angles_deg": [float(np.degrees(a)) for a in orbit_plan.orbit.angles],
            "orbit_radius": float(orbit_plan.orbit.radius),
        },
        "projection_stats_before": {
            "projected_count": int(novel_render.projected_count),
            "valid_ratio": float(novel_render.valid_mask.mean()),
            "hole_ratio": float((np.asarray(hole_mask) > 0).mean()),
        },
        "flux_inputs": flux_input_paths,
        "environment_anchor_paths": [str(p) for p in anchor_paths],
        "environment_anchor_sheet_path": str(environment_anchor_sheet_path) if environment_anchor_sheet_path else None,
        "flux_output": {
            "raw_path": str(generated_raw_path),
            "composited_path": str(generated_composited_path),
            "vggt_input_path": str(generated_vggt_input_path),
            "vggt_input_mask_path": str(generated_vggt_mask_path) if generated_vggt_mask_path is not None else None,
            "metadata": inpaint.metadata,
        },
        "pose_and_coverage_after": evaluation,
        "paths": {
            "before_point_cloud_ply": str(before_dir / "point_cloud.ply"),
            "before_point_cloud_png": str(before_dir / "point_cloud.png"),
            "pos2_render_before": str(novel_projection_path),
            "pos2_flux_base": str(novel_input_path),
            "hole_mask": str(hole_mask_path),
            "hole_mask_overlay": str(hole_overlay_path),
            "generated_raw": str(generated_raw_path),
            "generated_composited": str(generated_composited_path),
            "after_point_cloud_ply": str(after_dir / "point_cloud.ply"),
            "after_point_cloud_png": str(after_dir / "point_cloud.png"),
            "pos2_render_after": str(rerender_pos2_path),
        },
    }
    stats_path = save_json(iteration_dir / "stats.json", stats)
    manifest_item = {
        "iteration_index": iteration_index,
        "stats_path": str(stats_path),
        "generated_raw_path": str(generated_raw_path),
        "generated_vggt_input_path": str(generated_vggt_input_path),
        "largest_gap_before_deg": orbit_plan.largest_gap_deg,
        "largest_gap_after_deg": evaluation["largest_gap_after_deg"],
        "gap_improvement_deg": evaluation["gap_improvement_deg"],
        "min_angle_to_existing_deg": evaluation["min_angle_to_existing_deg"],
    }
    return manifest_item, scene_after, generated_vggt_input_path


def save_final_reconstruction(
    scene: dict[str, Any],
    image_paths: list[Path],
    output_root: Path,
    cfg: utils.PipelineConfig,
    background_filter: BackgroundColorFilter | None = None,
) -> dict[str, Any]:
    final_dir = utils.ensure_dir(output_root / "final")
    save_scene_bundle(
        scene,
        image_paths,
        final_dir,
        cfg,
        rng_seed=cfg.seed,
        title="Final Iterative Reconstruction",
        background_filter=background_filter,
    )
    return scene_summary_payload(scene, image_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iteratively generate synthetic novel views and save clean per-step artifacts.")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--video-path", type=str, default=None)
    parser.add_argument("--video-frame-count", type=int, default=4)
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--inpaint-prompt", type=str, default=None)
    parser.add_argument("--background-color-distance", type=float, default=None)
    parser.add_argument("--background-border-px", type=int, default=24)
    parser.add_argument("--disable-input-background-masking", action="store_true")
    parser.add_argument("--input-background-mask-distance", type=float, default=None)
    parser.add_argument("--mask-exterior-only", action="store_true")
    parser.add_argument("--mask-close-px", type=int, default=None,
                        help="Morphological close radius (default 15). Lower = tighter silhouette, less mask around object.")
    parser.add_argument("--mask-dilate-px", type=int, default=None,
                        help="Dilate holes for inpainting overlap (default 3)")
    parser.add_argument("--mask-shrink-px", type=int, default=None,
                        help="Erode silhouette to reduce mask at edges (default: close_px//2). Higher = less mask around object.")
    parser.add_argument("--max-total-views", type=int, default=30)
    parser.add_argument("--start-view-count", type=int, default=0)
    parser.add_argument("--max-reference-images", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--environment-anchor-count", type=int, default=4)
    parser.add_argument("--environment-anchor-start-iteration", type=int, default=1)
    parser.add_argument("--step-scale", type=float, default=0.6,
                        help="Scale factor for angular step size (default 0.6). Lower = smaller steps, more stable.")
    parser.add_argument("--min-step-deg", type=float, default=0.0)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for inpainting and other stochastic steps (default 1).")
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    resolved_output_root = args.output_root or default_output_root(args.video_path)
    inferred_subject = args.subject or infer_subject_from_video(args.video_path)
    loop_cfg = IterativeLoopConfig(
        max_total_views=args.max_total_views,
        start_view_count=args.start_view_count,
        max_reference_images=args.max_reference_images,
        max_iterations=args.max_iterations,
        environment_anchor_count=args.environment_anchor_count,
        environment_anchor_start_iteration=args.environment_anchor_start_iteration,
        step_scale=args.step_scale,
        min_step_deg=args.min_step_deg,
        output_root=resolved_output_root,
    )
    pipeline_cfg = utils.PipelineConfig(
        data_dir=str(Path(args.video_path).parent) if args.video_path else args.data_dir,
        preprocess_mode="crop",
        vggt_conf_percentile=55.0,
        max_points_plot=20000,
        max_points_render=120000,
        render_point_radius=2,
        inpaint_model_id="black-forest-labs/FLUX.1-Fill-dev",
        inpaint_steps=50,
        inpaint_guidance_scale=30.0,
        inpaint_prompt=args.inpaint_prompt or default_inpaint_prompt(inferred_subject, video_mode=args.video_path is not None),
        seed=args.seed if args.seed is not None else 1,
        n_frames_per_video=args.video_frame_count,
    )
    if args.mask_exterior_only or inferred_subject == "pyramid":
        pipeline_cfg.mask_exterior_only = True
    if inferred_subject == "pyramid":
        pipeline_cfg.mask_fill_rgb = (255, 255, 255)
    if args.mask_close_px is not None:
        pipeline_cfg.mask_close_px = args.mask_close_px
    if args.mask_dilate_px is not None:
        pipeline_cfg.mask_dilate_px = args.mask_dilate_px
    if args.mask_shrink_px is not None:
        pipeline_cfg.mask_shrink_px = args.mask_shrink_px

    utils.seed_everything(pipeline_cfg.seed)
    output_root = utils.ensure_dir(loop_cfg.output_root)
    device = utils.get_device()

    data_images, resolved_video_path = resolve_initial_images(
        data_dir=pipeline_cfg.data_dir,
        video_path=args.video_path,
        video_frame_count=args.video_frame_count,
        output_root=output_root,
    )
    resolved_start_view_count = loop_cfg.start_view_count if loop_cfg.start_view_count > 0 else len(data_images)
    if len(data_images) < resolved_start_view_count:
        raise RuntimeError(
            f"Need at least {resolved_start_view_count} images in {pipeline_cfg.data_dir}, found: {data_images}"
        )

    raw_initial_paths = [Path(p) for p in data_images[:resolved_start_view_count]]
    background_color_distance = args.background_color_distance
    if background_color_distance is None and inferred_subject == "chair":
        background_color_distance = 30.0
    background_filter = estimate_background_color_filter(
        raw_initial_paths,
        distance_threshold=background_color_distance,
        border_px=args.background_border_px,
    )
    mask_inputs_for_vggt = inferred_subject == "chair" and not args.disable_input_background_masking
    input_mask_distance_threshold = args.input_background_mask_distance

    current_paths = raw_initial_paths
    input_mask_paths: list[Path] = []
    if mask_inputs_for_vggt:
        current_paths, input_mask_paths = save_background_masked_images(
            raw_initial_paths,
            output_dir=output_root / "01_masked_seed_frames",
            background_filter=background_filter,
            distance_threshold=input_mask_distance_threshold,
        )
    anchor_paths = list(current_paths)
    manifest: dict[str, Any] = {
        "device": device,
        "pipeline_config": pipeline_cfg.__dict__,
        "loop_config": loop_cfg.__dict__,
        "subject": inferred_subject,
        "initial_video_path": str(resolved_video_path) if resolved_video_path is not None else None,
        "input_background_masking": {
            "enabled": mask_inputs_for_vggt,
            "distance_threshold": input_mask_distance_threshold,
            "masked_seed_images": [str(p) for p in current_paths] if mask_inputs_for_vggt else [],
            "masked_seed_masks": [str(p) for p in input_mask_paths] if mask_inputs_for_vggt else [],
        },
        "background_filter": (
            {
                "color_rgb": [float(x) for x in background_filter.color_rgb],
                "distance_threshold": background_filter.distance_threshold,
                "border_px": background_filter.border_px,
                "source_image_count": background_filter.source_image_count,
            }
            if background_filter is not None
            else None
        ),
        "resolved_start_view_count": resolved_start_view_count,
        "raw_initial_images": [str(p) for p in raw_initial_paths],
        "initial_images": [str(p) for p in current_paths],
        "environment_anchor_paths": [str(p) for p in anchor_paths],
        "iterations": [],
    }
    manifest_path = output_root / "manifest.json"
    save_json(manifest_path, manifest)

    print(f"Device: {device}")
    if resolved_video_path is not None:
        print(f"Seed video: {resolved_video_path}")
        print(f"Extracted seed frames: {len(data_images)}")
    if background_filter is not None:
        print(
            "Background color filter:"
            f" rgb={[round(float(x), 1) for x in background_filter.color_rgb]},"
            f" threshold={background_filter.distance_threshold:.1f},"
            f" border_px={background_filter.border_px}"
        )
    if mask_inputs_for_vggt:
        print(
            "Input background masking enabled:"
            f" mask_distance={input_mask_distance_threshold if input_mask_distance_threshold is not None else 'auto'}"
        )
    print("Initial images:")
    for p in current_paths:
        print(" -", p)

    utils.clear_loaded_model_caches(clear_vggt=True, clear_inpaint=True, clear_lpips=False)
    model = utils.load_vggt_model(pipeline_cfg.vggt_model_id, device=device)
    scene = utils.run_vggt_reconstruction(
        current_paths,
        model=model,
        device=device,
        preprocess_mode=pipeline_cfg.preprocess_mode,
        model_id=pipeline_cfg.vggt_model_id,
    )
    del model
    utils.clear_loaded_model_caches(clear_vggt=True, clear_inpaint=True, clear_lpips=False)

    iteration_index = 0
    while len(current_paths) < loop_cfg.max_total_views:
        if loop_cfg.max_iterations is not None and iteration_index >= loop_cfg.max_iterations:
            manifest["stop_reason"] = "max_iterations_reached"
            break

        iteration_index += 1
        iteration_dir = utils.ensure_dir(output_root / f"iter_{iteration_index:02d}")
        print(f"\n[iter {iteration_index}] planning with {len(current_paths)} views...")

        orbit_plan = plan_next_orbit_view(
            scene,
            pipeline_cfg,
            loop_cfg,
            iteration_index=iteration_index,
            rng_seed=pipeline_cfg.seed + iteration_index,
            background_filter=background_filter,
        )
        print(
            "  orbit coverage:"
            f" largest_gap={orbit_plan.largest_gap_deg:.1f} deg,"
            f" observed={orbit_plan.observed_coverage_deg:.1f} deg,"
            f" target<={orbit_plan.target_gap_deg:.1f} deg,"
            f" safe_step<={orbit_plan.max_safe_step_deg:.1f} deg,"
            f" step={orbit_plan.step_from_boundary_deg:.1f} deg from {orbit_plan.boundary_side}"
        )

        active_anchor_paths = (
            anchor_paths[: loop_cfg.environment_anchor_count]
            if iteration_index >= loop_cfg.environment_anchor_start_iteration
            else []
        )
        step_payload, next_scene, generated_raw_path = generate_iteration(
            scene_before=scene,
            image_paths=current_paths,
            anchor_paths=active_anchor_paths,
            iteration_dir=iteration_dir,
            iteration_index=iteration_index,
            pipeline_cfg=pipeline_cfg,
            loop_cfg=loop_cfg,
            orbit_plan=orbit_plan,
            background_filter=background_filter,
            mask_inputs_for_vggt=mask_inputs_for_vggt,
            input_mask_distance_threshold=input_mask_distance_threshold,
        )
        current_paths.append(generated_raw_path)
        scene = next_scene
        manifest["iterations"].append(step_payload)
        manifest["current_image_paths"] = [str(p) for p in current_paths]
        save_json(manifest_path, manifest)

    if "stop_reason" not in manifest:
        manifest["stop_reason"] = "max_total_views_reached"
    manifest["final_scene_summary"] = save_final_reconstruction(
        scene,
        current_paths,
        output_root,
        pipeline_cfg,
        background_filter=background_filter,
    )
    manifest["final_image_paths"] = [str(p) for p in current_paths]
    save_json(manifest_path, manifest)

    print("\nIterative loop finished.")
    print(f"Total views: {len(current_paths)}")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
