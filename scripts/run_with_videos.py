#!/usr/bin/env python3
"""
Iterative orbit-completion pipeline.

Takes a video of an object (partial orbit), reconstructs with VGGT,
then iteratively extends coverage by generating novel views:
  render point cloud → SDXL inpaint holes → FLUX sharpen → re-run VGGT → repeat.

Each iteration adds up to two views (left and right into the gap) before
re-running VGGT, halving the number of VGGT runs versus single-view steps.
Advances a small angle (default 12°) per side until the gap is closed. Smaller steps = less for
inpainting to fill = less hallucination. Prompt is optional: without --prompt
the model fills from image context only (recommended to reduce hallucination).

Usage:
    python scripts/run_with_videos.py --video Colosseum.mp4
    python scripts/run_with_videos.py --video Colosseum.mp4 --n-frames 15 --step-deg 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

sys.stdout.reconfigure(line_buffering=True)

import utils


STEP_DEG = 20.0
GAP_THRESHOLD_DEG = 25.0
MAX_ITERATIONS_DEFAULT = 45


def merged_point_cloud(
    scene: dict,
    view_indices: list[int],
    conf_percentile: float,
    max_points: int | None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    pts_all, cols_all = [], []
    for idx in view_indices:
        pts, cols, _ = utils.build_point_cloud_from_scene(
            scene, view_idx=idx, conf_percentile=conf_percentile, max_points=None,
        )
        if len(pts):
            pts_all.append(pts)
            cols_all.append(cols)
    if not pts_all:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    pts = np.concatenate(pts_all, axis=0)
    cols = np.concatenate(cols_all, axis=0)
    if max_points is not None and len(pts) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
    return pts.astype(np.float32), cols.astype(np.float32)


def _camera_at_angle(
    orbit: utils.OrbitInfo,
    target_angle: float,
    cam_height_offset: float,
    ref_intr: np.ndarray,
    extrinsics: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Build novel extrinsic/intrinsic at *target_angle* on the orbit (same logic as main loop)."""
    pos = (
        orbit.centroid
        + cam_height_offset * orbit.normal
        + orbit.radius * (np.cos(target_angle) * orbit.u + np.sin(target_angle) * orbit.v)
    )
    ref_angles = []
    for ext in extrinsics:
        ext = np.asarray(ext, dtype=np.float32)
        center = utils.camera_center_from_extrinsic(ext)
        horiz = (center - orbit.centroid) - np.dot(center - orbit.centroid, orbit.normal) * orbit.normal
        a = float(np.arctan2(np.dot(horiz, orbit.v), np.dot(horiz, orbit.u)))
        ref_angles.append(a)
    ref_angles_arr = np.array(ref_angles)
    diffs = np.abs(np.arctan2(
        np.sin(ref_angles_arr - target_angle),
        np.cos(ref_angles_arr - target_angle),
    ))
    closest_idx = int(np.argmin(diffs))
    closest_ext = np.asarray(extrinsics[closest_idx], dtype=np.float32)
    R_wfc_ref = closest_ext[:, :3].T
    delta = float(target_angle - ref_angles[closest_idx])
    R_delta = utils._rotation_matrix_around_axis(orbit.normal, delta)
    R_wfc_new = (R_delta @ R_wfc_ref).astype(np.float32)
    novel_extr = utils.extrinsic_from_camera_pose(R_wfc_new, pos.astype(np.float32))
    return novel_extr, ref_intr.copy()


def _closest_original_frame_by_angle(
    orbit: utils.OrbitInfo,
    extrinsics: list[np.ndarray],
    frame_paths: list[Path],
    target_angle: float,
) -> Path | None:
    """Return the original frame whose camera angle is closest to target_angle. Single ref minimizes geometry conflict."""
    if not frame_paths or len(extrinsics) < len(frame_paths):
        return None
    ref_angles = []
    for i in range(len(frame_paths)):
        ext = np.asarray(extrinsics[i], dtype=np.float32)
        center = utils.camera_center_from_extrinsic(ext)
        horiz = (center - orbit.centroid) - np.dot(center - orbit.centroid, orbit.normal) * orbit.normal
        a = float(np.arctan2(np.dot(horiz, orbit.v), np.dot(horiz, orbit.u)))
        ref_angles.append(a)
    ref_angles_arr = np.array(ref_angles)
    diffs = np.abs(np.arctan2(np.sin(ref_angles_arr - target_angle), np.cos(ref_angles_arr - target_angle)))
    closest_idx = int(np.argmin(diffs))
    return frame_paths[closest_idx]


def pick_next_angles_bilateral(
    orbit: utils.OrbitInfo,
    step_deg: float,
) -> list[float]:
    """
    Pick up to two target angles: one step from gap start (left) and one from gap end (right).
    When the gap is large enough, both views are added in one iteration before re-running VGGT.
    When the gap is small (< 2*step + margin), only one view at the midpoint is added.
    Returns empty list when the gap is too small for any step.
    """
    gap_deg = np.degrees(orbit.gap_size)
    margin_deg = 1.0
    if gap_deg < margin_deg:
        return []
    step_rad = np.radians(step_deg)
    margin_rad = np.radians(margin_deg)

    # Can we fit both left and right views with step each and margin between them?
    step_actual_deg = min(step_deg, (gap_deg - margin_deg) / 2.0)
    if step_actual_deg < 0.5:
        return []
    step_actual_rad = np.radians(step_actual_deg)

    angle_left = float(orbit.gap_start + step_actual_rad)
    angle_right = float(orbit.gap_end - step_actual_rad)

    if angle_left < angle_right - margin_rad:
        return [angle_left, angle_right]
    # Gap too small for both: add single view at midpoint
    mid = float((orbit.gap_start + orbit.gap_end) / 2.0)
    return [mid]


def run_orbit_pipeline(
    video_path: Path,
    cfg: utils.PipelineConfig,
    out_dir: Path,
    max_iterations: int,
    step_deg: float,
    gap_threshold_deg: float,
    flux_sharpen: bool = True,
    skip_sdxl: bool = False,
) -> None:
    """Run the iterative orbit-completion pipeline for a single video."""
    device = utils.get_device()
    t_total = time.time()

    # -- 1. Extract frames ------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Processing: {video_path.name}")
    print(f"{'=' * 60}")
    print(f"\n[1] Extracting {cfg.n_frames_per_video} frames ...")
    frame_dir = utils.ensure_dir(out_dir / "frames")
    frame_paths = utils.extract_n_frames_from_video(
        video_path, n_frames=cfg.n_frames_per_video, output_dir=frame_dir,
    )
    print(f"  Extracted {len(frame_paths)} frames (all used for initial VGGT)")
    for p in frame_paths:
        print(f"    - {p.name}")
    print("  (Renders are pixely because we project a point cloud; more frames + smaller steps = denser over time.)")

    image_paths: list[Path] = list(frame_paths)

    # -- 2. Initial VGGT reconstruction -----------------------------------
    print(f"\n[2] Running initial VGGT reconstruction ({len(image_paths)} views) ...")
    model = utils.load_vggt_model(cfg.vggt_model_id, device=device)
    scene = utils.run_vggt_reconstruction(
        image_paths, model=model, device=device, preprocess_mode=cfg.preprocess_mode,
    )
    del model
    print(json.dumps(utils.describe_scene(scene), indent=2, default=str))

    n_views = len(image_paths)
    view_indices = list(range(n_views))
    pts, cols = merged_point_cloud(
        scene, view_indices, cfg.vggt_conf_percentile, cfg.max_points_render, cfg.seed,
    )
    print(f"  Merged point cloud: {len(pts):,} points")

    fig, _ = utils.plot_point_cloud_3d(
        *merged_point_cloud(scene, view_indices, cfg.vggt_conf_percentile, cfg.max_points_plot, cfg.seed),
        title=f"Initial reconstruction — {n_views} views",
        point_size=0.25,
    )
    utils.save_matplotlib_figure(fig, out_dir / "initial_point_cloud.png")

    if len(pts) == 0:
        print("  ERROR: empty point cloud — cannot continue.")
        return

    image_hw = tuple(int(x) for x in scene["image_hw"])
    gen_dir = utils.ensure_dir(out_dir / "generated")
    generated_count = 0

    # -- 3. Iterative orbit-filling loop ----------------------------------
    for iteration in range(max_iterations):
        t_iter = time.time()
        print(f"\n{'─' * 50}")
        print(f"  Iteration {iteration + 1}/{max_iterations}  ({len(image_paths)} views so far)")
        print(f"{'─' * 50}")

        # 3a. Re-estimate orbit from current point cloud
        centroid = pts.mean(axis=0)
        n_current = len(image_paths)
        extrinsics = [np.asarray(scene["extrinsic"][i], dtype=np.float32) for i in range(n_current)]
        orbit, cam_height_offset = utils.estimate_horizontal_orbit(extrinsics, centroid, point_cloud=pts)

        gap_deg = float(np.degrees(orbit.gap_size))
        print(f"  Orbit: radius={orbit.radius:.4f}, height={cam_height_offset:.4f}")
        print(f"  Gap:   {gap_deg:.1f}° ({np.degrees(orbit.gap_start):.1f}° → {np.degrees(orbit.gap_end):.1f}°)")

        if gap_deg < gap_threshold_deg:
            print(f"  Gap below {gap_threshold_deg}° — orbit complete!")
            break

        # 3b. Pick next target angles (up to 2: left + right into the gap)
        ref_intr = np.asarray(scene["intrinsic"][0], dtype=np.float32)
        target_angles = pick_next_angles_bilateral(orbit, step_deg)
        if not target_angles:
            print(f"  Gap ({gap_deg:.1f}°) too small for another step — done.")
            break

        labels = ["left", "right"] if len(target_angles) == 2 else ["mid"]
        print(f"  Bilateral step {step_deg}° → {len(target_angles)} target(s): " + ", ".join(
            f"{labels[i]} {np.degrees(a):.1f}°" for i, a in enumerate(target_angles)
        ))

        for target_angle in target_angles:
            # 3c. Camera at chosen angle
            novel_extr, novel_intr = _camera_at_angle(
                orbit, target_angle, cam_height_offset, ref_intr, extrinsics,
            )

            # 3d. Render point cloud from novel camera
            render = utils.render_projected_point_cloud(
                pts, cols, novel_extr, novel_intr, image_hw, cfg.render_point_radius,
            )
            utils.save_pil(render.image, gen_dir / f"render_{generated_count:02d}.png")
            print(f"  Render {generated_count}: projected={render.projected_count:,}, valid_ratio={render.valid_mask.mean():.4f}")

            novel_input, hole_mask, hole_overlay = utils.prepare_novel_view_inpainting_inputs(
                render,
                dilate_px=cfg.mask_dilate_px,
                close_px=cfg.mask_close_px,
                min_area_px=cfg.mask_min_area_px,
                interior_only=True,
            )
            utils.save_pil(hole_overlay, gen_dir / f"hole_overlay_{generated_count:02d}.png")
            hole_ratio = float((np.asarray(hole_mask) > 0).mean())
            print(f"  Hole ratio: {hole_ratio:.4f}")

            # 3e. Inpaint holes
            t_inpaint = time.time()
            if skip_sdxl:
                # OpenCV fallback: fast interpolation, no hallucination
                final_image = utils.opencv_inpaint_fallback(novel_input, hole_mask, radius=5)
                utils.save_pil(final_image, gen_dir / f"inpaint_cv_{generated_count:02d}.png")
                print(f"  OpenCV inpainted in {time.time() - t_inpaint:.1f}s")
            else:
                # SDXL inpainting: fill holes with diffusion model
                utils.clear_loaded_model_caches(clear_vggt=True, clear_inpaint=False)
                sdxl_result = utils.inpaint_with_diffusion(
                    image=novel_input,
                    mask=hole_mask,
                    prompt=cfg.inpaint_prompt,
                    negative_prompt=cfg.inpaint_negative_prompt,
                    model_id=cfg.inpaint_model_id,
                    device=device,
                    num_inference_steps=cfg.inpaint_steps,
                    guidance_scale=cfg.inpaint_guidance_scale,
                    strength=0.95,
                    padding_mask_crop=32,
                    seed=cfg.seed + generated_count,
                    allow_fallback_to_opencv=True,
                )
                final_image = sdxl_result.composited
                utils.save_pil(final_image, gen_dir / f"sdxl_{generated_count:02d}.png")
                print(f"  SDXL inpainted (backend={sdxl_result.backend}) in {time.time() - t_inpaint:.1f}s")

            # 3f. FLUX refinement: sharpen inpainted regions into photorealistic texture
            if flux_sharpen:
                utils.clear_loaded_model_caches(clear_vggt=True, clear_inpaint=False)
                t_flux = time.time()
                flux_prompt = (
                    "Sharpen and enhance this image to be photorealistic. "
                    "The image has blurry or smeared regions — make them crisp and detailed. "
                    "Match the colors, textures, and lighting of the sharp surrounding areas. "
                    "Preserve the exact composition, geometry, and viewpoint. Do not add or remove objects."
                )
                final_image = utils.refine_image_with_flux2_klein(
                    final_image,
                    prompt=flux_prompt,
                    mask=hole_mask,
                    device=device,
                    num_inference_steps=8,
                    guidance_scale=1.0,
                    seed=cfg.seed + generated_count,
                )
                utils.save_pil(final_image, gen_dir / f"flux_{generated_count:02d}.png")
                print(f"  FLUX refined in {time.time() - t_flux:.1f}s")
                utils.clear_loaded_model_caches(clear_vggt=False, clear_inpaint=True)
            inpainted_path = gen_dir / f"inpainted_{generated_count:02d}.png"
            utils.save_pil(final_image, inpainted_path)
            image_paths.append(inpainted_path)
            generated_count += 1

        # 3f. Re-run VGGT with all images so far
        print(f"  Re-running VGGT with {len(image_paths)} views ...")
        utils.clear_loaded_model_caches(clear_vggt=True, clear_inpaint=True)
        model = utils.load_vggt_model(cfg.vggt_model_id, device=device)
        scene = utils.run_vggt_reconstruction(
            image_paths, model=model, device=device, preprocess_mode=cfg.preprocess_mode,
        )
        del model

        all_indices = list(range(len(image_paths)))
        pts, cols = merged_point_cloud(
            scene, all_indices, cfg.vggt_conf_percentile, cfg.max_points_render, cfg.seed,
        )
        print(f"  Updated point cloud: {len(pts):,} points")
        print(f"  Iteration time: {time.time() - t_iter:.1f}s")

    # -- 4. Final point cloud visualization --------------------------------
    print(f"\n[Final] Saving final reconstruction ({len(image_paths)} views) ...")
    fig_final, _ = utils.plot_point_cloud_3d(
        *merged_point_cloud(
            scene, list(range(len(image_paths))),
            cfg.vggt_conf_percentile, cfg.max_points_plot, cfg.seed,
        ),
        title=f"Final reconstruction — {len(image_paths)} views",
        point_size=0.25,
    )
    utils.save_matplotlib_figure(fig_final, out_dir / "final_point_cloud.png")

    print(f"\nDone.")
    print(f"  Original views:  {len(frame_paths)}")
    print(f"  Generated views: {generated_count}")
    print(f"  Total views:     {len(image_paths)}")
    print(f"  Total time:      {time.time() - t_total:.1f}s")
    print(f"  Artifacts in:    {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Iterative orbit-completion pipeline")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/iterative")
    parser.add_argument("--video", type=str, default=None,
                        help="Run on a single video (e.g. Colosseum.mp4). Omit to process all.")
    parser.add_argument("--n-frames", type=int, default=12,
                        help="Frames to extract from video for initial VGGT. More frames = denser point cloud (less pixely).")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS_DEFAULT,
                        help=f"Max iterative steps (default {MAX_ITERATIONS_DEFAULT}).")
    parser.add_argument("--step-deg", type=float, default=STEP_DEG,
                        help=f"Angular step size in degrees (default {STEP_DEG}).")
    parser.add_argument("--gap-threshold", type=float, default=GAP_THRESHOLD_DEG,
                        help=f"Stop when gap falls below this (default {GAP_THRESHOLD_DEG}°).")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Override inpainting prompt (e.g. for Colosseum).")
    parser.add_argument("--no-flux-refine", action="store_true",
                        help="Disable FLUX sharpening pass after inpainting.")
    parser.add_argument("--skip-sdxl", action="store_true",
                        help="Use OpenCV interpolation instead of SDXL for hole filling (faster, less realistic).")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    videos = utils.list_videos(data_dir)
    if not videos:
        raise FileNotFoundError(f"No video files found in {data_dir}")

    if args.video:
        stem = Path(args.video).stem
        videos = [v for v in videos if v.stem == stem or v.name == args.video]
        if not videos:
            raise FileNotFoundError(f"Video '{args.video}' not found in {data_dir}")

    print(f"Found {len(videos)} video(s):")
    for v in videos:
        print(f"  - {v.name}")

    cfg_kw: dict = {
        "data_dir": str(data_dir),
        "output_dir": args.output_dir,
        "preprocess_mode": "crop",
        "vggt_conf_percentile": 55.0,
        "max_points_plot": 20000,
        "max_points_render": 120000,
        "render_point_radius": 1,
        "seed": 0,
        "n_frames_per_video": args.n_frames,
    }
    if args.prompt:
        cfg_kw["inpaint_prompt"] = (
            f"{args.prompt}. "
            "Fill missing regions only, preserve the visible rendered geometry, "
            "keep the camera pose consistent with the target view, and keep "
            "lighting, textures, and surroundings consistent with the reference frames."
        )
    else:
        cfg_kw["inpaint_prompt"] = (
            "Photorealistic image. Fill the masked hole regions to seamlessly continue the visible "
            "geometry, textures, lighting, and colors from the surrounding context. "
            "Preserve the exact composition, perspective, and camera viewpoint. "
            "High detail, professional photography, no artifacts."
        )
    cfg_kw["inpaint_negative_prompt"] = (
        "blurry, distorted, duplicated structures, changed visible regions, artifacts, "
        "bad anatomy, deformed, disfigured, oversaturated, floating objects, cropped"
    )
    cfg = utils.PipelineConfig(**cfg_kw)
    utils.seed_everything(cfg.seed)

    for video_path in videos:
        video_out_dir = utils.ensure_dir(Path(cfg.output_dir) / video_path.stem)
        run_orbit_pipeline(
            video_path, cfg, video_out_dir,
            max_iterations=args.max_iterations,
            step_deg=args.step_deg,
            gap_threshold_deg=args.gap_threshold,
            flux_sharpen=not args.no_flux_refine,
            skip_sdxl=args.skip_sdxl,
        )


if __name__ == "__main__":
    main()
