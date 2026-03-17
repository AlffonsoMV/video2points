#!/usr/bin/env python3
"""
Iterative orbit-completion pipeline.

Takes a video of an object (partial orbit), reconstructs with VGGT,
then iteratively extends coverage by generating novel views:
  render point cloud → upscale to original resolution → FLUX quality enhance → inpaint holes → re-run VGGT → repeat.

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
from pipeline.output_layout import (
    dir_final,
    dir_frames,
    dir_initial,
    dir_iterations,
    dir_view,
    STRUCTURE_TXT,
)


STEP_DEG = 20.0
GAP_THRESHOLD_DEG = 25.0
MAX_ITERATIONS_DEFAULT = 45


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
    frame_dir = utils.ensure_dir(dir_frames(out_dir))
    frame_paths = utils.extract_n_frames_from_video(
        video_path, n_frames=cfg.n_frames_per_video, output_dir=frame_dir,
    )
    print(f"  Extracted {len(frame_paths)} frames (all used for initial VGGT)")
    for p in frame_paths:
        print(f"    - {p.name}")
    print("  (Renders are pixely because we project a point cloud; more frames + smaller steps = denser over time.)")

    image_paths: list[Path] = list(frame_paths)
    original_frame_size = Image.open(frame_paths[0]).size  # (w, h) at native resolution

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
    pts, cols = utils.merge_scene_point_cloud(
        scene, view_indices=view_indices,
        conf_percentile=cfg.vggt_conf_percentile, max_points=cfg.max_points_render, rng_seed=cfg.seed,
    )
    print(f"  Merged point cloud: {len(pts):,} points")

    fig, _ = utils.plot_point_cloud_3d(
        *utils.merge_scene_point_cloud(
            scene, view_indices=view_indices,
            conf_percentile=cfg.vggt_conf_percentile, max_points=cfg.max_points_plot, rng_seed=cfg.seed,
        ),
        title=f"Initial reconstruction — {n_views} views",
        point_size=0.25,
    )
    utils.ensure_dir(dir_initial(out_dir))
    utils.save_matplotlib_figure(fig, dir_initial(out_dir) / "point_cloud.png")

    if len(pts) == 0:
        print("  ERROR: empty point cloud — cannot continue.")
        return

    image_hw = tuple(int(x) for x in scene["image_hw"])
    utils.ensure_dir(dir_iterations(out_dir))
    generated_count = 0
    manifest_iterations: list[dict] = []

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
        target_angles = utils.pick_next_angles_bilateral(orbit, step_deg)
        if not target_angles:
            print(f"  Gap ({gap_deg:.1f}°) too small for another step — done.")
            break

        labels = ["left", "right"] if len(target_angles) == 2 else ["mid"]
        print(f"  Bilateral step {step_deg}° → {len(target_angles)} target(s): " + ", ".join(
            f"{labels[i]} {np.degrees(a):.1f}°" for i, a in enumerate(target_angles)
        ))

        iter_num = iteration + 1
        iter_views: list[dict] = []
        for view_idx, target_angle in enumerate(target_angles):
            view_dir = utils.ensure_dir(dir_view(out_dir, iter_num, view_idx))

            # 3c. Camera at chosen angle
            novel_extr, novel_intr = utils.camera_at_angle(
                orbit, target_angle, cam_height_offset, ref_intr, extrinsics,
            )

            # 3d. Render point cloud from novel camera
            render = utils.render_projected_point_cloud(
                pts, cols, novel_extr, novel_intr, image_hw, cfg.render_point_radius,
            )
            utils.save_pil(render.image, view_dir / "01_render.png")
            print(f"  Render {generated_count}: projected={render.projected_count:,}, valid_ratio={render.valid_mask.mean():.4f}")

            novel_input, hole_mask, hole_overlay = utils.prepare_novel_view_inpainting_inputs(
                render,
                dilate_px=cfg.mask_dilate_px,
                close_px=cfg.mask_close_px,
                min_area_px=cfg.mask_min_area_px,
                interior_only=True,
                gap_break_px=cfg.mask_gap_break_px,
            )
            utils.save_pil(hole_mask, view_dir / "02_hole_mask.png")
            utils.save_pil(hole_overlay, view_dir / "03_hole_overlay.png")
            hole_ratio = float((np.asarray(hole_mask) > 0).mean())
            print(f"  Hole ratio: {hole_ratio:.4f}")

            # 3e. Resize to original frame resolution
            current_w, current_h = novel_input.size
            orig_w, orig_h = original_frame_size
            if (current_w, current_h) != (orig_w, orig_h):
                novel_input = novel_input.resize(original_frame_size, Image.Resampling.LANCZOS)
                hole_mask = hole_mask.resize(original_frame_size, Image.Resampling.NEAREST)
                utils.save_pil(novel_input, view_dir / "04_upscaled.png")
                print(f"  Upscaled {current_w}x{current_h} → {orig_w}x{orig_h}")

            # 3f. FLUX quality enhancement on rendered content (before inpainting)
            if flux_sharpen:
                utils.clear_loaded_model_caches(clear_vggt=True, clear_inpaint=False)
                t_flux = time.time()
                flux_prompt = (
                    "Enhance this image to photorealistic quality with sharp, crisp details "
                    "and natural textures. Preserve the exact geometry, composition, viewpoint, "
                    "and object identity. Do not add or remove objects."
                )
                novel_input = utils.enhance_quality_with_flux2_klein(
                    novel_input,
                    prompt=flux_prompt,
                    reference_images=frame_paths[:2],
                    device=device,
                    num_inference_steps=8,
                    guidance_scale=1.0,
                    seed=cfg.seed + generated_count,
                )
                utils.save_pil(novel_input, view_dir / "05_flux_quality.png")
                print(f"  FLUX quality enhanced in {time.time() - t_flux:.1f}s")
                utils.clear_loaded_model_caches(clear_vggt=False, clear_inpaint=True)

            # 3g. Inpaint holes
            t_inpaint = time.time()
            if skip_sdxl:
                final_image = utils.opencv_inpaint_fallback(novel_input, hole_mask, radius=5)
                utils.save_pil(final_image, view_dir / "06_inpainted.png")
                print(f"  OpenCV inpainted in {time.time() - t_inpaint:.1f}s")
            else:
                utils.clear_loaded_model_caches(clear_vggt=True, clear_inpaint=False)
                sdxl_result = utils.inpaint_with_diffusion(
                    image=novel_input,
                    mask=hole_mask,
                    prompt=cfg.inpaint_prompt,
                    negative_prompt=cfg.inpaint_negative_prompt,
                    model_id=cfg.inpaint_model_id,
                    device=device,
                    num_inference_steps=max(cfg.inpaint_steps, 20),
                    guidance_scale=8.0,
                    strength=0.99,
                    padding_mask_crop=None,
                    seed=cfg.seed + generated_count,
                    allow_fallback_to_opencv=True,
                )
                final_image = sdxl_result.composited
                utils.save_pil(final_image, view_dir / "06_inpainted.png")
                print(f"  SDXL inpainted (backend={sdxl_result.backend}) in {time.time() - t_inpaint:.1f}s")

            final_path = view_dir / "final.png"
            utils.save_pil(final_image, final_path)
            image_paths.append(final_path)
            iter_views.append({
                "view_idx": view_idx,
                "final_path": str(final_path.resolve()),
                "angle_deg": float(np.degrees(target_angle)),
            })
            generated_count += 1

        manifest_iterations.append({
            "iteration": iter_num,
            "gap_before_deg": gap_deg,
            "views": iter_views,
        })

        # 3h. Re-run VGGT with all images so far
        print(f"  Re-running VGGT with {len(image_paths)} views ...")
        utils.clear_loaded_model_caches(clear_vggt=True, clear_inpaint=True)
        model = utils.load_vggt_model(cfg.vggt_model_id, device=device)
        scene = utils.run_vggt_reconstruction(
            image_paths, model=model, device=device, preprocess_mode=cfg.preprocess_mode,
        )
        del model

        all_indices = list(range(len(image_paths)))
        pts, cols = utils.merge_scene_point_cloud(
            scene, view_indices=all_indices,
            conf_percentile=cfg.vggt_conf_percentile, max_points=cfg.max_points_render, rng_seed=cfg.seed,
        )
        print(f"  Updated point cloud: {len(pts):,} points")
        print(f"  Iteration time: {time.time() - t_iter:.1f}s")

    # -- 4. Final point cloud visualization --------------------------------
    print(f"\n[Final] Saving final reconstruction ({len(image_paths)} views) ...")
    fig_final, _ = utils.plot_point_cloud_3d(
        *utils.merge_scene_point_cloud(
            scene, view_indices=list(range(len(image_paths))),
            conf_percentile=cfg.vggt_conf_percentile, max_points=cfg.max_points_plot, rng_seed=cfg.seed,
        ),
        title=f"Final reconstruction — {len(image_paths)} views",
        point_size=0.25,
    )
    utils.ensure_dir(dir_final(out_dir))
    utils.save_matplotlib_figure(fig_final, dir_final(out_dir) / "point_cloud.png")

    # -- 5. Manifest and structure doc -------------------------------------
    manifest = {
        "pipeline": "run_with_videos",
        "video_path": str(video_path.resolve()),
        "output_root": str(out_dir.resolve()),
        "initial_frames": [str(p.resolve()) for p in frame_paths],
        "final_image_paths": [str(p.resolve()) for p in image_paths],
        "iterations": manifest_iterations,
        "total_views": len(image_paths),
        "generated_views": generated_count,
        "total_time_sec": round(time.time() - t_total, 1),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out_dir / "STRUCTURE.txt").write_text(STRUCTURE_TXT)

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
                        help="Disable FLUX quality enhancement pass after upscaling.")
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
