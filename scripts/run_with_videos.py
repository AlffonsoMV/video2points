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

from math import ceil

from PIL import ImageDraw, ImageOps

import utils
from pipeline.output_layout import (
    dir_final,
    dir_frames,
    dir_initial,
    dir_iterations,
    dir_view,
    STRUCTURE_TXT,
)

ANCHOR_COUNT_DEFAULT = 4
STEP_DEG = 25.0
GAP_THRESHOLD_DEG = 25.0
MAX_ITERATIONS_DEFAULT = 45


def build_reference_collage(
    frame_paths: list[Path],
    output_size: tuple[int, int],
    max_images: int = ANCHOR_COUNT_DEFAULT,
) -> Image.Image | None:
    """Build a tiled collage of reference frames for FLUX context."""
    if max_images <= 0 or not frame_paths:
        return None
    selected = frame_paths[:max_images]
    width, height = output_size
    columns = min(2, len(selected))
    rows = ceil(len(selected) / columns)
    short_side = min(width, height)
    outer_margin = max(short_side // 12, 20)
    gutter = max(short_side // 12, 24)
    border = max(gutter // 3, 4)
    usable_w = max(width - 2 * outer_margin - (columns - 1) * gutter, columns)
    usable_h = max(height - 2 * outer_margin - (rows - 1) * gutter, rows)
    tile_w = max(usable_w // columns, 1)
    tile_h = max(usable_h // rows, 1)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, path in enumerate(selected):
        img = utils.load_pil_rgb(path)
        inner = (max(tile_w - 2 * border, 1), max(tile_h - 2 * border, 1))
        tile = ImageOps.fit(img, inner, method=Image.Resampling.LANCZOS)
        x = outer_margin + (idx % columns) * (tile_w + gutter)
        y = outer_margin + (idx // columns) * (tile_h + gutter)
        draw.rectangle([x, y, x + tile_w - 1, y + tile_h - 1],
                        fill=(245, 245, 245), outline=(0, 0, 0), width=border)
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


def run_orbit_pipeline(
    video_path: Path,
    cfg: utils.PipelineConfig,
    out_dir: Path,
    max_iterations: int,
    step_deg: float,
    gap_threshold_deg: float,
    flux_sharpen: bool = True,
    skip_sdxl: bool = False,
    anchor_count: int = ANCHOR_COUNT_DEFAULT,
) -> None:
    """Run the iterative orbit-completion pipeline for a single video."""
    device = utils.get_device()
    t_total = time.time()

    # Pre-flight: warn if disk space is low (FLUX.1-Fill needs ~33GB)
    free_gb, ok = utils.check_disk_space(out_dir, min_gb=5.0)
    if not ok:
        print(
            f"\n[WARNING] Low disk space: {free_gb:.1f} GB free. "
            "FLUX.1-Fill needs ~33GB. Options:\n"
            "  python -m scripts.clear_hf_cache           # remove SDXL (~14GB)\n"
            "  python -m scripts.clear_hf_cache --remove-partial  # remove partial FLUX (~21GB)\n"
            "  Or use --skip-sdxl for OpenCV-only inpainting (no model download).\n"
        )

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
    prev_generated: list[Path] = []  # generated views from previous iteration (replaced each round)
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
                close_px=cfg.mask_close_px,
                dilate_px=cfg.mask_dilate_px,
                min_area_px=cfg.mask_min_area_px,
                shrink_px=cfg.mask_shrink_px,
            )
            utils.save_pil(hole_mask, view_dir / "02_hole_mask.png")
            hole_overlay_colored = utils.overlay_mask_by_hole_size(
                render.image, hole_mask, min_hole_area=16,
            )
            utils.save_pil(hole_overlay_colored, view_dir / "03_hole_overlay.png")
            hole_ratio = float((np.asarray(hole_mask) > 0).mean())
            print(f"  Hole ratio: {hole_ratio:.4f}")

            # 3e. Upscale: at least original frame res, and short side >= 1024 for FLUX
            FLUX_MIN_SIDE = 1024
            current_w, current_h = novel_input.size
            orig_w, orig_h = original_frame_size
            target_w, target_h = orig_w, orig_h
            short_side = min(target_w, target_h)
            if short_side < FLUX_MIN_SIDE:
                scale = FLUX_MIN_SIDE / short_side
                target_w = int(orig_w * scale)
                target_h = int(orig_h * scale)
            if (current_w, current_h) != (target_w, target_h):
                novel_input = novel_input.resize((target_w, target_h), Image.Resampling.LANCZOS)
                hole_mask = hole_mask.resize((target_w, target_h), Image.Resampling.NEAREST)
                utils.save_pil(novel_input, view_dir / "04_upscaled.png")
                print(f"  Upscaled {current_w}x{current_h} → {target_w}x{target_h}")

            # 3f. Build reference collage from original frames
            collage = build_reference_collage(
                frame_paths, output_size=novel_input.size, max_images=anchor_count,
            )
            flux_refs: list[Image.Image] = []
            if collage is not None:
                flux_refs.append(collage)
                utils.save_pil(collage, view_dir / "reference_collage.png")

            # 3g. FLUX quality enhancement with collage reference
            if flux_sharpen:
                utils.clear_loaded_model_caches(clear_vggt=True, clear_inpaint=False)
                t_flux = time.time()
                flux_prompt = (
                    "Transform this into a stunning 4K ultra-HD photograph, sharp and detailed, "
                    "with smooth continuous surfaces, fine-grained textures, and seamless color gradients. "
                    "Every surface should look like a real-world material — stone, grass, sky, metal — "
                    "with natural micro-detail and soft tonal transitions. "
                    "One of the reference images is a clearly separated collage labeled REFERENCE ONLY — "
                    "use it strictly for style, texture, color, and lighting reference. "
                    "NEVER reproduce the collage layout. "
                    "Preserve the exact geometry, composition, and viewpoint. "
                    "Do not add or remove objects. Keep empty or unmasked regions unchanged."
                ) if flux_refs else (
                    "Transform this into a stunning 4K ultra-HD photograph, sharp and detailed, "
                    "with smooth continuous surfaces, fine-grained textures, and seamless color gradients. "
                    "Every surface should look like a real-world material with natural micro-detail "
                    "and soft tonal transitions. "
                    "Preserve the exact geometry, composition, viewpoint, and object identity. "
                    "Do not add or remove objects."
                )
                novel_input = utils.enhance_quality_with_flux2_klein(
                    novel_input,
                    prompt=flux_prompt,
                    protect_mask=hole_mask,
                    reference_images=flux_refs or None,
                    device=device,
                    num_inference_steps=8,
                    guidance_scale=1.0,
                    seed=cfg.seed + generated_count,
                )
                utils.save_pil(novel_input, view_dir / "05_flux_quality.png")
                print(f"  FLUX quality enhanced in {time.time() - t_flux:.1f}s")
                utils.clear_loaded_model_caches(clear_vggt=False, clear_inpaint=True)

            # 3h. Diffusion inpainting (FLUX Fill) — fill holes only
            t_inpaint = time.time()
            if skip_sdxl or hole_ratio > 0.6:
                if hole_ratio > 0.6:
                    print(f"  Hole ratio {hole_ratio:.2%} too high for diffusion; using OpenCV inpainting")
                final_image = utils.opencv_inpaint_fallback(novel_input, hole_mask, radius=5)
                utils.save_pil(final_image, view_dir / "06_inpainted.png")
                print(f"  OpenCV inpainted in {time.time() - t_inpaint:.1f}s")
            else:
                utils.clear_loaded_model_caches(clear_vggt=True, clear_inpaint=False)
                inpaint_result = utils.inpaint_holes_individually(
                    image=novel_input,
                    mask=hole_mask,
                    prompt=cfg.inpaint_prompt,
                    negative_prompt=cfg.inpaint_negative_prompt,
                    model_id=cfg.inpaint_model_id,
                    device=device,
                    num_inference_steps=max(cfg.inpaint_steps, 50),
                    guidance_scale=cfg.inpaint_guidance_scale,
                    strength=1.0,
                    seed=cfg.seed + generated_count,
                    allow_fallback_to_opencv=True,
                )
                final_image = inpaint_result.composited
                utils.save_pil(final_image, view_dir / "06_inpainted.png")
                print(f"  Inpainted (backend={inpaint_result.backend}) in {time.time() - t_inpaint:.1f}s")

            final_path = view_dir / "final.png"
            utils.save_pil(final_image, final_path)
            iter_views.append({
                "view_idx": view_idx,
                "final_path": str(final_path.resolve()),
                "angle_deg": float(np.degrees(target_angle)),
            })
            generated_count += 1

        # Replace previous iteration's generated views with this iteration's
        curr_generated = [Path(v["final_path"]) for v in iter_views]
        image_paths = list(frame_paths) + curr_generated
        prev_generated = curr_generated

        manifest_iterations.append({
            "iteration": iter_num,
            "gap_before_deg": gap_deg,
            "views": iter_views,
        })

        # 3h. Re-run VGGT with original frames + latest generated views only
        print(f"  Re-running VGGT with {len(frame_paths)} original + {len(curr_generated)} generated = {len(image_paths)} views ...")
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
    parser.add_argument("--inpaint-model", type=str, default=None,
                        help="Inpainting model. Default: FLUX.1-Fill-dev (gated; set HF_TOKEN).")
    parser.add_argument("--no-flux-refine", action="store_true",
                        help="Disable FLUX quality enhancement pass after upscaling.")
    parser.add_argument("--skip-sdxl", action="store_true",
                        help="Use OpenCV interpolation instead of diffusion for hole filling (faster, less realistic).")
    parser.add_argument("--mask-close-px", type=int, default=None,
                        help="Morphological close radius (default 15). Lower = tighter silhouette.")
    parser.add_argument("--mask-dilate-px", type=int, default=None,
                        help="Dilate holes for inpainting overlap (default 3)")
    parser.add_argument("--mask-shrink-px", type=int, default=None,
                        help="Erode silhouette to reduce mask at edges (default: close_px//2). Higher = less mask around object.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for inpainting and other stochastic steps (default 0).")
    parser.add_argument("--anchor-count", type=int, default=ANCHOR_COUNT_DEFAULT,
                        help=f"Number of original frames in the reference collage (default {ANCHOR_COUNT_DEFAULT}). 0 = no collage.")
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
            "extremely sharp. "
            "Photorealistic image, 4K ultra-HD, sharp and detailed. Fill the hole regions "
            "to seamlessly continue the visible geometry, textures, lighting, and colors from the "
            "surrounding context. In focus, in foreground, extremely sharp. Preserve the exact "
            "composition, perspective, and camera viewpoint. High detail, professional photography, no artifacts."
        )
    cfg_kw["inpaint_negative_prompt"] = (
        "blurry, out of focus, depth of field, distorted, duplicated structures"
        "solid color fill, flat color, blank, empty, untextured, "
        "minimal backdrop, monochrome patch, featureless, "
        "blurry, out of focus, depth of field, distorted, duplicated structures, "
        "changed visible regions, artifacts, bad anatomy, deformed, disfigured, "
        "oversaturated, floating objects, cropped"
    )
    if args.mask_close_px is not None:
        cfg_kw["mask_close_px"] = args.mask_close_px
    if args.mask_dilate_px is not None:
        cfg_kw["mask_dilate_px"] = args.mask_dilate_px
    if args.mask_shrink_px is not None:
        cfg_kw["mask_shrink_px"] = args.mask_shrink_px
    if args.seed is not None:
        cfg_kw["seed"] = args.seed
    if args.inpaint_model is not None:
        cfg_kw["inpaint_model_id"] = args.inpaint_model
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
            anchor_count=args.anchor_count,
        )


if __name__ == "__main__":
    main()
