#!/usr/bin/env python3
"""
Orbit-completion pipeline.

Takes a video of an object (partial orbit), runs VGGT to reconstruct the
visible part, estimates the camera orbit, generates novel views in the
angular gap to complete 360 degrees, inpaints them, and re-runs VGGT with
the augmented view set for an improved reconstruction.

Each video in data/ is processed as a separate run.

Usage:
    python scripts/run_with_videos.py
    python scripts/run_with_videos.py --data-dir data --n-frames 7 --n-fill-views 4
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


def run_orbit_pipeline(
    video_path: Path,
    cfg: utils.PipelineConfig,
    out_dir: Path,
) -> None:
    """Run the orbit-completion pipeline for a single video."""
    device = utils.get_device()
    t_total = time.time()

    # -- 1. Extract frames ------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Processing: {video_path.name}")
    print(f"{'=' * 60}")
    print(f"\n[1/7] Extracting {cfg.n_frames_per_video} frames ...")
    frame_dir = utils.ensure_dir(out_dir / "frames")
    frame_paths = utils.extract_n_frames_from_video(
        video_path, n_frames=cfg.n_frames_per_video, output_dir=frame_dir,
    )
    print(f"  Extracted {len(frame_paths)} frames")
    for p in frame_paths:
        print(f"    - {p.name}")

    image_paths: list[Path] = list(frame_paths)

    # -- 2. Initial VGGT reconstruction -----------------------------------
    print(f"\n[2/7] Running initial VGGT reconstruction ({len(image_paths)} views) ...")
    model = utils.load_vggt_model(cfg.vggt_model_id, device=device)
    scene = utils.run_vggt_reconstruction(
        image_paths, model=model, device=device, preprocess_mode=cfg.preprocess_mode,
    )
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

    # -- 3. Estimate orbit ------------------------------------------------
    print(f"\n[3/7] Estimating camera orbit ...")
    centroid = pts.mean(axis=0)
    extrinsics = [np.asarray(scene["extrinsic"][i], dtype=np.float32) for i in range(n_views)]
    orbit = utils.estimate_orbit(extrinsics, centroid)

    print(f"  Orbit radius:  {orbit.radius:.4f}")
    print(f"  Gap size:      {np.degrees(orbit.gap_size):.1f} degrees")
    print(f"  Gap range:     {np.degrees(orbit.gap_start):.1f} -> {np.degrees(orbit.gap_end):.1f} deg")
    print(f"  Camera angles: {[f'{np.degrees(a):.1f}' for a in orbit.angles]}")

    # -- 4. Generate cameras in the gap -----------------------------------
    n_fill = cfg.n_orbit_fill_views
    print(f"\n[4/7] Generating {n_fill} cameras in the orbital gap ...")
    ref_intr = np.asarray(scene["intrinsic"][0], dtype=np.float32)
    novel_cameras = utils.generate_orbit_cameras(orbit, n_fill, ref_intr)
    image_hw = tuple(int(x) for x in scene["image_hw"])
    print(f"  Generated {len(novel_cameras)} novel camera positions")

    # -- 5. Render + inpaint each novel view ------------------------------
    print(f"\n[5/7] Rendering and inpainting {len(novel_cameras)} novel views ...")
    del model
    utils.clear_loaded_model_caches()

    gen_dir = utils.ensure_dir(out_dir / "generated")
    for i, (novel_extr, novel_intr) in enumerate(novel_cameras):
        t_view = time.time()
        print(f"\n  --- Novel view {i + 1}/{len(novel_cameras)} ---")

        render = utils.render_projected_point_cloud(
            pts, cols, novel_extr, novel_intr, image_hw, cfg.render_point_radius,
        )
        utils.save_pil(render.image, gen_dir / f"render_{i:02d}.png")
        print(f"  Rendered: projected={render.projected_count:,}, valid_ratio={render.valid_mask.mean():.4f}")

        # Full mask: regenerate the entire image.
        # The sparse render is passed as the base image (geometry anchor) AND as
        # the first reference image, so the diffusion model uses its spatial
        # layout to constrain shapes while the original frames provide
        # appearance/texture. This avoids preserving low-quality sparse splats.
        full_mask = Image.new("L", render.image.size, 255)
        inpaint_result = utils.inpaint_with_diffusion(
            image=render.image,
            mask=full_mask,
            prompt=cfg.inpaint_prompt,
            negative_prompt=cfg.inpaint_negative_prompt,
            model_id=cfg.inpaint_model_id,
            device=device,
            reference_images=[render.image, *frame_paths[:3]],
            num_inference_steps=cfg.inpaint_steps,
            guidance_scale=cfg.inpaint_guidance_scale,
            seed=cfg.seed,
            allow_fallback_to_opencv=True,
        )
        inpainted_path = gen_dir / f"inpainted_{i:02d}.png"
        utils.save_pil(inpaint_result.raw_generated, inpainted_path)
        image_paths.append(inpainted_path)
        print(f"  Generated (backend={inpaint_result.backend}) in {time.time() - t_view:.1f}s")

    # -- 6. Augmented VGGT reconstruction ---------------------------------
    print(f"\n[6/7] Running augmented VGGT reconstruction ({len(image_paths)} views) ...")
    utils.clear_loaded_model_caches()
    model = utils.load_vggt_model(cfg.vggt_model_id, device=device)
    scene_aug = utils.run_vggt_reconstruction(
        image_paths, model=model, device=device, preprocess_mode=cfg.preprocess_mode,
    )
    print(json.dumps(utils.describe_scene(scene_aug), indent=2, default=str))

    aug_indices = list(range(len(image_paths)))
    pts_aug, cols_aug = merged_point_cloud(
        scene_aug, aug_indices, cfg.vggt_conf_percentile, cfg.max_points_plot, cfg.seed,
    )
    fig_aug, _ = utils.plot_point_cloud_3d(
        pts_aug, cols_aug,
        title=f"Augmented reconstruction — {len(image_paths)} views",
        point_size=0.25,
    )
    utils.save_matplotlib_figure(fig_aug, out_dir / "augmented_point_cloud.png")

    # -- 7. Summary -------------------------------------------------------
    print(f"\n[7/7] Done.")
    print(f"  Original views:  {len(frame_paths)}")
    print(f"  Generated views: {len(novel_cameras)}")
    print(f"  Total views:     {len(image_paths)}")
    print(f"  Total time:      {time.time() - t_total:.1f}s")
    print(f"  Artifacts in:    {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Orbit-completion pipeline")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/iterative")
    parser.add_argument("--n-frames", type=int, default=7)
    parser.add_argument("--n-fill-views", type=int, default=4)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    videos = utils.list_videos(data_dir)
    if not videos:
        raise FileNotFoundError(f"No video files found in {data_dir}")

    print(f"Found {len(videos)} video(s):")
    for v in videos:
        print(f"  - {v.name}")

    cfg = utils.PipelineConfig(
        data_dir=str(data_dir),
        output_dir=args.output_dir,
        preprocess_mode="crop",
        vggt_conf_percentile=55.0,
        max_points_plot=20000,
        max_points_render=120000,
        render_point_radius=1,
        inpaint_model_id="black-forest-labs/FLUX.2-klein-4B",
        inpaint_steps=4,
        inpaint_guidance_scale=1.0,
        seed=0,
        n_frames_per_video=args.n_frames,
        n_orbit_fill_views=args.n_fill_views,
    )
    utils.seed_everything(cfg.seed)

    for video_path in videos:
        video_out_dir = utils.ensure_dir(Path(cfg.output_dir) / video_path.stem)
        run_orbit_pipeline(video_path, cfg, video_out_dir)


if __name__ == "__main__":
    main()
