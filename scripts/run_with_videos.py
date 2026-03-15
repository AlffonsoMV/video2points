#!/usr/bin/env python3
"""
Run the full pipeline using videos from data/.

Extracts one frame per video (middle frame by default), then runs:
- Mode "full": 2 frames -> VGGT -> novel view -> inpaint -> 3-view VGGT (original pipeline)
- Mode "direct": all 3 frames -> VGGT directly (no inpainting, uses real views only)

Usage:
    python scripts/run_with_videos.py [--mode full|direct] [--data-dir data]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Ensure project root is on path so we import local utils, not a vendored one
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import utils


def _list_videos(data_dir: str | Path) -> list[Path]:
    """Return sorted paths to video files in data_dir."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    exts = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}
    return sorted([p for p in data_dir.glob("*") if p.is_file() and p.suffix in exts])


def _extract_frames_from_videos(
    data_dir: str | Path,
    frame_index: str = "middle",
    output_stem: str = "image",
) -> list[Path]:
    """Extract one frame per video, save as image_01.png, image_02.png, etc."""
    import cv2
    from PIL import Image

    videos = _list_videos(data_dir)
    if not videos:
        raise FileNotFoundError(f"No video files found in {data_dir}")
    data_dir = Path(data_dir)
    saved: list[Path] = []
    for i, v in enumerate(videos, start=1):
        out_path = data_dir / f"{output_stem}_{i:02d}.png"
        cap = cv2.VideoCapture(str(v))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {v}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            raise RuntimeError(f"Video has no frames: {v}")
        if frame_index == "first":
            idx = 0
        elif frame_index == "last":
            idx = total - 1
        else:
            idx = total // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Cannot read frame {idx} from {v}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        utils.save_pil(Image.fromarray(frame, mode="RGB"), out_path)
        saved.append(out_path)
    return saved


def _list_data_images(data_dir: str | Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    return sorted([p for p in Path(data_dir).glob("*") if p.is_file() and p.suffix in exts])


def _merged_scene_point_cloud(
    scene: dict,
    view_indices: list[int],
    conf_percentile: float,
    max_points: int | None,
    prefer_depth_unprojection: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    pts_all, cols_all = [], []
    for idx in view_indices:
        pts, cols, _ = utils.build_point_cloud_from_scene(
            scene,
            view_idx=idx,
            conf_percentile=conf_percentile,
            max_points=None,
            prefer_depth_unprojection=prefer_depth_unprojection,
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
        pts = pts[idx]
        cols = cols[idx]
    return pts.astype(np.float32), cols.astype(np.float32)


def run_full_pipeline(
    src_imgs: list[Path],
    cfg: utils.PipelineConfig,
    out_dir: Path,
) -> None:
    """Original pipeline: 2 views -> novel view -> inpaint -> 3-view VGGT."""
    device = utils.get_device()
    src_imgs = [utils.ensure_png_copy(p) for p in src_imgs[:2]]

    print("\n[1/6] Running initial two-view VGGT...")
    model = utils.load_vggt_model(cfg.vggt_model_id, device=device)
    scene1 = utils.run_vggt_reconstruction(
        src_imgs, model=model, device=device, preprocess_mode=cfg.preprocess_mode
    )
    print(json.dumps(utils.describe_scene(scene1), indent=2, default=str))

    pts1, cols1 = _merged_scene_point_cloud(
        scene1, list(range(len(src_imgs))), cfg.vggt_conf_percentile,
        cfg.max_points_plot, True, cfg.seed
    )
    fig1, _ = utils.plot_point_cloud_3d(pts1, cols1, title="Initial Two-view VGGT Reconstruction", point_size=0.25)
    utils.save_matplotlib_figure(fig1, out_dir / "01_initial_point_cloud.png")

    pts_render, cols_render = _merged_scene_point_cloud(
        scene1, list(range(len(src_imgs))), cfg.vggt_conf_percentile,
        cfg.max_points_render, True, cfg.seed
    )
    base_extr = np.asarray(scene1["extrinsic"][0], dtype=np.float32)
    base_intr = np.asarray(scene1["intrinsic"][0], dtype=np.float32)
    base_hw = tuple(int(x) for x in scene1["image_hw"])

    print("\n[2/6] Rendering novel view...")
    novel_extr = utils.perturb_camera_extrinsic(
        base_extr, shift_right=cfg.novel_shift_right, yaw_deg=cfg.novel_yaw_deg,
        pitch_deg=cfg.novel_pitch_deg, roll_deg=cfg.novel_roll_deg,
    )
    novel_render = utils.render_projected_point_cloud(
        pts_render, cols_render, novel_extr, base_intr, base_hw, cfg.render_point_radius
    )
    utils.save_pil(novel_render.image, out_dir / "03_novel_view_projection.png")

    print("\n[3/6] Building hole mask...")
    novel_img, hole_mask, overlay = utils.prepare_novel_view_inpainting_inputs(
        novel_render, dilate_px=cfg.mask_dilate_px, close_px=cfg.mask_close_px
    )
    utils.save_pil(hole_mask, out_dir / "04_hole_mask.png")

    print("\n[4/6] Inpainting...")
    del model
    utils.clear_loaded_model_caches()
    inpaint = utils.inpaint_with_diffusion(
        image=novel_img,
        mask=hole_mask,
        prompt=cfg.inpaint_prompt,
        negative_prompt=cfg.inpaint_negative_prompt,
        model_id=cfg.inpaint_model_id,
        device=device,
        reference_images=src_imgs,
        num_inference_steps=cfg.inpaint_steps,
        guidance_scale=cfg.inpaint_guidance_scale,
        seed=cfg.seed,
        allow_fallback_to_opencv=True,
    )
    utils.save_pil(inpaint.composited, out_dir / "05_inpaint_composited.png")
    inpainted_path = out_dir / "05_inpaint_composited.png"

    print("\n[5/6] Augmented VGGT (2 originals + inpainted)...")
    utils.clear_loaded_model_caches()
    model2 = utils.load_vggt_model(cfg.vggt_model_id, device=device)
    scene2 = utils.run_vggt_reconstruction(
        [*src_imgs, inpainted_path],
        model=model2, device=device, preprocess_mode=cfg.preprocess_mode, model_id=cfg.vggt_model_id
    )

    pts2, cols2 = _merged_scene_point_cloud(
        scene2, list(range(3)), cfg.vggt_conf_percentile, cfg.max_points_plot, True, cfg.seed
    )
    fig2, _ = utils.plot_point_cloud_3d(pts2, cols2, title="Augmented 3-view Reconstruction", point_size=0.25)
    utils.save_matplotlib_figure(fig2, out_dir / "06_augmented_point_cloud.png")

    print("\n[6/6] Done.")
    print(f"Outputs in: {out_dir.resolve()}")


def run_direct_pipeline(
    src_imgs: list[Path],
    cfg: utils.PipelineConfig,
    out_dir: Path,
) -> None:
    """Use all real views directly: no inpainting."""
    device = utils.get_device()
    src_imgs = [utils.ensure_png_copy(p) for p in src_imgs]

    print("\n[1/2] Running VGGT on all views...")
    model = utils.load_vggt_model(cfg.vggt_model_id, device=device)
    scene = utils.run_vggt_reconstruction(
        src_imgs, model=model, device=device, preprocess_mode=cfg.preprocess_mode
    )
    print(json.dumps(utils.describe_scene(scene), indent=2, default=str))

    pts, cols = _merged_scene_point_cloud(
        scene, list(range(len(src_imgs))), cfg.vggt_conf_percentile,
        cfg.max_points_plot, True, cfg.seed
    )
    fig, _ = utils.plot_point_cloud_3d(
        pts, cols,
        title=f"Direct {len(src_imgs)}-view VGGT Reconstruction",
        point_size=0.25
    )
    utils.save_matplotlib_figure(fig, out_dir / "01_direct_point_cloud.png")

    print("\n[2/2] Done.")
    print(f"Outputs in: {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline from videos in data/")
    parser.add_argument("--mode", choices=["full", "direct"], default="direct",
                        help="full: 2 views + inpainted novel; direct: all views, no inpainting")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory with videos")
    parser.add_argument("--frame", choices=["first", "middle", "last"], default="middle",
                        help="Which frame to extract per video")
    parser.add_argument("--output-dir", type=str, default="outputs/video_run", help="Output directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    videos = _list_videos(data_dir)
    images = _list_data_images(data_dir)

    if videos and (not images or len(images) < len(videos)):
        print(f"Found {len(videos)} video(s). Extracting frames ({args.frame})...")
        saved = _extract_frames_from_videos(
            data_dir, frame_index=args.frame
        )
        for p in saved:
            print("  -", p.name)
        images = _list_data_images(data_dir)

    if len(images) < 2:
        raise RuntimeError(
            f"Need at least 2 images. Found {len(images)} in {data_dir}. "
            "Add videos (.mp4) or images (.png) to data/"
        )

    print(f"Using {len(images)} image(s):")
    for p in images[:5]:
        print("  -", p.name)
    if len(images) > 5:
        print(f"  ... and {len(images) - 5} more")

    cfg = utils.PipelineConfig(
        data_dir=str(data_dir),
        output_dir=args.output_dir,
        preprocess_mode="crop",
        vggt_conf_percentile=55.0,
        max_points_plot=20000,
        max_points_render=120000,
        render_point_radius=1,
        novel_shift_right=0.10,
        novel_yaw_deg=-3.5,
        inpaint_model_id="black-forest-labs/FLUX.2-klein-4B",
        inpaint_steps=4,
        inpaint_guidance_scale=1.0,
        seed=0,
    )
    utils.seed_everything(cfg.seed)

    out_dir = utils.ensure_dir(Path(cfg.output_dir))
    print(f"Device: {utils.get_device()}")

    if args.mode == "direct":
        run_direct_pipeline(images, cfg, out_dir)
    else:
        run_full_pipeline(images, cfg, out_dir)


if __name__ == "__main__":
    main()
