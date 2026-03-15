from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import utils


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


def main() -> None:
    cfg = utils.PipelineConfig(
        preprocess_mode="crop",
        vggt_conf_percentile=55.0,
        max_points_plot=20000,
        max_points_render=120000,
        render_point_radius=1,
        novel_shift_right=0.10,
        novel_yaw_deg=-3.5,
        inpaint_model_id="black-forest-labs/FLUX.2-klein-4B",
        inpaint_steps=4,  # FLUX2-klein local smoke-test speed
        inpaint_guidance_scale=1.0,
        seed=0,
    )

    utils.seed_everything(cfg.seed)
    device = utils.get_device()
    out_dir = utils.ensure_dir(Path(cfg.output_dir) / "tests" / "e2e")

    data_images = _list_data_images(cfg.data_dir)
    if len(data_images) < 2:
        raise RuntimeError(f"Need at least 2 images in {cfg.data_dir} for this test, found: {data_images}")
    src_imgs = [utils.ensure_png_copy(p) for p in data_images[:2]]
    print("Input images:")
    for p in src_imgs:
        print(" -", p)
    print(f"Device: {device}")

    print("\n[1/6] Running initial two-view VGGT...")
    model = utils.load_vggt_model(cfg.vggt_model_id, device=device)
    scene1 = utils.run_vggt_reconstruction(src_imgs, model=model, device=device, preprocess_mode=cfg.preprocess_mode)
    summary1 = utils.describe_scene(scene1)
    print(json.dumps(summary1, indent=2, default=str))

    initial_view_indices = list(range(len(src_imgs)))
    pts1, cols1 = _merged_scene_point_cloud(
        scene1,
        view_indices=initial_view_indices,
        conf_percentile=cfg.vggt_conf_percentile,
        max_points=cfg.max_points_plot,
        prefer_depth_unprojection=True,
        seed=cfg.seed,
    )
    fig1, _ = utils.plot_point_cloud_3d(pts1, cols1, title="Initial Two-view VGGT Reconstruction", point_size=0.25)
    utils.save_matplotlib_figure(fig1, out_dir / "01_single_view_point_cloud.png")

    print("\n[2/6] Rendering projection from VGGT camera...")
    pts_render, cols_render = _merged_scene_point_cloud(
        scene1,
        view_indices=initial_view_indices,
        conf_percentile=cfg.vggt_conf_percentile,
        max_points=cfg.max_points_render,
        prefer_depth_unprojection=True,
        seed=cfg.seed,
    )
    base_extr = np.asarray(scene1["extrinsic"][0], dtype=np.float32)
    base_intr = np.asarray(scene1["intrinsic"][0], dtype=np.float32)
    base_hw = tuple(int(x) for x in scene1["image_hw"])
    base_render = utils.render_projected_point_cloud(
        pts_render,
        cols_render,
        extrinsic=base_extr,
        intrinsic=base_intr,
        image_hw=base_hw,
        point_radius=cfg.render_point_radius,
    )
    utils.save_pil(base_render.image, out_dir / "02_projection_from_vggt_camera.png")
    print(f"Base render projected_count={base_render.projected_count}, valid_ratio={base_render.valid_mask.mean():.4f}")

    print("\n[3/6] Rendering novel view (camera shifted right + rotated)...")
    novel_extr = utils.perturb_camera_extrinsic(
        base_extr,
        shift_right=cfg.novel_shift_right,
        yaw_deg=cfg.novel_yaw_deg,
        pitch_deg=cfg.novel_pitch_deg,
        roll_deg=cfg.novel_roll_deg,
    )
    novel_intr = base_intr.copy()
    novel_render = utils.render_projected_point_cloud(
        pts_render,
        cols_render,
        extrinsic=novel_extr,
        intrinsic=novel_intr,
        image_hw=base_hw,
        point_radius=cfg.render_point_radius,
    )
    utils.save_pil(novel_render.image, out_dir / "03_novel_view_projection.png")
    np.save(out_dir / "03_novel_extrinsic.npy", novel_extr)
    np.save(out_dir / "03_novel_intrinsic.npy", novel_intr)
    print(f"Novel render projected_count={novel_render.projected_count}, valid_ratio={novel_render.valid_mask.mean():.4f}")

    print("\n[4/6] Building hole mask from projection gaps...")
    novel_img, hole_mask, overlay = utils.prepare_novel_view_inpainting_inputs(
        novel_render,
        dilate_px=cfg.mask_dilate_px,
        close_px=cfg.mask_close_px,
    )
    utils.save_pil(novel_img, out_dir / "04_novel_view_input.png")
    utils.save_pil(hole_mask, out_dir / "04_hole_mask.png")
    utils.save_pil(overlay, out_dir / "04_hole_mask_overlay.png")
    hole_ratio = float((np.asarray(hole_mask) > 0).mean())
    print(f"Hole mask ratio={hole_ratio:.4f}")

    print("\n[5/6] Diffusion inpainting (with strict preserve-unmasked compositing)...")
    # Free VGGT weights before loading diffusion pipeline to reduce memory pressure on Mac/MPS.
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
        allow_fallback_to_opencv=False,
    )
    utils.save_pil(inpaint.raw_generated, out_dir / "05_inpaint_raw_generated.png")
    utils.save_pil(inpaint.composited, out_dir / "05_inpaint_composited.png")
    print(f"Inpainting backend={inpaint.backend}, resized_for_model={inpaint.resized_for_model}")

    # Verify compositing preserved unmasked pixels exactly.
    orig_np = np.asarray(novel_img.convert("RGB"))
    comp_np = np.asarray(inpaint.composited.convert("RGB"))
    mask_np = np.asarray(hole_mask.convert("L")) > 0
    changed_unmasked = int(np.any(orig_np != comp_np, axis=-1)[~mask_np].sum())
    print(f"Changed pixels outside mask (should be 0): {changed_unmasked}")
    if changed_unmasked != 0:
        raise RuntimeError("Unmasked pixels changed after compositing")

    inpainted_path = out_dir / "05_inpaint_composited.png"

    print("\n[6/6] Running augmented VGGT (2 original views + inpainted novel view)...")
    utils.clear_loaded_model_caches()
    model2 = utils.load_vggt_model(cfg.vggt_model_id, device=device)
    scene2 = utils.run_vggt_reconstruction(
        [*src_imgs, inpainted_path],
        model=model2,
        device=device,
        preprocess_mode=cfg.preprocess_mode,
        model_id=cfg.vggt_model_id,
    )
    summary2 = utils.describe_scene(scene2)
    print(json.dumps(summary2, indent=2, default=str))

    pts2, cols2 = _merged_scene_point_cloud(
        scene2,
        view_indices=list(range(len(src_imgs) + 1)),
        conf_percentile=cfg.vggt_conf_percentile,
        max_points=cfg.max_points_plot,
        prefer_depth_unprojection=True,
        seed=cfg.seed,
    )
    fig2, _ = utils.plot_point_cloud_3d(pts2, cols2, title="Augmented Multi-view VGGT Reconstruction", point_size=0.25)
    utils.save_matplotlib_figure(fig2, out_dir / "06_two_view_point_cloud.png")

    print("\nE2E smoke test finished successfully.")
    print(f"Artifacts saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
