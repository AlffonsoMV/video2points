from __future__ import annotations

from pathlib import Path

import numpy as np

import utils


def main() -> None:
    out_dir = utils.ensure_dir("outputs/synthetic")
    utils.seed_everything(0)

    n = 4000
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, n)
    y = rng.uniform(-0.7, 0.7, n)
    z = rng.uniform(2.5, 4.5, n)
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = np.stack(
        [
            (x - x.min()) / (x.max() - x.min() + 1e-8),
            (y - y.min()) / (y.max() - y.min() + 1e-8),
            (z - z.min()) / (z.max() - z.min() + 1e-8),
        ],
        axis=1,
    ).astype(np.float32)

    h, w = 256, 256
    fx = fy = 180.0
    cx, cy = w / 2.0, h / 2.0
    intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    extr = np.concatenate([np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)], axis=1)

    render = utils.render_projected_point_cloud(
        pts,
        colors,
        extrinsic=extr,
        intrinsic=intr,
        image_hw=(h, w),
        point_radius=1,
    )
    novel_extr = utils.perturb_camera_extrinsic(extr, shift_right=0.2, yaw_deg=-8.0)
    novel_render = utils.render_projected_point_cloud(
        pts,
        colors,
        extrinsic=novel_extr,
        intrinsic=intr,
        image_hw=(h, w),
        point_radius=1,
    )
    novel_img, hole_mask, overlay = utils.prepare_novel_view_inpainting_inputs(novel_render, dilate_px=3, close_px=2)
    fallback = utils.opencv_inpaint_fallback(novel_img, hole_mask)

    utils.save_pil(render.image, out_dir / "synthetic_base_projection.png")
    utils.save_pil(novel_render.image, out_dir / "synthetic_novel_projection.png")
    utils.save_pil(hole_mask, out_dir / "synthetic_hole_mask.png")
    utils.save_pil(overlay, out_dir / "synthetic_hole_overlay.png")
    utils.save_pil(fallback, out_dir / "synthetic_inpaint_fallback.png")

    hole_ratio = float((np.asarray(hole_mask) > 0).mean())
    valid_ratio = float(render.valid_mask.mean())
    assert render.image_np.shape == (h, w, 3)
    assert novel_render.image_np.shape == (h, w, 3)
    assert 0.0 < valid_ratio < 1.0
    assert 0.0 < hole_ratio < 1.0

    print("Synthetic utils smoke test passed")
    print(f"base projected_count={render.projected_count}, valid_ratio={valid_ratio:.4f}")
    print(f"novel projected_count={novel_render.projected_count}, hole_ratio={hole_ratio:.4f}")
    print(f"outputs: {Path(out_dir).resolve()}")


if __name__ == "__main__":
    main()

