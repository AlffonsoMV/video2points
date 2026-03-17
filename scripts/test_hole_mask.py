#!/usr/bin/env python3
"""
Test hole masking in isolation. Tweaks mask params without running the full pipeline.

WHERE TO TWEAK (hole mask too large / bleeding around object):
  - pipeline/config.py: mask_close_px, mask_dilate_px, mask_shrink_px
  - run_with_videos.py: --mask-close-px, --mask-dilate-px, --mask-shrink-px
  - run_iterative_loop.py: set pipeline_cfg.mask_close_px etc. in main()

KEY PARAMS (to reduce mask around object):
  - close_px: LOWER = tighter silhouette (default 15). Try 5–8 for Colosseum.
  - shrink_px: HIGHER = less mask at edges (default close_px//2). Try 8–12.
  - dilate_px: LOWER = smaller holes (default 3). Try 1–2.

Usage:
    # Single test with custom params:
    python -m scripts.test_hole_mask outputs/iterative/Colosseum/03_iterations/iter_001/view_00/01_render.png --close-px 8 --shrink-px 6

    # SWEEP: try many (close, shrink) combos, save grid for visual comparison:
    python -m scripts.test_hole_mask outputs/iterative/Colosseum/03_iterations/iter_001/view_00/01_render.png --sweep

    # Sweep with custom ranges:
    python -m scripts.test_hole_mask path/to/01_render.png --sweep --close-range 5,8,12 --shrink-range 4,8,12
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))



def _parse_range(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Test hole mask params on a render image")
    parser.add_argument("render_path", type=Path, help="Path to 01_render.png (or any render)")
    parser.add_argument("--close-px", type=int, default=3,
                        help="Morphological close radius to bridge gaps. Lower = tighter silhouette, may miss interior holes.")
    parser.add_argument("--dilate-px", type=int, default=1,
                        help="Dilate holes for inpainting overlap. Lower = smaller holes.")
    parser.add_argument("--shrink-px", type=int, default=15,
                        help="Erode silhouette to reduce mask around edges. Higher = less mask at silhouette.")
    parser.add_argument("--min-area-px", type=int, default=2,
                        help="Filter out holes smaller than this (default 2)")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Output dir (default: same as render)")
    parser.add_argument("--sweep", action="store_true",
                        help="Try multiple (close_px, shrink_px) combos; save grid image for visual comparison.")
    parser.add_argument("--close-range", type=str, default="3,4,5,8,10",
                        help="Comma-separated close_px values for sweep (default: 3,4,5,8,10)")
    parser.add_argument("--shrink-range", type=str, default="10,12,15,18,20",
                        help="Comma-separated shrink_px values for sweep (default: 10,12,15,18,20)")
    args = parser.parse_args()

    render_path = args.render_path
    if not render_path.exists():
        sys.exit(f"Render not found: {render_path}")

    # Reconstruct valid_mask from render: non-white pixels = valid
    # (Pipeline uses depth-based valid_mask; this approximates from saved image)
    img = np.array(Image.open(render_path).convert("RGB"))
    white_dist = np.linalg.norm(img.astype(float) - 255.0, axis=-1)
    valid_mask = white_dist > 15

    out_dir = args.output_dir or render_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    from pipeline.viz import overlay_mask_on_image, to_pil_mask

    if args.sweep:
        _run_sweep(
            img=img,
            valid_mask=valid_mask,
            out_dir=out_dir,
            dilate_px=args.dilate_px,
            min_area_px=args.min_area_px,
            close_range=_parse_range(args.close_range),
            shrink_range=_parse_range(args.shrink_range),
        )
        return

    # Single run
    shrink = args.shrink_px if args.shrink_px is not None else args.close_px // 2
    hole_mask_np = _build_hole_mask_with_shrink(
        valid_mask, args.close_px, args.dilate_px, args.min_area_px, shrink,
    )

    hole_ratio = (hole_mask_np > 0).mean()
    print(f"close_px={args.close_px} dilate_px={args.dilate_px} shrink_px={shrink}")
    print(f"valid_ratio={valid_mask.mean():.4f} hole_ratio={hole_ratio:.4f}")

    render = type("Render", (), {"image": Image.fromarray(img)})()
    hole_mask_pil = to_pil_mask(hole_mask_np)
    overlay = overlay_mask_on_image(render.image, hole_mask_pil)

    mask_path = out_dir / "test_hole_mask.png"
    overlay_path = out_dir / "test_hole_overlay.png"
    hole_mask_pil.save(mask_path)
    overlay.save(overlay_path)

    print(f"Saved: {mask_path}")
    print(f"Saved: {overlay_path}")


def _run_sweep(
    img: np.ndarray,
    valid_mask: np.ndarray,
    out_dir: Path,
    dilate_px: int,
    min_area_px: int,
    close_range: list[int],
    shrink_range: list[int],
) -> None:
    """Generate a grid of hole overlays for different (close_px, shrink_px) combos."""
    import cv2

    from pipeline.viz import overlay_mask_on_image, to_pil_mask

    overlays: list[tuple[int, int, np.ndarray]] = []
    for close_px in close_range:
        for shrink_px in shrink_range:
            hole_mask_np = _build_hole_mask_with_shrink(
                valid_mask, close_px, dilate_px, min_area_px, shrink_px,
            )
            hole_mask_pil = to_pil_mask(hole_mask_np)
            render_pil = Image.fromarray(img)
            overlay = overlay_mask_on_image(render_pil, hole_mask_pil)
            overlays.append((close_px, shrink_px, np.array(overlay)))

    # Build grid: rows = close_px, cols = shrink_px
    n_close = len(close_range)
    n_shrink = len(shrink_range)
    h, w = overlays[0][2].shape[:2]
    grid = np.zeros((n_close * h, n_shrink * w, 3), dtype=np.uint8)
    grid[:] = 255

    for i, close_px in enumerate(close_range):
        for j, shrink_px in enumerate(shrink_range):
            idx = i * n_shrink + j
            _, _, ov = overlays[idx]
            grid[i * h : (i + 1) * h, j * w : (j + 1) * w] = ov

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(w, h) / 400.0
    for j, shrink_px in enumerate(shrink_range):
        cv2.putText(
            grid, f"shrink={shrink_px}",
            (j * w + 8, 28), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA,
        )
    for i, close_px in enumerate(close_range):
        cv2.putText(
            grid, f"close={close_px}",
            (8, i * h + 28), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA,
        )

    grid_path = out_dir / "test_hole_sweep_grid.png"
    Image.fromarray(grid).save(grid_path)
    print(f"Sweep grid saved: {grid_path}")
    print(f"  close_px: {close_range}")
    print(f"  shrink_px: {shrink_range}")
    print("  Pick best visually, then run full pipeline with e.g.:")
    print(f"  --mask-close-px <best_close> --mask-shrink-px <best_shrink>")


def _build_hole_mask_with_shrink(
    valid_mask: np.ndarray,
    close_px: int,
    dilate_px: int,
    min_area_px: int,
    shrink_px: int,
) -> np.ndarray:
    """Same logic as build_hole_mask_from_valid_mask but with explicit shrink."""
    import cv2
    from scipy.ndimage import binary_fill_holes

    valid = np.asarray(valid_mask, dtype=bool)

    closed = valid.astype(np.uint8) * 255
    if close_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1),
        )
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, k)

    silhouette = binary_fill_holes(closed > 0)

    if shrink_px > 0:
        ek = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * shrink_px + 1, 2 * shrink_px + 1),
        )
        silhouette = cv2.erode(
            silhouette.astype(np.uint8), ek,
        ).astype(bool)

    hole_mask = (silhouette & ~valid).astype(np.uint8) * 255

    if min_area_px > 1:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (hole_mask > 0).astype(np.uint8), connectivity=8,
        )
        cleaned = np.zeros_like(hole_mask)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_area_px:
                cleaned[labels == label] = 255
        hole_mask = cleaned

    if dilate_px > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1),
        )
        hole_mask = cv2.dilate(hole_mask, k)
        hole_mask[valid] = 0

    return hole_mask


if __name__ == "__main__":
    main()
