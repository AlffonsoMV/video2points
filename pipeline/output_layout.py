"""Standard output layout for iterative pipeline runs.

Layout (run_with_videos):
    outputs/iterative/<video_stem>/
    ├── 01_inputs/
    │   └── frames/              # extracted video frames
    ├── 02_initial/
    │   └── point_cloud.png
    ├── 03_iterations/
    │   ├── iter_001/
    │   │   ├── view_00/
    │   │   │   ├── 01_render.png         # point cloud projection (VGGT resolution)
    │   │   │   ├── 02_hole_mask.png      # hole mask (VGGT resolution)
    │   │   │   ├── 03_hole_overlay.png   # mask overlaid on render
    │   │   │   ├── 04_upscaled.png       # upscaled to original frame resolution (if differs)
    │   │   │   ├── 05_flux_quality.png   # after FLUX quality enhancement (if enabled)
    │   │   │   ├── 06_inpainted.png      # after hole inpainting (SDXL or OpenCV)
    │   │   │   └── final.png             # output fed to next VGGT run
    │   │   └── view_01/
    │   │       └── ...
    │   └── iter_002/
    │       └── ...
    ├── 04_final/
    │   └── point_cloud.png
    ├── manifest.json
    └── STRUCTURE.txt
"""
from __future__ import annotations

from pathlib import Path


def dir_inputs(out_dir: Path) -> Path:
    return out_dir / "01_inputs"


def dir_frames(out_dir: Path) -> Path:
    return dir_inputs(out_dir) / "frames"


def dir_initial(out_dir: Path) -> Path:
    return out_dir / "02_initial"


def dir_iterations(out_dir: Path) -> Path:
    return out_dir / "03_iterations"


def dir_iter(out_dir: Path, iteration: int) -> Path:
    return dir_iterations(out_dir) / f"iter_{iteration:03d}"


def dir_view(out_dir: Path, iteration: int, view_idx: int) -> Path:
    return dir_iter(out_dir, iteration) / f"view_{view_idx:02d}"


def dir_final(out_dir: Path) -> Path:
    return out_dir / "04_final"


STRUCTURE_TXT = """Output layout (pipeline flow)

01_inputs/frames/      Input frames extracted from video (fed to initial VGGT)
02_initial/            Point cloud after first VGGT run
03_iterations/         Per-iteration, per-view artifacts
  iter_XXX/view_YY/
    01_render.png        Point cloud projection at VGGT resolution (sparse)
    02_hole_mask.png     Binary mask of interior holes to inpaint
    03_hole_overlay.png  Mask overlaid on render for visualization
    04_upscaled.png      Render + mask upscaled to original frame resolution (*)
    05_flux_quality.png  After FLUX quality enhancement on foreground (*)
    06_inpainted.png     After hole inpainting (SDXL or OpenCV)
    final.png            Output image fed to next VGGT run
04_final/              Final point cloud after all iterations
manifest.json          Run metadata, paths, iteration summary

(*) = conditional: 04 only when VGGT resolution differs from original frames,
      05 only when FLUX quality enhancement is enabled (default: on).

Pipeline per view: render → upscale → FLUX enhance → inpaint → VGGT
"""
