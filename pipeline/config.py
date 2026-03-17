"""Pipeline configuration and result dataclasses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class PipelineConfig:
    data_dir: str = "data"
    output_dir: str = "outputs"
    input_image_stem: str = "image_01"
    vggt_model_id: str = "facebook/VGGT-1B"
    inpaint_model_id: str = "black-forest-labs/FLUX.1-Fill-dev"
    inpaint_backend: str = "auto"  # auto | diffusers | flux2_klein_bfl
    bfl_api_base_url: str = "https://api.bfl.ai"
    bfl_api_key_env: str = "BFL_API_KEY"
    bfl_poll_interval_seconds: float = 1.0
    bfl_timeout_seconds: int = 600
    bfl_safety_tolerance: int = 2
    bfl_output_format: str = "png"
    preprocess_mode: str = "crop"
    vggt_conf_percentile: float = 50.0
    max_points_plot: int = 30000
    max_points_render: int = 180000
    render_point_radius: int = 1
    novel_shift_right: float = 0.12
    novel_yaw_deg: float = -4.0
    novel_pitch_deg: float = 0.0
    novel_roll_deg: float = 0.0
    mask_close_px: int = 6
    mask_dilate_px: int = 12
    mask_shrink_px: int | None = 5  # Erode silhouette to reduce mask at edges. None = close_px//2
    mask_min_area_px: int = 2
    mask_exterior_only: bool = False
    mask_all_invalid: bool = False
    mask_fill_rgb: tuple[int, int, int] | None = None
    inpaint_steps: int = 20
    inpaint_guidance_scale: float = 30.0
    inpaint_prompt: str = (
        "Photorealistic continuation, 4K ultra-HD, sharp and detailed. In focus, in foreground, "
        "extremely sharp. Fill only missing regions and preserve visible content, lighting, geometry, and textures."
    )
    inpaint_negative_prompt: str = (
        "solid color fill, flat color, blank, empty, minimal backdrop, changes to visible regions, "
        "duplicated objects, warped geometry, blurry, out of focus, depth of field, oversmoothing"
    )
    seed: int = 0
    n_frames_per_video: int = 7
    max_iterations: int = 1
    n_orbit_fill_views: int = 4


@dataclass
class RenderResult:
    image: Image.Image
    image_np: np.ndarray  # HxWx3 uint8
    valid_mask: np.ndarray  # HxW bool
    depth_map: np.ndarray  # HxW float32, inf where invalid
    projected_count: int


@dataclass
class InpaintResult:
    raw_generated: Image.Image
    composited: Image.Image
    mask_used: Image.Image
    backend: str
    resized_for_model: bool
    metadata: dict[str, Any] | None = None
