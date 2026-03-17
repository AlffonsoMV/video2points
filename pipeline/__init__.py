"""Pipeline package: VGGT reconstruction, rendering, inpainting, orbit estimation."""
from __future__ import annotations

from pipeline._state import (
    clear_loaded_model_caches,
    clear_torch_cache,
    get_device,
    seed_everything,
)
from pipeline.config import InpaintResult, PipelineConfig, RenderResult
from pipeline.geometry import (
    camera_baseline_scale,
    camera_center_from_extrinsic,
    extrinsic_from_camera_pose,
    perturb_camera_extrinsic,
    project_world_points,
    relative_camera_pose_from_reference,
    relative_pose_comparison,
    resize_intrinsics,
    rotation_error_deg,
    vector_angle_deg,
)
from pipeline.inpainting import (
    build_hole_mask_from_valid_mask,
    edit_with_flux2_klein_local,
    enhance_quality_with_flux2_klein,
    inpaint_with_diffusion,
    inpaint_with_flux2_klein_bfl,
    load_flux2_klein_pipeline,
    load_inpainting_pipeline,
    opencv_inpaint_fallback,
    prepare_novel_view_inpainting_inputs,
    refine_image_with_flux2_klein,
)
from pipeline.io import (
    ensure_dir,
    ensure_png_copy,
    extract_frame_from_video,
    extract_frames_from_videos,
    extract_n_frames_from_video,
    list_data_images,
    list_videos,
    load_pil_rgb,
    np_to_pil_rgb,
    pil_to_np_rgb,
    resolve_input_image,
    save_pil,
)
from pipeline.metrics import compute_image_metrics, load_lpips_model
from pipeline.orbit import (
    OrbitInfo,
    angular_distance_rad,
    angular_position_rad,
    average_camera_up,
    camera_at_angle,
    camera_forward_world,
    estimate_horizontal_orbit,
    estimate_orbit,
    generate_orbit_cameras,
    pick_next_angles_bilateral,
)
from pipeline.point_cloud import build_point_cloud_from_scene, merge_scene_point_cloud
from pipeline.rendering import (
    build_novel_view_render,
    render_projected_point_cloud,
    render_scene_from_custom_camera,
    render_scene_view,
)
from pipeline.vggt import load_vggt_model, run_vggt_reconstruction
from pipeline.viz import (
    build_two_view_reconstruction,
    describe_scene,
    overlay_mask_on_image,
    plot_image_grid,
    plot_point_cloud_3d,
    save_matplotlib_figure,
    save_point_cloud_ply,
    to_pil_mask,
)

__all__ = [
    # _state
    "seed_everything",
    "clear_torch_cache",
    "clear_loaded_model_caches",
    "get_device",
    # config
    "PipelineConfig",
    "RenderResult",
    "InpaintResult",
    # geometry
    "camera_center_from_extrinsic",
    "extrinsic_from_camera_pose",
    "perturb_camera_extrinsic",
    "project_world_points",
    "resize_intrinsics",
    "camera_baseline_scale",
    "relative_camera_pose_from_reference",
    "rotation_error_deg",
    "vector_angle_deg",
    "relative_pose_comparison",
    # io
    "ensure_dir",
    "resolve_input_image",
    "list_data_images",
    "list_videos",
    "extract_frame_from_video",
    "extract_frames_from_videos",
    "extract_n_frames_from_video",
    "ensure_png_copy",
    "load_pil_rgb",
    "save_pil",
    "pil_to_np_rgb",
    "np_to_pil_rgb",
    # orbit
    "OrbitInfo",
    "estimate_orbit",
    "estimate_horizontal_orbit",
    "angular_position_rad",
    "angular_distance_rad",
    "camera_at_angle",
    "pick_next_angles_bilateral",
    "generate_orbit_cameras",
    "average_camera_up",
    "camera_forward_world",
    # point_cloud
    "build_point_cloud_from_scene",
    "merge_scene_point_cloud",
    # rendering
    "render_projected_point_cloud",
    "render_scene_view",
    "render_scene_from_custom_camera",
    "build_novel_view_render",
    # vggt
    "load_vggt_model",
    "run_vggt_reconstruction",
    # inpainting
    "build_hole_mask_from_valid_mask",
    "prepare_novel_view_inpainting_inputs",
    "opencv_inpaint_fallback",
    "load_flux2_klein_pipeline",
    "edit_with_flux2_klein_local",
    "refine_image_with_flux2_klein",
    "enhance_quality_with_flux2_klein",
    "inpaint_with_flux2_klein_bfl",
    "load_inpainting_pipeline",
    "inpaint_with_diffusion",
    # metrics
    "load_lpips_model",
    "compute_image_metrics",
    # viz
    "plot_point_cloud_3d",
    "save_point_cloud_ply",
    "overlay_mask_on_image",
    "to_pil_mask",
    "save_matplotlib_figure",
    "plot_image_grid",
    "describe_scene",
    "build_two_view_reconstruction",
]
