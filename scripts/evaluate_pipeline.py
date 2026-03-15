from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import utils


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


def list_data_images(data_dir: str | Path) -> list[Path]:
    return sorted([p for p in Path(data_dir).glob("*") if p.is_file() and p.suffix in IMAGE_SUFFIXES])


def render_merged_scene_at_camera(
    scene: dict[str, Any],
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    image_hw: tuple[int, int],
    conf_percentile: float,
    max_points: int | None,
    point_radius: int,
    seed: int,
) -> utils.RenderResult:
    pts, cols = utils.merge_scene_point_cloud(
        scene,
        conf_percentile=conf_percentile,
        max_points=max_points,
        prefer_depth_unprojection=True,
        rng_seed=seed,
    )
    return utils.render_projected_point_cloud(
        pts,
        cols,
        extrinsic=np.asarray(extrinsic, dtype=np.float32),
        intrinsic=np.asarray(intrinsic, dtype=np.float32),
        image_hw=image_hw,
        point_radius=point_radius,
    )


def metric_improvement(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        "psnr": float(candidate["psnr"] - baseline["psnr"]),
        "ssim": float(candidate["ssim"] - baseline["ssim"]),
        "lpips": float(baseline["lpips"] - candidate["lpips"]),
    }


def save_image_grid(images: list, titles: list[str], path: Path, figsize: tuple[int, int] = (18, 5)) -> None:
    fig, _ = utils.plot_image_grid(images, titles=titles, figsize=figsize)
    utils.save_matplotlib_figure(fig, path)


def camera_baseline_scale(extrinsics: np.ndarray, idx_a: int = 0, idx_b: int = 1) -> float:
    c_a = utils.camera_center_from_extrinsic(np.asarray(extrinsics[idx_a], dtype=np.float32))
    c_b = utils.camera_center_from_extrinsic(np.asarray(extrinsics[idx_b], dtype=np.float32))
    return float(np.linalg.norm(c_b - c_a))


def relative_camera_pose_from_reference(
    reference_extrinsic: np.ndarray,
    target_extrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(reference_extrinsic, dtype=np.float32)
    tgt = np.asarray(target_extrinsic, dtype=np.float32)
    c_ref = utils.camera_center_from_extrinsic(ref)
    c_tgt = utils.camera_center_from_extrinsic(tgt)
    delta_world = c_tgt - c_ref
    delta_in_ref_camera = ref[:, :3] @ delta_world
    relative_rotation = tgt[:, :3] @ ref[:, :3].T
    return relative_rotation.astype(np.float32), delta_in_ref_camera.astype(np.float32)


def rotation_error_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    delta = np.asarray(rotation_a, dtype=np.float64) @ np.asarray(rotation_b, dtype=np.float64).T
    trace_value = float(np.trace(delta))
    cos_angle = np.clip((trace_value - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def vector_angle_deg(vector_a: np.ndarray, vector_b: np.ndarray, eps: float = 1e-8) -> float:
    a = np.asarray(vector_a, dtype=np.float64)
    b = np.asarray(vector_b, dtype=np.float64)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a < eps or norm_b < eps:
        return 0.0
    cos_angle = np.clip(float(np.dot(a, b)) / (norm_a * norm_b), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def relative_pose_comparison(
    target_rotation: np.ndarray,
    target_translation: np.ndarray,
    predicted_rotation: np.ndarray,
    predicted_translation: np.ndarray,
) -> dict[str, float]:
    target_translation = np.asarray(target_translation, dtype=np.float32)
    predicted_translation = np.asarray(predicted_translation, dtype=np.float32)
    target_norm = float(np.linalg.norm(target_translation))
    predicted_norm = float(np.linalg.norm(predicted_translation))
    return {
        "rotation_error_deg": rotation_error_deg(predicted_rotation, target_rotation),
        "translation_l2_normalized": float(np.linalg.norm(predicted_translation - target_translation)),
        "translation_direction_error_deg": vector_angle_deg(predicted_translation, target_translation),
        "target_translation_norm": target_norm,
        "predicted_translation_norm": predicted_norm,
        "translation_norm_ratio_pred_over_target": float(predicted_norm / target_norm) if target_norm > 1e-8 else 0.0,
    }


def evaluate_novel_view_regeneration(
    model: Any,
    device: str,
    data_images: list[Path],
    e2e_dir: Path,
    out_dir: Path,
    cfg: utils.PipelineConfig,
) -> dict[str, Any]:
    baseline_render_path = e2e_dir / "03_novel_view_projection.png"
    target_generated_path = e2e_dir / "05_inpaint_raw_generated.png"
    novel_extr_path = e2e_dir / "03_novel_extrinsic.npy"
    novel_intr_path = e2e_dir / "03_novel_intrinsic.npy"

    required = [baseline_render_path, target_generated_path, novel_extr_path, novel_intr_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Novel-view evaluation requires existing e2e artifacts: {missing}")

    baseline_render = utils.load_pil_rgb(baseline_render_path)
    target_generated = utils.load_pil_rgb(target_generated_path)
    novel_extr = np.load(novel_extr_path)
    novel_intr = np.load(novel_intr_path)

    scene_base = utils.run_vggt_reconstruction(
        data_images,
        model=model,
        device=device,
        preprocess_mode=cfg.preprocess_mode,
        model_id=cfg.vggt_model_id,
    )
    augmented_inputs = [*data_images, target_generated_path]
    scene_augmented = utils.run_vggt_reconstruction(
        augmented_inputs,
        model=model,
        device=device,
        preprocess_mode=cfg.preprocess_mode,
        model_id=cfg.vggt_model_id,
    )
    rerender = render_merged_scene_at_camera(
        scene_augmented,
        extrinsic=novel_extr,
        intrinsic=novel_intr,
        image_hw=tuple(int(x) for x in scene_augmented["image_hw"]),
        conf_percentile=cfg.vggt_conf_percentile,
        max_points=cfg.max_points_render,
        point_radius=cfg.render_point_radius,
        seed=cfg.seed,
    )
    rerender_path = out_dir / "01_augmented_same_pose_rerender.png"
    utils.save_pil(rerender.image, rerender_path)

    same_pose_baseline = utils.compute_image_metrics(target_generated, baseline_render, lpips_device="cpu")
    same_pose_augmented = utils.compute_image_metrics(target_generated, rerender.image, lpips_device="cpu")

    base_extrinsics = np.asarray(scene_base["extrinsic"], dtype=np.float32)
    augmented_extrinsics = np.asarray(scene_augmented["extrinsic"], dtype=np.float32)
    generated_index = len(data_images)
    base_scale = camera_baseline_scale(base_extrinsics, idx_a=0, idx_b=1)
    augmented_scale = camera_baseline_scale(augmented_extrinsics, idx_a=0, idx_b=1)

    target_rotation, target_translation = relative_camera_pose_from_reference(base_extrinsics[0], novel_extr)
    predicted_generated_rotation, predicted_generated_translation = relative_camera_pose_from_reference(
        augmented_extrinsics[0],
        augmented_extrinsics[generated_index],
    )

    target_translation_normalized = target_translation / max(base_scale, 1e-8)
    predicted_generated_translation_normalized = predicted_generated_translation / max(augmented_scale, 1e-8)

    predicted_intrinsic = np.asarray(scene_augmented["intrinsic"][generated_index], dtype=np.float32)
    target_intrinsic = np.asarray(novel_intr, dtype=np.float32)
    focal_target = np.array([target_intrinsic[0, 0], target_intrinsic[1, 1]], dtype=np.float32)
    focal_predicted = np.array([predicted_intrinsic[0, 0], predicted_intrinsic[1, 1]], dtype=np.float32)
    principal_target = np.array([target_intrinsic[0, 2], target_intrinsic[1, 2]], dtype=np.float32)
    principal_predicted = np.array([predicted_intrinsic[0, 2], predicted_intrinsic[1, 2]], dtype=np.float32)

    original_pose_comparisons: list[dict[str, Any]] = []
    for idx in range(1, len(data_images)):
        original_rotation, original_translation = relative_camera_pose_from_reference(
            base_extrinsics[0],
            base_extrinsics[idx],
        )
        original_translation_normalized = original_translation / max(base_scale, 1e-8)
        original_pose_comparisons.append(
            {
                "original_index": idx,
                "original_path": str(data_images[idx]),
                **relative_pose_comparison(
                    original_rotation,
                    original_translation_normalized,
                    predicted_generated_rotation,
                    predicted_generated_translation_normalized,
                ),
            }
        )
    closest_original_pose = min(
        original_pose_comparisons,
        key=lambda item: (item["translation_l2_normalized"], item["rotation_error_deg"]),
    )

    cross_view_reference_metrics: dict[str, Any] = {}
    for src in data_images:
        original = utils.load_pil_rgb(src)
        cross_view_reference_metrics[src.name] = {
            "note": "Different viewpoints. These numbers are only rough references, not a proper image-quality metric.",
            "baseline_render_vs_original": utils.compute_image_metrics(original, baseline_render, lpips_device="cpu"),
            "augmented_rerender_vs_original": utils.compute_image_metrics(original, rerender.image, lpips_device="cpu"),
            "generated_target_vs_original": utils.compute_image_metrics(original, target_generated, lpips_device="cpu"),
        }

    save_image_grid(
        [*(utils.load_pil_rgb(p) for p in data_images), baseline_render, target_generated, rerender.image],
        [*(p.name for p in data_images), "Baseline novel render", "Raw generated target", "Augmented rerender"],
        out_dir / "02_novel_view_comparison_grid.png",
        figsize=(22, 5),
    )

    return {
        "augmented_inputs": [str(p) for p in augmented_inputs],
        "same_pose_target": str(target_generated_path),
        "baseline_render": str(baseline_render_path),
        "augmented_rerender": str(rerender_path),
        "same_pose_metrics": {
            "baseline_render_vs_generated_target": same_pose_baseline,
            "augmented_rerender_vs_generated_target": same_pose_augmented,
            "improvement_over_baseline": metric_improvement(same_pose_augmented, same_pose_baseline),
        },
        "camera_pose_metrics": {
            "note": "Pose comparisons use camera-0-relative geometry and baseline-normalized translations so the result is meaningful even if the two reconstructions use slightly different global frames or scales.",
            "generated_view_index": generated_index,
            "target_vs_predicted_generated": relative_pose_comparison(
                target_rotation,
                target_translation_normalized,
                predicted_generated_rotation,
                predicted_generated_translation_normalized,
            ),
            "closest_original_vs_predicted_generated": closest_original_pose,
            "intrinsic_error": {
                "focal_l2": float(np.linalg.norm(focal_predicted - focal_target)),
                "principal_point_l2": float(np.linalg.norm(principal_predicted - principal_target)),
                "predicted_focal": [float(x) for x in focal_predicted],
                "target_focal": [float(x) for x in focal_target],
                "predicted_principal_point": [float(x) for x in principal_predicted],
                "target_principal_point": [float(x) for x in principal_target],
            },
        },
        "cross_view_reference_metrics": cross_view_reference_metrics,
    }


def main() -> None:
    cfg = utils.PipelineConfig(
        preprocess_mode="crop",
        vggt_conf_percentile=55.0,
        max_points_render=120000,
        render_point_radius=1,
        seed=0,
    )
    utils.seed_everything(cfg.seed)

    data_images = list_data_images(cfg.data_dir)
    if len(data_images) < 2:
        raise RuntimeError(f"Need at least 2 real images in {cfg.data_dir}. Found: {data_images}")

    out_root = utils.ensure_dir(Path(cfg.output_dir) / "tests" / "eval")
    novel_dir = utils.ensure_dir(out_root / "novel_view_regeneration")
    e2e_dir = Path(cfg.output_dir) / "tests" / "e2e"

    device = utils.get_device()
    print(f"Device: {device}")
    print("Data images:")
    for p in data_images:
        print(" -", p)

    model = utils.load_vggt_model(cfg.vggt_model_id, device=device)

    novel_view_eval = evaluate_novel_view_regeneration(model, device, data_images, e2e_dir, novel_dir, cfg)
    summary = {"novel_view_regeneration": novel_view_eval}
    summary_path = out_root / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nSaved evaluation summary:")
    print(summary_path.resolve())
    print("\nNovel-view same-pose comparison:")
    print(json.dumps(novel_view_eval["same_pose_metrics"], indent=2))


if __name__ == "__main__":
    main()
