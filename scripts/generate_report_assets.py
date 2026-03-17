from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import utils


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
FIGURES_DIR = REPORT_DIR / "figures"
DATA_DIR = REPORT_DIR / "data"
E2E_DIR = ROOT / "outputs" / "e2e"
EVAL_DIR = ROOT / "outputs" / "eval"
ITER_DIR = ROOT / "outputs" / "iterative" / "plant"


@dataclass
class PrefixEvaluation:
    iteration: int
    label: str
    image_paths: list[Path]
    rerender_path: str
    psnr: float
    ssim: float
    lpips: float


def ensure_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return path


def save_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def detect_iterative_root(preferred: str | None = None) -> Path:
    if preferred:
        preferred_path = ROOT / preferred
        manifest = preferred_path / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(f"Preferred iterative root does not contain a manifest: {preferred_path}")
        return preferred_path

    candidates = [
        ROOT / "outputs" / "iterative" / "plant",
        ROOT / "outputs" / "iterative" / "horizontal_probe",
        ROOT / "outputs" / "iterative" / "Colosseum",
    ]
    valid = []
    for candidate in candidates:
        manifest = candidate / "manifest.json"
        if manifest.exists():
            payload = json.loads(manifest.read_text())
            valid.append((len(payload.get("final_image_paths", [])), candidate))
    if not valid:
        raise FileNotFoundError("No iterative loop manifest found under outputs/iterative.")
    valid.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    return valid[0][1]


def resolve_iterative_artifact(path_str: str, iter_root: Path) -> Path:
    direct = ROOT / path_str
    if direct.exists():
        return direct
    rel = Path(path_str)
    for marker in (
        Path("outputs") / "iterative" / "plant",
        Path("outputs") / "iterative",
    ):
        parts = rel.parts
        marker_parts = marker.parts
        if len(parts) >= len(marker_parts) and tuple(parts[: len(marker_parts)]) == marker_parts:
            suffix = Path(*parts[len(marker_parts) :])
            candidate = iter_root / suffix
            if candidate.exists():
                return candidate
    return direct


def copy_asset(src: Path, dst_name: str) -> Path:
    dst = FIGURES_DIR / dst_name
    shutil.copy2(require(src), dst)
    return dst


def save_grid(
    images: list[Any],
    titles: list[str],
    out_name: str,
    figsize: tuple[int, int],
) -> Path:
    pil_images = []
    for image in images:
        if hasattr(image, "save"):
            pil_images.append(image.convert("RGB"))
        else:
            pil_images.append(utils.load_pil_rgb(image))
    fig, _ = utils.plot_image_grid(pil_images, titles=titles, figsize=figsize)
    return utils.save_matplotlib_figure(fig, FIGURES_DIR / out_name)


def render_merged_scene_at_camera(
    scene: dict[str, Any],
    view_idx: int,
    cfg: utils.PipelineConfig,
    seed: int,
):
    pts, cols = utils.merge_scene_point_cloud(
        scene,
        conf_percentile=cfg.vggt_conf_percentile,
        max_points=cfg.max_points_render,
        prefer_depth_unprojection=True,
        rng_seed=seed,
    )
    return utils.render_projected_point_cloud(
        pts,
        cols,
        extrinsic=np.asarray(scene["extrinsic"][view_idx], dtype=np.float32),
        intrinsic=np.asarray(scene["intrinsic"][view_idx], dtype=np.float32),
        image_hw=tuple(int(x) for x in scene["image_hw"]),
        point_radius=cfg.render_point_radius,
    )


def build_prefix_sets(manifest: dict[str, Any], iter_root: Path) -> list[tuple[int, str, list[Path]]]:
    current_paths = [resolve_iterative_artifact(p, iter_root) if p.startswith("outputs/") else ROOT / p for p in manifest["initial_images"]]
    prefix_sets: list[tuple[int, str, list[Path]]] = [(0, "Baseline (4 real views)", list(current_paths))]
    for item in manifest["iterations"]:
        current_paths = [*current_paths, resolve_iterative_artifact(item["generated_raw_path"], iter_root)]
        prefix_sets.append(
            (
                int(item["iteration_index"]),
                f"+{item['iteration_index']} synthetic view" + ("s" if int(item["iteration_index"]) > 1 else ""),
                list(current_paths),
            )
        )
    return prefix_sets


def create_pipeline_figures(
    iter_root: Path,
    e2e_dir: Path | None = None,
    eval_dir: Path | None = None,
) -> dict[str, str]:
    e2e_dir = e2e_dir or E2E_DIR
    eval_dir = eval_dir or EVAL_DIR
    figure_paths: dict[str, str] = {}
    originals = [ROOT / "data" / f"image_0{i}.png" for i in range(1, 5)]

    figure_paths["pipeline_inputs_and_reprojection"] = str(
        save_grid(
            [*originals, e2e_dir / "02_projection_from_vggt_camera.png"],
            ["Input 1", "Input 2", "Input 3", "Input 4", "VGGT render"],
            "pipeline_inputs_and_reprojection.png",
            figsize=(20, 4),
        )
    )
    figure_paths["pipeline_camera_move"] = str(
        save_grid(
            [e2e_dir / "02_projection_from_vggt_camera.png", e2e_dir / "03_novel_view_projection.png"],
            ["Render at original camera", "Target sparse novel render"],
            "pipeline_camera_move.png",
            figsize=(12, 4),
        )
    )
    figure_paths["pipeline_masking"] = str(
        save_grid(
            [e2e_dir / "03_novel_view_projection.png", e2e_dir / "04_hole_mask.png", e2e_dir / "04_hole_mask_overlay.png"],
            ["Sparse novel render", "Hole mask", "Mask overlay"],
            "pipeline_masking.png",
            figsize=(15, 4),
        )
    )
    figure_paths["pipeline_flux_completion"] = str(
        save_grid(
            [e2e_dir / "04_novel_view_input.png", e2e_dir / "05_inpaint_raw_generated.png"],
            ["Masked FLUX input", "FLUX raw generation"],
            "pipeline_flux_completion.png",
            figsize=(12, 4),
        )
    )
    figure_paths["pipeline_reconstruction_comparison"] = str(
        copy_asset(e2e_dir / "07_point_cloud_comparison.png", "pipeline_reconstruction_comparison.png")
    )
    figure_paths["pipeline_same_pose_rerender"] = str(
        save_grid(
            [
                e2e_dir / "03_novel_view_projection.png",
                e2e_dir / "05_inpaint_raw_generated.png",
                eval_dir / "novel_view_regeneration" / "01_augmented_same_pose_rerender.png",
            ],
            ["Baseline sparse render", "Generated target", "Augmented rerender"],
            "pipeline_same_pose_rerender.png",
            figsize=(16, 4),
        )
    )
    figure_paths["results_final_point_cloud"] = str(
        copy_asset(iter_root / "final" / "point_cloud.png", "results_final_point_cloud.png")
    )
    return figure_paths


def collect_iterative_metrics(manifest: dict[str, Any], iter_root: Path) -> dict[str, Any]:
    baseline_gap = float(manifest["iterations"][0]["largest_gap_before_deg"])
    gap_series = [{"iteration": 0, "largest_unseen_gap_deg": baseline_gap}]
    pose_series = []

    for item in manifest["iterations"]:
        stats_path = resolve_iterative_artifact(item["stats_path"], iter_root)
        stats = json.loads(stats_path.read_text())
        iteration = int(item["iteration_index"])
        gap_series.append(
            {
                "iteration": iteration,
                "largest_unseen_gap_deg": float(item["largest_gap_after_deg"]),
            }
        )
        pose_error = stats["pose_and_coverage_after"]["target_pose_error"]
        pose_series.append(
            {
                "iteration": iteration,
                "rotation_error_deg": float(pose_error["rotation_error_deg"]),
                "translation_l2_normalized": float(pose_error["translation_l2_normalized"]),
                "min_angle_to_existing_deg": float(stats["pose_and_coverage_after"]["min_angle_to_existing_deg"]),
            }
        )

    return {"gap_series": gap_series, "pose_series": pose_series}


def plot_gap_series(gap_series: list[dict[str, float]], target_gap_deg: float) -> str:
    iterations = [point["iteration"] for point in gap_series]
    gaps = [point["largest_unseen_gap_deg"] for point in gap_series]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(iterations, gaps, marker="o", linewidth=2.0, color="#1f4f7a")
    ax.axhline(target_gap_deg, linestyle="--", linewidth=1.5, color="#b22222", label="Uniform target gap")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Largest unseen gap (deg)")
    ax.set_title("Coverage improvement over the closed loop")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return str(utils.save_matplotlib_figure(fig, FIGURES_DIR / "results_largest_gap_curve.png"))


def plot_pose_series(pose_series: list[dict[str, float]]) -> str:
    iterations = [point["iteration"] for point in pose_series]
    rotation = [point["rotation_error_deg"] for point in pose_series]
    translation = [point["translation_l2_normalized"] for point in pose_series]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(iterations, rotation, marker="o", linewidth=2.0, color="#2a7f62")
    axes[0].set_title("Rotation error")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Degrees")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(iterations, translation, marker="o", linewidth=2.0, color="#8a4f00")
    axes[1].set_title("Normalized translation error")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("L2 distance")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("Recovered camera fidelity of generated views", y=1.02)
    fig.tight_layout()
    return str(utils.save_matplotlib_figure(fig, FIGURES_DIR / "results_pose_fidelity_curve.png"))


def evaluate_fixed_camera_preservation(iter_root: Path) -> list[PrefixEvaluation]:
    manifest = json.loads((iter_root / "manifest.json").read_text())
    prefix_sets = build_prefix_sets(manifest, iter_root)
    cfg = utils.PipelineConfig(
        preprocess_mode="crop",
        vggt_conf_percentile=55.0,
        max_points_render=80000,
        render_point_radius=2,
        seed=1,
    )
    reference_image = ROOT / "data" / "image_01.png"
    device = utils.get_device()

    utils.seed_everything(cfg.seed)
    evaluations: list[PrefixEvaluation] = []
    rerender_images = [utils.load_pil_rgb(reference_image)]
    rerender_titles = ["Original image_01"]

    for iteration, label, image_paths in prefix_sets:
        print(f"[report] fixed-camera evaluation for prefix {iteration} ({len(image_paths)} views)")
        utils.clear_loaded_model_caches(clear_vggt=True, clear_inpaint=True, clear_lpips=False)
        model = utils.load_vggt_model(cfg.vggt_model_id, device=device)
        try:
            scene = utils.run_vggt_reconstruction(
                image_paths,
                model=model,
                device=device,
                preprocess_mode=cfg.preprocess_mode,
                model_id=cfg.vggt_model_id,
            )
            rerender = render_merged_scene_at_camera(scene, view_idx=0, cfg=cfg, seed=cfg.seed + iteration)
            rerender_path = FIGURES_DIR / f"results_fixed_camera_rerender_iter_{iteration:02d}.png"
            utils.save_pil(rerender.image, rerender_path)
            metrics = utils.compute_image_metrics(reference_image, rerender.image, lpips_device="cpu")
            evaluations.append(
                PrefixEvaluation(
                    iteration=iteration,
                    label=label,
                    image_paths=image_paths,
                    rerender_path=str(rerender_path),
                    psnr=float(metrics["psnr"]),
                    ssim=float(metrics["ssim"]),
                    lpips=float(metrics["lpips"]),
                )
            )
            rerender_images.append(rerender.image)
            rerender_titles.append(label.replace(" synthetic view", "\n+1 synthetic").replace(" synthetic views", "\nsynthetic"))
        finally:
            del model
            utils.clear_loaded_model_caches(clear_vggt=True, clear_inpaint=True, clear_lpips=False)

    save_grid(
        rerender_images,
        rerender_titles,
        "results_fixed_camera_rerender_strip.png",
        figsize=(18, 4),
    )
    return evaluations


def plot_fixed_camera_metrics(evaluations: list[PrefixEvaluation]) -> str:
    iterations = [item.iteration for item in evaluations]
    psnr = [item.psnr for item in evaluations]
    ssim = [item.ssim for item in evaluations]
    lpips = [item.lpips for item in evaluations]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, values, title, color, ylabel in [
        (axes[0], psnr, "PSNR", "#1f4f7a", "Higher is better"),
        (axes[1], ssim, "SSIM", "#2a7f62", "Higher is better"),
        (axes[2], lpips, "LPIPS", "#8a4f00", "Lower is better"),
    ]:
        ax.plot(iterations, values, marker="o", linewidth=2.0, color=color)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    fig.suptitle("Fixed-camera preservation over iterations", y=1.02)
    fig.tight_layout()
    return str(utils.save_matplotlib_figure(fig, FIGURES_DIR / "results_fixed_camera_metrics.png"))


def create_generated_views_strip(manifest: dict[str, Any], iter_root: Path) -> str:
    generated_paths = [resolve_iterative_artifact(item["generated_raw_path"], iter_root) for item in manifest["iterations"]]
    titles = [f"Iter {item['iteration_index']}" for item in manifest["iterations"]]
    return str(save_grid(generated_paths, titles, "results_generated_views_strip.png", figsize=(16, 4)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report figures and metrics from existing pipeline outputs.")
    parser.add_argument(
        "--iterative-root",
        type=str,
        default=None,
        help="Optional path relative to the repo root for the iterative results folder (e.g. outputs/iterative/plant).",
    )
    parser.add_argument(
        "--e2e-dir",
        type=str,
        default=None,
        help="Optional path relative to the repo root for the e2e pipeline outputs (default: outputs/e2e).",
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=None,
        help="Optional path relative to the repo root for the evaluation outputs (default: outputs/eval).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    e2e_dir = ROOT / (args.e2e_dir or "outputs/e2e")
    eval_dir = ROOT / (args.eval_dir or "outputs/eval")

    evaluation_summary = json.loads(require(eval_dir / "evaluation_summary.json").read_text())
    iterative_root = detect_iterative_root(args.iterative_root)
    iterative_manifest = json.loads(require(iterative_root / "manifest.json").read_text())
    require(e2e_dir / "05_inpaint_metadata.json")

    figure_paths = create_pipeline_figures(iterative_root, e2e_dir=e2e_dir, eval_dir=eval_dir)
    iterative_metrics = collect_iterative_metrics(iterative_manifest, iterative_root)
    figure_paths["results_gap_curve"] = plot_gap_series(
        iterative_metrics["gap_series"],
        target_gap_deg=360.0 / float(iterative_manifest["loop_config"]["max_total_views"]),
    )
    figure_paths["results_pose_curve"] = plot_pose_series(iterative_metrics["pose_series"])

    fixed_camera_evaluations = evaluate_fixed_camera_preservation(iterative_root)
    figure_paths["results_fixed_camera_metrics"] = plot_fixed_camera_metrics(fixed_camera_evaluations)
    figure_paths["results_fixed_camera_strip"] = str(FIGURES_DIR / "results_fixed_camera_rerender_strip.png")
    figure_paths["results_generated_views_strip"] = create_generated_views_strip(iterative_manifest, iterative_root)

    metrics_payload = {
        "same_pose_bridge": evaluation_summary["novel_view_regeneration"],
        "iterative": {
            "gap_series": iterative_metrics["gap_series"],
            "pose_series": iterative_metrics["pose_series"],
            "fixed_camera_preservation": [
                {
                    "iteration": item.iteration,
                    "label": item.label,
                    "image_paths": [str(path) for path in item.image_paths],
                    "rerender_path": item.rerender_path,
                    "psnr": item.psnr,
                    "ssim": item.ssim,
                    "lpips": item.lpips,
                }
                for item in fixed_camera_evaluations
            ],
        },
        "figure_paths": figure_paths,
    }
    save_json(DATA_DIR / "report_metrics.json", metrics_payload)
    print(json.dumps(metrics_payload, indent=2))


if __name__ == "__main__":
    main()
