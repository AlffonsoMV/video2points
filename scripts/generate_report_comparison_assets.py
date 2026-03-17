from __future__ import annotations

import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils


REPORT_DIR = ROOT / "report"
FIGURES_DIR = REPORT_DIR / "figures"
DATA_DIR = REPORT_DIR / "data"

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
    }
)


@dataclass
class RunSummary:
    name: str
    root: Path
    manifest: dict[str, Any]
    stats: list[dict[str, Any]]

    @property
    def iterations_completed(self) -> int:
        return len(self.stats)

    @property
    def baseline_gap_deg(self) -> float | None:
        if not self.manifest.get("iterations"):
            return None
        return float(self.manifest["iterations"][0]["largest_gap_before_deg"])

    @property
    def final_gap_deg(self) -> float | None:
        if not self.manifest.get("iterations"):
            return None
        return float(self.manifest["iterations"][self.iterations_completed - 1]["largest_gap_after_deg"])

    @property
    def gap_series(self) -> list[dict[str, float]]:
        if not self.manifest.get("iterations"):
            return []
        series = [{"iteration": 0, "largest_unseen_gap_deg": float(self.manifest["iterations"][0]["largest_gap_before_deg"])}]
        for item in self.manifest["iterations"][: self.iterations_completed]:
            series.append(
                {
                    "iteration": int(item["iteration_index"]),
                    "largest_unseen_gap_deg": float(item["largest_gap_after_deg"]),
                }
            )
        return series

    @property
    def rotation_errors_deg(self) -> list[float]:
        return [
            float(stats["pose_and_coverage_after"]["target_pose_error"]["rotation_error_deg"])
            for stats in self.stats
        ]

    @property
    def total_cost(self) -> float:
        total = 0.0
        for stats in self.stats:
            metadata = stats.get("flux_output", {}).get("metadata", {}) or {}
            usage = metadata.get("usage", {}) or {}
            total += float(usage.get("cost") or 0.0)
        return total


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


def load_run(name: str, relative_root: str) -> RunSummary:
    root = ROOT / relative_root
    manifest = json.loads(require(root / "manifest.json").read_text())
    stats: list[dict[str, Any]] = []
    for item in manifest.get("iterations", []):
        stats_path = root / f"iter_{int(item['iteration_index']):02d}" / "stats.json"
        if not stats_path.exists():
            break
        stats.append(json.loads(stats_path.read_text()))
    return RunSummary(name=name, root=root, manifest=manifest, stats=stats)


def _load_image(path: str | Path):
    return utils.load_pil_rgb(path)


def _non_white_mask(image: np.ndarray, threshold: int = 245) -> np.ndarray:
    return ~(
        (image[..., 0] > threshold)
        & (image[..., 1] > threshold)
        & (image[..., 2] > threshold)
    )


def compute_supported_pair_fidelity(
    run: RunSummary,
    *,
    left_relative_path: str,
    right_relative_path: str,
) -> dict[str, Any]:
    mae_values: list[float] = []
    rmse_values: list[float] = []
    psnr_values: list[float] = []
    support_ratios: list[float] = []
    for item in run.manifest.get("iterations", [])[: run.iterations_completed]:
        iteration = int(item["iteration_index"])
        left_image = np.asarray(
            _load_image(run.root / f"iter_{iteration:02d}" / left_relative_path),
            dtype=np.float32,
        )
        right_image = np.asarray(
            _load_image(run.root / f"iter_{iteration:02d}" / right_relative_path),
            dtype=np.float32,
        )
        support_mask = _non_white_mask(left_image)
        if not np.any(support_mask):
            continue
        diff = right_image - left_image
        support_values = diff[support_mask]
        mae = float(np.abs(support_values).mean())
        rmse = float(np.sqrt((support_values**2).mean()))
        psnr = float(20.0 * math.log10(255.0 / max(rmse, 1e-8)))
        mae_values.append(mae)
        rmse_values.append(rmse)
        psnr_values.append(psnr)
        support_ratios.append(float(support_mask.mean()))
    return {
        "mae_mean": float(np.mean(mae_values)),
        "mae_series": mae_values,
        "rmse_mean": float(np.mean(rmse_values)),
        "rmse_series": rmse_values,
        "psnr_mean": float(np.mean(psnr_values)),
        "psnr_series": psnr_values,
        "support_ratio_mean": float(np.mean(support_ratios)),
        "support_ratio_series": support_ratios,
    }


def compute_target_support_fidelity(run: RunSummary) -> dict[str, Any]:
    return compute_supported_pair_fidelity(
        run,
        left_relative_path="02_target_pos2/pos2_render_before.png",
        right_relative_path="03_flux/generated_raw.png",
    )


def compute_rerender_support_fidelity(run: RunSummary) -> dict[str, Any]:
    return compute_supported_pair_fidelity(
        run,
        left_relative_path="02_target_pos2/pos2_render_before.png",
        right_relative_path="04_after/pos2_render_after.png",
    )


def save_comparison_grid(
    rows: list[list[str | Path]],
    row_labels: list[str],
    col_labels: list[str],
    out_name: str,
    figsize: tuple[float, float],
) -> str:
    n_rows = len(rows)
    n_cols = len(rows[0]) if rows else 0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = [axes]
    for row_idx, row in enumerate(rows):
        if n_cols == 1:
            row_axes = [axes[row_idx]]
        else:
            row_axes = axes[row_idx]
        for col_idx, path in enumerate(row):
            ax = row_axes[col_idx]
            ax.imshow(_load_image(path))
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(col_labels[col_idx], fontsize=11)
    if n_rows > 1:
        for row_idx, label in enumerate(row_labels):
            y = 1.0 - (row_idx + 0.5) / n_rows
            fig.text(0.015, y, label, rotation=90, va="center", ha="center", fontsize=11)
        fig.tight_layout(rect=(0.04, 0.0, 1.0, 1.0))
    else:
        fig.tight_layout()
    return str(utils.save_matplotlib_figure(fig, FIGURES_DIR / out_name))


def copy_asset(src: Path, dst_name: str) -> str:
    dst = FIGURES_DIR / dst_name
    shutil.copy2(require(src), dst)
    return str(dst)


def _triplet_paths(run: RunSummary, iteration: int) -> list[Path]:
    return [
        run.root / f"iter_{iteration:02d}" / "02_target_pos2" / "pos2_render_before.png",
        run.root / f"iter_{iteration:02d}" / "03_flux" / "generated_raw.png",
        run.root / f"iter_{iteration:02d}" / "04_after" / "pos2_render_after.png",
    ]


def _placeholder_tile(size: tuple[int, int], text: str) -> Image.Image:
    image = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((size[0] * 0.2, size[1] * 0.45), text, fill=(80, 80, 80))
    return image


def save_triplet_iteration_grid(
    flux_run: RunSummary,
    nano_run: RunSummary,
    *,
    iterations: list[int],
    out_name: str,
    scene_label: str,
) -> str:
    col_labels = [
        "FLUX target",
        "FLUX generated",
        "FLUX rerender",
        "Nano target",
        "Nano generated",
        "Nano rerender",
    ]
    fig, axes = plt.subplots(len(iterations), 6, figsize=(15.5, max(3.0 * len(iterations), 4.5)))
    if len(iterations) == 1:
        axes = [axes]
    for row_idx, iteration in enumerate(iterations):
        row_axes = axes[row_idx]
        flux_paths = _triplet_paths(flux_run, iteration) if iteration <= flux_run.iterations_completed else []
        nano_paths = _triplet_paths(nano_run, iteration) if iteration <= nano_run.iterations_completed else []
        tiles = [*flux_paths, *nano_paths]
        for col_idx, ax in enumerate(row_axes):
            if row_idx == 0:
                ax.set_title(col_labels[col_idx], fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"Iter {iteration}", fontsize=10)
            if col_idx < len(tiles) and Path(tiles[col_idx]).exists():
                image = _load_image(tiles[col_idx])
            else:
                image = _placeholder_tile((512, 384), "not\navailable")
            ax.imshow(image)
            ax.axis("off")
    fig.suptitle(scene_label, y=1.01, fontsize=14)
    fig.tight_layout()
    return str(utils.save_matplotlib_figure(fig, FIGURES_DIR / out_name))


def save_chair_triplet_grid(flux_run: RunSummary, nano_run: RunSummary) -> str:
    rows = [
        _triplet_paths(flux_run, 1),
        _triplet_paths(nano_run, 1),
    ]
    return save_comparison_grid(
        rows=rows,
        row_labels=["FLUX.2-klein", "Nano Banana 2"],
        col_labels=["Target sparse render", "Generated view", "VGGT rerender after update"],
        out_name="limitations_chair_generator_vs_reconstruction.png",
        figsize=(12, 8),
    )


def plot_average_metrics(runs: dict[str, RunSummary]) -> tuple[str, dict[str, Any]]:
    scene_names = {
        "plant": "Plant",
        "colosseum": "Colosseum",
        "pyramid": "Pyramid",
    }
    generators = {
        "flux": {"color": "#8a4f00", "label": "FLUX.2-klein"},
        "nanobanana": {"color": "#1f4f7a", "label": "Nano Banana 2"},
    }

    scene_run_map = {
        "plant": {"flux": runs["plant_flux"], "nanobanana": runs["plant_nanobanana"]},
        "colosseum": {"flux": runs["colosseum_flux"], "nanobanana": runs["colosseum_nanobanana"]},
        "pyramid": {"flux": runs["pyramid_flux"], "nanobanana": runs["pyramid_nanobanana"]},
    }

    target_fidelity_metrics: dict[str, dict[str, Any]] = {}
    rerender_fidelity_metrics: dict[str, dict[str, Any]] = {}
    for scene_key in scene_names:
        for generator_key in generators:
            run = scene_run_map[scene_key][generator_key]
            key = f"{scene_key}_{generator_key}"
            target_fidelity_metrics[key] = compute_target_support_fidelity(run)
            rerender_fidelity_metrics[key] = compute_rerender_support_fidelity(run)

    labels = [scene_names["plant"], scene_names["colosseum"], scene_names["pyramid"], "Average"]
    target_flux_mae = [target_fidelity_metrics[f"{scene_key}_flux"]["mae_mean"] for scene_key in scene_names]
    target_nano_mae = [target_fidelity_metrics[f"{scene_key}_nanobanana"]["mae_mean"] for scene_key in scene_names]
    rerender_flux_mae = [rerender_fidelity_metrics[f"{scene_key}_flux"]["mae_mean"] for scene_key in scene_names]
    rerender_nano_mae = [rerender_fidelity_metrics[f"{scene_key}_nanobanana"]["mae_mean"] for scene_key in scene_names]

    for series in [
        target_flux_mae,
        target_nano_mae,
        rerender_flux_mae,
        rerender_nano_mae,
    ]:
        series.append(float(np.mean(series)))

    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    target_mae_flux_bars = axes[0].bar(x - width / 2, target_flux_mae, width, color=generators["flux"]["color"], label=generators["flux"]["label"])
    target_mae_nano_bars = axes[0].bar(x + width / 2, target_nano_mae, width, color=generators["nanobanana"]["color"], label=generators["nanobanana"]["label"])
    rerender_mae_flux_bars = axes[1].bar(x - width / 2, rerender_flux_mae, width, color=generators["flux"]["color"], label=generators["flux"]["label"])
    rerender_mae_nano_bars = axes[1].bar(x + width / 2, rerender_nano_mae, width, color=generators["nanobanana"]["color"], label=generators["nanobanana"]["label"])

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_title("Render before -> generated")
    axes[0].set_ylabel("Supported-pixel MAE (lower is better)")
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].set_title("Render before -> render after")
    axes[1].set_ylabel("Supported-pixel MAE (lower is better)")

    target_avg_flux_mae = target_flux_mae[-1]
    target_avg_nano_mae = target_nano_mae[-1]
    rerender_avg_flux_mae = rerender_flux_mae[-1]
    rerender_avg_nano_mae = rerender_nano_mae[-1]

    target_mae_improvement_pct = 100.0 * (target_avg_flux_mae - target_avg_nano_mae) / target_avg_flux_mae
    rerender_mae_improvement_pct = 100.0 * (rerender_avg_flux_mae - rerender_avg_nano_mae) / rerender_avg_flux_mae

    axes[0].annotate(
        f"Nano: {target_mae_improvement_pct:.0f}% lower average error",
        xy=(x[-1] + width / 2, target_avg_nano_mae),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color=generators["nanobanana"]["color"],
        fontweight="bold",
    )
    axes[1].annotate(
        f"Nano: {rerender_mae_improvement_pct:.0f}% lower average error",
        xy=(x[-1] + width / 2, rerender_avg_nano_mae),
        xytext=(0, 12),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color=generators["nanobanana"]["color"],
        fontweight="bold",
    )

    for bars in [
        target_mae_flux_bars,
        target_mae_nano_bars,
        rerender_mae_flux_bars,
        rerender_mae_nano_bars,
    ]:
        for rect in bars:
            height = rect.get_height()
            ax = rect.axes
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height + 0.6,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("MAE on pixels already supported by the sparse target render", y=1.02, fontsize=14)
    fig.tight_layout()
    return (
        str(utils.save_matplotlib_figure(fig, FIGURES_DIR / "comparison_plant_metrics.png")),
        {
            "labels": labels,
            "target_to_generated": {
                "supported_pixel_mae": {
                    "flux": target_flux_mae,
                    "nanobanana": target_nano_mae,
                },
                "per_run": target_fidelity_metrics,
            },
            "target_to_rerender": {
                "supported_pixel_mae": {
                    "flux": rerender_flux_mae,
                    "nanobanana": rerender_nano_mae,
                },
                "per_run": rerender_fidelity_metrics,
            },
        },
    )


def build_metrics_payload(
    runs: dict[str, RunSummary],
    figure_paths: dict[str, str],
    fidelity_payload: dict[str, Any],
) -> dict[str, Any]:
    payload_runs: dict[str, Any] = {}
    for name, run in runs.items():
        payload_runs[name] = {
            "root": str(run.root),
            "subject": run.manifest.get("subject"),
            "iterations_completed": run.iterations_completed,
            "baseline_gap_deg": run.baseline_gap_deg,
            "final_gap_deg": run.final_gap_deg,
            "gap_series": run.gap_series,
            "rotation_errors_deg": run.rotation_errors_deg,
            "mean_rotation_error_deg": (
                sum(run.rotation_errors_deg) / len(run.rotation_errors_deg)
                if run.rotation_errors_deg
                else None
            ),
            "total_cost": run.total_cost or None,
        }
    return {"runs": payload_runs, "figure_paths": figure_paths, "generator_fidelity_metrics": fidelity_payload}


def main() -> None:
    ensure_dirs()

    runs = {
        "plant_flux": load_run("plant_flux", "outputs/tests/iterative_loop_plant_flux_report"),
        "plant_nanobanana": load_run("plant_nanobanana", "outputs/tests/iterative_loop_plant_nanobanana"),
        "chair_flux": load_run("chair_flux", "outputs/tests/iterative_loop_chair1_filtered_t30"),
        "chair_nanobanana": load_run("chair_nanobanana", "outputs/tests/iterative_loop_chair1_nanobanana"),
        "colosseum_flux": load_run("colosseum_flux", "outputs/tests/iterative_loop_colosseum_flux_report"),
        "colosseum_nanobanana": load_run("colosseum_nanobanana", "outputs/tests/iterative_loop_colosseum_nanobanana"),
        "pyramid_flux": load_run("pyramid_flux", "outputs/tests/iterative_loop_pyramid_flux_report"),
        "pyramid_nanobanana": load_run("pyramid_nanobanana", "outputs/tests/iterative_loop_pyramid_nanobanana"),
    }

    metrics_figure_path, fidelity_payload = plot_average_metrics(runs)

    figure_paths = {
        "comparison_plant_target_rerender_grid": save_triplet_iteration_grid(
            runs["plant_flux"],
            runs["plant_nanobanana"],
            iterations=[1, 2, 3, 4],
            out_name="comparison_plant_target_rerender_grid.png",
            scene_label="Plant: target render, generated view, and rerender after update",
        ),
        "comparison_plant_metrics": metrics_figure_path,
        "comparison_colosseum_flux_vs_nanobanana": save_triplet_iteration_grid(
            runs["colosseum_flux"],
            runs["colosseum_nanobanana"],
            iterations=[1, 2, 3, 4],
            out_name="comparison_colosseum_flux_vs_nanobanana.png",
            scene_label="Colosseum: target render, generated view, and rerender after update",
        ),
        "comparison_pyramid_flux_vs_nanobanana": save_triplet_iteration_grid(
            runs["pyramid_flux"],
            runs["pyramid_nanobanana"],
            iterations=[1, 2, 3],
            out_name="comparison_pyramid_flux_vs_nanobanana.png",
            scene_label="Pyramid: target render, generated view, and rerender after update",
        ),
        "limitations_chair_generator_vs_reconstruction": save_chair_triplet_grid(
            runs["chair_flux"],
            runs["chair_nanobanana"],
        ),
    }

    payload = build_metrics_payload(runs, figure_paths, fidelity_payload)
    save_json(DATA_DIR / "generator_comparison_metrics.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
