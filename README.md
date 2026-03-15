# 3DCV Project: VGGT Novel-View Inpainting

This repository wraps a small 3D vision pipeline around a vendored copy of VGGT. The main flow is:

1. reconstruct geometry from one or two input images with VGGT,
2. render a nearby novel camera view from the recovered point cloud,
3. detect holes caused by missing geometry,
4. inpaint only those holes,
5. rerun VGGT with the synthetic view added.

## Project layout

- `utils.py`: shared pipeline code and the main `PipelineConfig`
- `scripts/test_utils_synthetic.py`: fast smoke test for rendering, masking, and fallback inpainting
- `scripts/test_pipeline_e2e.py`: full two-image pipeline on the images in `data/`
- `notebooks/vggt_coliseum_inpaint_pipeline.ipynb`: interactive notebook version of the full pipeline
- `data/`: sample inputs (`image_01.png`, `image_02.png`)
- `outputs/`: generated figures and intermediate artifacts
- `vendor/vggt/`: vendored VGGT package

## Quickstart

Use Python 3.10+ and run everything from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install the PyTorch wheel that matches your machine if needed.
python -m pip install torch torchvision

# Install the vendored VGGT package plus extra deps used by this project.
python -m pip install -e vendor/vggt
python -m pip install opencv-python matplotlib scipy requests accelerate transformers

# Needed for the local FLUX.2-klein backend used by the E2E script.
python -m pip install --upgrade --no-cache-dir git+https://github.com/huggingface/diffusers.git@main
```

Note: `utils.py` uses `Flux2KleinPipeline`. In the current environment, packaged `diffusers==0.32.2` did not expose that class, so the `main` branch install is required for the local FLUX backend.

## Fastest way to run it

Smoke test the geometry/rendering utilities:

```bash
python -m scripts.test_utils_synthetic
```

This writes debug images to `outputs/tests/synthetic/`.

Run the full two-image pipeline:

```bash
python -m scripts.test_pipeline_e2e
```

This uses the first two images in `data/`, downloads the VGGT and inpainting weights on first run, and saves outputs to `outputs/tests/e2e/`.

## What the full pipeline does

1. Run VGGT on the input views to recover cameras, depth, and point maps.
2. Merge the recovered 3D points into a point cloud.
3. Shift the camera slightly and render a novel projection.
4. Build a hole mask from pixels with no projected geometry.
5. Inpaint the masked regions while preserving unmasked pixels exactly.
6. Run VGGT again with the inpainted novel view included.

## Using your own images

- Put your images in `data/`, or change `PipelineConfig.data_dir`.
- `scripts/test_pipeline_e2e.py` uses the first two image files it finds in that folder.
- The single interactive notebook is [`notebooks/vggt_coliseum_inpaint_pipeline.ipynb`](notebooks/vggt_coliseum_inpaint_pipeline.ipynb).

If you want an interactive run:

```bash
python -m pip install jupyterlab
jupyter lab
```

## Configuration

Most knobs live in `PipelineConfig` in `utils.py` and are overridden near the top of the scripts/notebook:

- input/output paths: `data_dir`, `output_dir`
- novel camera motion: `novel_shift_right`, `novel_yaw_deg`, `novel_pitch_deg`, `novel_roll_deg`
- point cloud density: `vggt_conf_percentile`, `max_points_plot`, `max_points_render`
- hole-mask cleanup: `mask_dilate_px`, `mask_close_px`
- inpainting: `inpaint_model_id`, `inpaint_backend`, `inpaint_prompt`, `inpaint_steps`, `inpaint_guidance_scale`

Supported inpainting modes in `utils.py`:

- local FLUX.2-klein via diffusers: `black-forest-labs/FLUX.2-klein-4B`
- BFL API: model ids like `bfl:flux-2-klein-9b` plus `BFL_API_KEY`
- standard diffusers inpainting: e.g. `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`

## Important gotchas

- Run scripts as modules: `python -m scripts.test_utils_synthetic` and `python -m scripts.test_pipeline_e2e`.
- Do not run `python scripts/...py` directly; that makes `import utils` fail because the repo root is not on `sys.path`.
- Even the synthetic smoke test needs the vendored `vggt` package installed, because `utils.py` imports VGGT at module import time.
- CUDA or Apple MPS is strongly recommended for the full pipeline. CPU is fine for the synthetic smoke test but will be slow for model-based runs.
- There is no dedicated CLI yet; the notebook and the two scripts are the intended entry points.
