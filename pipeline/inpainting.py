"""Inpainting: hole masks, FLUX2-klein, SDXL, OpenCV fallback."""
from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from PIL import Image

from pipeline._state import _INPAINT_PIPE_CACHE, get_device
from pipeline.config import InpaintResult, RenderResult
from pipeline.io import pil_to_np_rgb, np_to_pil_rgb
from pipeline.viz import overlay_mask_on_image, to_pil_mask


def build_hole_mask_from_valid_mask(
    valid_mask: np.ndarray,
    dilate_px: int = 4,
    close_px: int = 3,
    min_area_px: int = 16,
    exterior_only: bool = False,
    interior_only: bool = False,
    support_close_px: int = 0,
    support_dilate_px: int = 0,
    open_px: int = 0,
    gap_break_px: int = 2,
) -> np.ndarray:
    valid = np.asarray(valid_mask, dtype=bool)

    # Start with all holes
    hole_mask = (~valid).astype(np.uint8) * 255

    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_px + 1, 2 * open_px + 1))
        hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_OPEN, k)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
        hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_CLOSE, k)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        hole_mask = cv2.dilate(hole_mask, k)

    # Don't mask over actual valid pixels
    hole_mask[valid] = 0

    # Filter by interior/exterior: check which hole components touch the image border
    if interior_only or exterior_only:
        # Opening (erode→dilate) on the mask breaks thin channels that
        # falsely connect interior holes to the border, without affecting
        # the bulk shape of real holes.
        classify_mask = hole_mask
        if gap_break_px > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * gap_break_px + 1, 2 * gap_break_px + 1),
            )
            classify_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_OPEN, k)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (classify_mask > 0).astype(np.uint8), connectivity=8,
        )
        border_labels = set(np.unique(np.concatenate([
            labels[0], labels[-1], labels[:, 0], labels[:, -1],
        ])))
        border_labels.discard(0)

        # Build a region mask from the interior/exterior classification
        interior_region = np.zeros_like(hole_mask)
        for label in range(1, num_labels):
            touches_border = label in border_labels
            if stats[label, cv2.CC_STAT_AREA] < min_area_px:
                continue
            keep = (interior_only and not touches_border) or (exterior_only and touches_border)
            if keep:
                interior_region[labels == label] = 255

        if gap_break_px > 0 and np.any(interior_region):
            # Dilate the classified regions back to recover original hole
            # boundary pixels that were eroded away by the opening step.
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * (gap_break_px + 2) + 1, 2 * (gap_break_px + 2) + 1),
            )
            interior_region = cv2.dilate(interior_region, k)
            return (hole_mask & interior_region).astype(np.uint8)

        return interior_region

    # No interior/exterior filtering — just min area
    if min_area_px > 1:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (hole_mask > 0).astype(np.uint8), connectivity=8,
        )
        cleaned = np.zeros_like(hole_mask)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= min_area_px:
                cleaned[labels == label] = 255
        hole_mask = cleaned
    return hole_mask


def prepare_novel_view_inpainting_inputs(
    render: RenderResult,
    dilate_px: int = 4,
    close_px: int = 3,
    min_area_px: int = 16,
    exterior_only: bool = False,
    interior_only: bool = False,
    open_px: int = 0,
    support_close_px: int = 0,
    support_dilate_px: int = 0,
    gap_break_px: int = 2,
    fill_mask_rgb: tuple[int, int, int] | None = None,
    feather_px: int = 3,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    novel_img = render.image.convert("RGB")
    hole_mask_np = build_hole_mask_from_valid_mask(
        render.valid_mask,
        dilate_px=dilate_px,
        close_px=close_px,
        min_area_px=min_area_px,
        exterior_only=exterior_only,
        interior_only=interior_only,
        open_px=open_px,
        support_close_px=support_close_px,
        support_dilate_px=support_dilate_px,
        gap_break_px=gap_break_px,
    )

    if feather_px > 0:
        k_size = 2 * feather_px + 1
        hole_mask_np = cv2.GaussianBlur(
            hole_mask_np, (k_size, k_size), 0,
        )

    if fill_mask_rgb is not None:
        masked_base_np = np.asarray(novel_img, dtype=np.uint8).copy()
        masked_base_np[hole_mask_np > 0] = np.asarray(fill_mask_rgb, dtype=np.uint8)
        novel_img = Image.fromarray(masked_base_np, mode="RGB")
    hole_mask = to_pil_mask(hole_mask_np)
    overlay = overlay_mask_on_image(novel_img, hole_mask)
    return novel_img, hole_mask, overlay


def _resize_to_multiple(image: Image.Image, mask: Image.Image, multiple: int = 8) -> tuple[Image.Image, Image.Image, bool]:
    w, h = image.size
    new_w = max(multiple, (w // multiple) * multiple)
    new_h = max(multiple, (h // multiple) * multiple)
    if new_w == w and new_h == h:
        return image, mask, False
    return (
        image.resize((new_w, new_h), Image.Resampling.LANCZOS),
        mask.resize((new_w, new_h), Image.Resampling.NEAREST),
        True,
    )


def _resize_to_multiple_of_8(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image, bool]:
    return _resize_to_multiple(image, mask, multiple=8)


def _composite_preserve_unmasked(
    original: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    orig = pil_to_np_rgb(original)
    gen = pil_to_np_rgb(generated)
    m = np.asarray(mask.convert("L")) > 0
    out = orig.copy()
    out[m] = gen[m]
    return np_to_pil_rgb(out)


def _pil_to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _looks_like_bfl_flux2_klein(model_id: str) -> bool:
    mid = (model_id or "").strip().lower()
    return (
        mid.startswith("bfl:flux-2-klein-")
        or mid.startswith("flux-2-klein-")
        or "flux.2-klein" in mid
    )


def _normalize_bfl_flux2_klein_endpoint(model_id: str) -> str:
    mid = (model_id or "").strip()
    if mid.startswith("bfl:"):
        mid = mid[len("bfl:"):]
    if not mid.startswith("flux-2-klein-"):
        raise ValueError(f"Unsupported FLUX2-klein model id '{model_id}'. Expected e.g. 'bfl:flux-2-klein-9b'.")
    return mid


def opencv_inpaint_fallback(image: Image.Image, mask: Image.Image, radius: int = 3) -> Image.Image:
    img_np = cv2.cvtColor(pil_to_np_rgb(image), cv2.COLOR_RGB2BGR)
    mask_np = np.asarray(mask.convert("L"))
    out = cv2.inpaint(img_np, mask_np, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return np_to_pil_rgb(out)


def _masked_region_degenerate(image: Image.Image, mask: Image.Image) -> bool:
    """Detect degenerate SDXL output: masked region is nearly all-black or all-white."""
    mask_np = np.asarray(mask.convert("L")) > 0
    if not np.any(mask_np):
        return False
    img_np = np.asarray(image.convert("RGB"), dtype=np.float32)
    region = img_np[mask_np]
    if region.size == 0:
        return False
    mean, std = float(region.mean()), float(region.std())
    near_black = mean < 2.0 and std < 2.0
    near_white = mean > 253.0 and std < 2.0
    return near_black or near_white


def _load_reference_images(reference_images: Iterable[Image.Image | str | Path] | None) -> list[Image.Image]:
    if reference_images is None:
        return []
    refs: list[Image.Image] = []
    for ref in reference_images:
        if isinstance(ref, Image.Image):
            refs.append(ref.convert("RGB"))
        else:
            refs.append(Image.open(ref).convert("RGB"))
    return refs


def load_flux2_klein_pipeline(
    model_id: str = "black-forest-labs/FLUX.2-klein-4B",
    device: str | None = None,
    use_cache: bool = True,
) -> Any:
    import diffusers

    device = device or get_device()
    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    cache_key = (f"flux2_klein::{model_id}", device, str(dtype))
    if use_cache and cache_key in _INPAINT_PIPE_CACHE:
        print(f"  [inpaint] Using cached FLUX2-klein pipeline on {device}")
        return _INPAINT_PIPE_CACHE[cache_key]

    print(f"  [inpaint] Loading FLUX2-klein pipeline {model_id} (dtype={dtype}) ...")
    t0 = time.time()
    Flux2KleinPipeline = getattr(diffusers, "Flux2KleinPipeline", None)
    if Flux2KleinPipeline is None:
        raise RuntimeError(
            "Local FLUX.2-klein requires diffusers main branch (Flux2KleinPipeline is missing). "
            "Install with: pip install --upgrade --no-cache-dir git+https://github.com/huggingface/diffusers.git@main"
        )

    pipe = Flux2KleinPipeline.from_pretrained(model_id, torch_dtype=dtype)

    # GPU-only offload helpers are not useful on MPS/CPU; only enable on CUDA.
    if device == "cuda":
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pipe.to(device)
    else:
        pipe.to(device)

    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    print(f"  [inpaint] FLUX2-klein pipeline ready in {time.time() - t0:.1f}s")
    if use_cache:
        _INPAINT_PIPE_CACHE[cache_key] = pipe
    return pipe


def edit_with_flux2_klein_local(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str | None = None,  # kept for API symmetry; may be unsupported by local Flux2Klein pipeline
    model_id: str = "black-forest-labs/FLUX.2-klein-4B",
    device: str | None = None,
    reference_images: Iterable[Image.Image | str | Path] | None = None,
    num_inference_steps: int = 4,
    guidance_scale: float = 1.0,
    seed: int = 0,
) -> InpaintResult:
    del negative_prompt

    device = device or get_device()
    image = image.convert("RGB")
    mask = mask.convert("L")
    refs = _load_reference_images(reference_images)

    # FLUX2 docs/BFL docs mention image inputs are adjusted to multiples of 16. Pre-align ourselves.
    image_for_pipe, mask_for_pipe, resized = _resize_to_multiple(image, mask, multiple=16)
    # Encourage the edit model to focus on the holes by zeroing them in the base input while preserving context elsewhere.
    masked_base = _composite_preserve_unmasked(
        image_for_pipe,
        Image.new("RGB", image_for_pipe.size, (255, 255, 255)),
        mask_for_pipe,
    )
    refs_for_pipe = [
        ref.resize(image_for_pipe.size, Image.Resampling.LANCZOS) if ref.size != image_for_pipe.size else ref
        for ref in refs
    ]

    pipe = load_flux2_klein_pipeline(model_id=model_id, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    image_arg: list[Image.Image] | Image.Image
    if refs_for_pipe:
        # Flux2KleinPipeline supports multi-reference editing by passing a list of images.
        image_arg = [masked_base, *refs_for_pipe]
    else:
        image_arg = masked_base

    result = pipe(
        prompt=prompt,
        image=image_arg,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
    )
    raw = result.images[0].convert("RGB")
    if resized:
        raw = raw.resize(image.size, Image.Resampling.LANCZOS)
    composited = _composite_preserve_unmasked(image, raw, mask)
    return InpaintResult(
        raw_generated=raw,
        composited=composited,
        mask_used=mask,
        backend="flux2_klein_local_diffusers",
        resized_for_model=resized,
        metadata={"reference_count": len(refs_for_pipe), "model_id": model_id},
    )


def refine_image_with_flux2_klein(
    image: Image.Image,
    prompt: str = (
        "Photorealistic image. Slight softening for natural look, enhance realism. "
        "Preserve the exact same geometry, colors, composition, and content. "
        "Keep white background and surroundings white. Do not add or remove objects."
    ),
    model_id: str = "black-forest-labs/FLUX.2-klein-4B",
    device: str | None = None,
    reference_images: Iterable[Image.Image | str | Path] | None = None,
    mask: Image.Image | None = None,
    num_inference_steps: int = 4,
    guidance_scale: float = 1.0,
    seed: int = 0,
) -> Image.Image:
    """Use FLUX.2-klein to refine a single image for photorealism.

    Optionally uses reference_images (e.g. original video frames) for style guidance.
    When mask is provided and non-empty, FLUX only edits the masked regions (inpaint mode),
    leaving the rest of the image unchanged. Returns the refined image (composited when masked).
    """
    device = device or get_device()
    image = image.convert("RGB")
    w, h = image.size
    if mask is not None:
        mask = mask.convert("L")
        if np.asarray(mask).max() == 0:
            mask = Image.new("L", (w, h), 0)
    else:
        mask = Image.new("L", (w, h), 0)

    refs = _load_reference_images(reference_images)
    # When no refs: pass [] so FLUX gets only the base image and must generate from context.
    # Using [image] as fallback caused FLUX to copy the input (no real generation).
    refs_for_pipe = list(refs) if refs else []

    ref_result = edit_with_flux2_klein_local(
        image=image,
        mask=mask,
        prompt=prompt,
        model_id=model_id,
        device=device,
        reference_images=refs_for_pipe,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    # When mask has holes, composited pastes generated only into masked regions; otherwise use raw.
    mask_max = np.asarray(mask).max()
    return ref_result.composited if mask_max > 0 else ref_result.raw_generated


def enhance_quality_with_flux2_klein(
    image: Image.Image,
    prompt: str = (
        "Enhance this image to photorealistic quality with sharp, crisp details "
        "and natural textures. Preserve the exact geometry, composition, viewpoint, "
        "and object identity. Do not add or remove objects."
    ),
    model_id: str = "black-forest-labs/FLUX.2-klein-4B",
    device: str | None = None,
    reference_images: Iterable[Image.Image | str | Path] | None = None,
    foreground_threshold: float = 20.0,
    num_inference_steps: int = 8,
    guidance_scale: float = 1.0,
    seed: int = 0,
) -> Image.Image:
    """Enhance image quality with FLUX.2-klein, preserving white background.

    The full image content is passed to Flux for quality enhancement (no
    hole-blanking). A foreground mask built from non-white pixels is used to
    composite the result so white background areas stay unchanged.
    """
    device = device or get_device()
    image = image.convert("RGB")
    w, h = image.size

    img_np = np.asarray(image, dtype=np.float32)
    white_distance = np.linalg.norm(img_np - np.array([255.0, 255.0, 255.0]), axis=-1)
    fg_mask_np = (white_distance > foreground_threshold).astype(np.uint8) * 255

    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask_np = cv2.dilate(fg_mask_np, dilate_k, iterations=1)

    if fg_mask_np.max() == 0:
        return image

    refs = _load_reference_images(reference_images)

    empty_mask = Image.new("L", (w, h), 0)
    result = edit_with_flux2_klein_local(
        image=image,
        mask=empty_mask,
        prompt=prompt,
        model_id=model_id,
        device=device,
        reference_images=refs if refs else None,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    enhanced = result.raw_generated

    enhanced_np = np.asarray(enhanced.convert("RGB"), dtype=np.uint8)
    orig_np = np.asarray(image, dtype=np.uint8)
    fg_bool = fg_mask_np > 0
    out_np = orig_np.copy()
    out_np[fg_bool] = enhanced_np[fg_bool]

    return Image.fromarray(out_np, mode="RGB")


def inpaint_with_flux2_klein_bfl(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str | None = None,  # kept for API symmetry; currently not used by BFL FLUX2-klein API
    model_id: str = "bfl:flux-2-klein-9b",
    reference_images: Iterable[Image.Image | str | Path] | None = None,
    seed: int = 0,
    num_inference_steps: int = 20,
    guidance_scale: float = 6.5,
    bfl_api_key: str | None = None,
    bfl_api_key_env: str = "BFL_API_KEY",
    bfl_api_base_url: str = "https://api.bfl.ai",
    bfl_poll_interval_seconds: float = 1.0,
    bfl_timeout_seconds: int = 600,
    bfl_safety_tolerance: int = 2,
    bfl_output_format: str = "png",
) -> InpaintResult:
    import requests

    del negative_prompt  # not supported in the BFL FLUX2-klein editing endpoint (as documented)

    image = image.convert("RGB")
    mask = mask.convert("L")
    # BFL notes input images may be cropped/resized to multiples of 16; pre-align ourselves.
    image_for_api, mask_for_api, resized = _resize_to_multiple(image, mask, multiple=16)

    endpoint = _normalize_bfl_flux2_klein_endpoint(model_id)
    api_key = bfl_api_key or os.getenv(bfl_api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing BFL API key. Set {bfl_api_key_env} or pass bfl_api_key to use {endpoint}."
        )

    refs = _load_reference_images(reference_images)
    # BFL docs note FLUX2-klein supports up to 4 reference images via API.
    # We use input_image as the editable base image, then add up to 3 extra references.
    max_extra_refs = 3
    if len(refs) > max_extra_refs:
        print(f"[inpaint_with_flux2_klein_bfl] Truncating references from {len(refs)} to {max_extra_refs} for FLUX2-klein API.")
        refs = refs[:max_extra_refs]

    # Resize references to match the edited image resolution for more stable alignment/style transfer.
    refs_for_api = [ref.resize(image_for_api.size, Image.Resampling.LANCZOS) if ref.size != image_for_api.size else ref for ref in refs]

    payload: dict[str, Any] = {
        "prompt": prompt,
        "input_image": _pil_to_base64_png(image_for_api),
        "seed": int(seed),
        "steps": int(num_inference_steps),
        "guidance": float(guidance_scale),
        "safety_tolerance": int(bfl_safety_tolerance),
        "output_format": bfl_output_format,
        "prompt_upsampling": False,
    }
    for idx, ref in enumerate(refs_for_api, start=2):
        payload[f"input_image_{idx}"] = _pil_to_base64_png(ref)

    base_url = bfl_api_base_url.rstrip("/")
    url = f"{base_url}/v1/{endpoint}"
    headers = {"x-key": api_key, "accept": "application/json", "content-type": "application/json"}

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    start_data = resp.json()
    polling_url = start_data.get("polling_url")
    if not polling_url:
        raise RuntimeError(f"BFL FLUX2-klein start response missing polling_url: {start_data}")
    if polling_url.startswith("/"):
        polling_url = f"{base_url}{polling_url}"

    t0 = time.time()
    last_data: dict[str, Any] | None = None
    while True:
        if time.time() - t0 > float(bfl_timeout_seconds):
            raise TimeoutError(f"BFL FLUX2-klein request timed out after {bfl_timeout_seconds}s ({polling_url})")

        poll = requests.get(polling_url, headers={"x-key": api_key, "accept": "application/json"}, timeout=60)
        poll.raise_for_status()
        last_data = poll.json()
        status = str(last_data.get("status", "")).strip().lower()

        if status == "ready":
            break
        if status in {"error", "failed", "request_moderated", "content_moderated", "moderated"}:
            raise RuntimeError(f"BFL FLUX2-klein failed with status='{last_data.get('status')}'. Payload: {last_data}")
        if status in {"pending", "processing", "running"}:
            time.sleep(float(bfl_poll_interval_seconds))
            continue

        # Unknown status: keep polling unless a result already exists.
        if "result" in last_data and isinstance(last_data["result"], dict) and last_data["result"].get("sample"):
            break
        time.sleep(float(bfl_poll_interval_seconds))

    result_data = (last_data or {}).get("result", {})
    sample_url = result_data.get("sample")
    if not sample_url:
        raise RuntimeError(f"BFL FLUX2-klein result missing result.sample. Payload: {last_data}")

    img_resp = requests.get(sample_url, timeout=120)
    img_resp.raise_for_status()
    raw = Image.open(io.BytesIO(img_resp.content)).convert("RGB")

    if resized:
        raw = raw.resize(image.size, Image.Resampling.LANCZOS)
    composited = _composite_preserve_unmasked(image, raw, mask)
    return InpaintResult(
        raw_generated=raw,
        composited=composited,
        mask_used=mask,
        backend=f"bfl_{endpoint}_api",
        resized_for_model=resized,
        metadata={
            "endpoint": endpoint,
            "polling_url": polling_url,
            "result": last_data,
            "sample_url": sample_url,
            "reference_count": len(refs_for_api),
        },
    )


def load_inpainting_pipeline(
    model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    device: str | None = None,
    use_cache: bool = True,
) -> Any:
    from diffusers import AutoPipelineForInpainting

    device = device or get_device()
    # MPS is often more stable with float32 for SD inpainting than float16.
    dtype = torch.float16 if device == "cuda" else torch.float32
    cache_key = (model_id, device, str(dtype))
    if use_cache and cache_key in _INPAINT_PIPE_CACHE:
        return _INPAINT_PIPE_CACHE[cache_key]

    def _load_pipe(**kwargs: Any) -> Any:
        return AutoPipelineForInpainting.from_pretrained(model_id, **kwargs)

    pipe = None
    load_errors: list[str] = []
    base_kwargs = {"torch_dtype": dtype, "safety_checker": None, "requires_safety_checker": False}
    if dtype == torch.float16:
        try:
            pipe = _load_pipe(**base_kwargs, variant="fp16")
        except Exception as exc:
            load_errors.append(f"fp16 variant load failed: {exc}")
    if pipe is None:
        try:
            pipe = _load_pipe(**base_kwargs)
        except Exception as exc:
            load_errors.append(f"default dtype load failed: {exc}")
            pipe = _load_pipe(torch_dtype=torch.float32)
            dtype = torch.float32

    pipe = pipe.to(device)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "requires_safety_checker"):
        pipe.requires_safety_checker = False
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    if use_cache:
        _INPAINT_PIPE_CACHE[cache_key] = pipe
    if load_errors:
        print("[load_inpainting_pipeline] warnings:")
        for err in load_errors:
            print(" -", err)
    return pipe


def inpaint_with_diffusion(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str | None = "blurry, distorted, duplicated structures, changed visible regions",
    model_id: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    device: str | None = None,
    reference_images: Iterable[Image.Image | str | Path] | None = None,
    backend: str = "auto",
    num_inference_steps: int = 20,
    guidance_scale: float = 6.5,
    strength: float = 0.95,
    padding_mask_crop: int | None = 32,
    seed: int = 0,
    allow_fallback_to_opencv: bool = True,
    bfl_api_key: str | None = None,
    bfl_api_key_env: str = "BFL_API_KEY",
    bfl_api_base_url: str = "https://api.bfl.ai",
    bfl_poll_interval_seconds: float = 1.0,
    bfl_timeout_seconds: int = 600,
    bfl_safety_tolerance: int = 2,
    bfl_output_format: str = "png",
) -> InpaintResult:
    device = device or get_device()
    if backend == "auto":
        if _looks_like_bfl_flux2_klein(model_id):
            if str(model_id).startswith("bfl:"):
                backend = "flux2_klein_bfl"
            else:
                backend = "flux2_klein_local"
        else:
            backend = "diffusers"

    print(f"  [inpaint] backend={backend}, model={model_id}, steps={num_inference_steps}")
    t0 = time.time()

    if backend == "flux2_klein_bfl":
        return inpaint_with_flux2_klein_bfl(
            image=image,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_id=model_id,
            reference_images=reference_images,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            bfl_api_key=bfl_api_key,
            bfl_api_key_env=bfl_api_key_env,
            bfl_api_base_url=bfl_api_base_url,
            bfl_poll_interval_seconds=bfl_poll_interval_seconds,
            bfl_timeout_seconds=bfl_timeout_seconds,
            bfl_safety_tolerance=bfl_safety_tolerance,
            bfl_output_format=bfl_output_format,
        )

    if backend == "flux2_klein_local":
        try:
            return edit_with_flux2_klein_local(
                image=image,
                mask=mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                model_id=model_id,
                device=device,
                reference_images=reference_images,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
            )
        except Exception as exc:
            if not allow_fallback_to_opencv:
                raise
            print(f"[inpaint_with_diffusion] Local FLUX2-klein failed, falling back to OpenCV Telea: {exc}")
            image = image.convert("RGB")
            mask = mask.convert("L")
            raw = opencv_inpaint_fallback(image, mask)
            composited = _composite_preserve_unmasked(image, raw, mask)
            return InpaintResult(raw_generated=raw, composited=composited, mask_used=mask, backend="opencv_telea_fallback", resized_for_model=False, metadata={"flux2_error": str(exc)})

    image = image.convert("RGB")
    mask = mask.convert("L")

    # SDXL inpainting was trained with WHITE in holes; gray/other fills are OOD and cause the model
    # to preserve input instead of generating. Ensure white in masked regions before calling pipeline.
    image_for_sdxl = _composite_preserve_unmasked(
        image,
        Image.new("RGB", image.size, (255, 255, 255)),
        mask,
    )

    # SDXL was trained at 1024x1024. Small images (e.g. 294x518) produce degenerate output.
    # Upscale so the short side is at least 1024, then downscale the result back.
    SDXL_MIN_SIDE = 1024
    orig_size = image.size  # (w, h)
    w, h = orig_size
    short_side = min(w, h)
    upscaled_for_sdxl = False
    if short_side < SDXL_MIN_SIDE:
        scale = SDXL_MIN_SIDE / short_side
        new_w, new_h = int(w * scale), int(h * scale)
        image_for_sdxl = image_for_sdxl.resize((new_w, new_h), Image.Resampling.LANCZOS)
        mask_for_sdxl = mask.resize((new_w, new_h), Image.Resampling.NEAREST)
        upscaled_for_sdxl = True
        print(f"  [inpaint] Upscaling {w}x{h} → {new_w}x{new_h} for SDXL (trained at 1024x1024)")
    else:
        mask_for_sdxl = mask

    image_for_pipe, mask_for_pipe, resized = _resize_to_multiple_of_8(image_for_sdxl, mask_for_sdxl)

    try:
        pipe = load_inpainting_pipeline(model_id=model_id, device=device)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        pipe_kw: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": image_for_pipe,
            "mask_image": mask_for_pipe,
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "strength": float(strength),
            "generator": generator,
        }
        if padding_mask_crop is not None:
            pipe_kw["padding_mask_crop"] = int(padding_mask_crop)
        result = pipe(**pipe_kw)
        raw = result.images[0].convert("RGB")
        if resized or upscaled_for_sdxl:
            raw = raw.resize(orig_size, Image.Resampling.LANCZOS)
        composited = _composite_preserve_unmasked(image, raw, mask)
        # MPS can produce degenerate (all-black or all-white) output; retry on CPU
        if device == "mps" and _masked_region_degenerate(raw, mask):
            print("  [inpaint] Degenerate MPS output detected, retrying on CPU.")
            return inpaint_with_diffusion(
                image=image,
                mask=mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                model_id=model_id,
                device="cpu",
                backend="diffusers",
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                padding_mask_crop=padding_mask_crop,
                seed=seed,
                allow_fallback_to_opencv=allow_fallback_to_opencv,
            )
        return InpaintResult(raw_generated=raw, composited=composited, mask_used=mask, backend="diffusers", resized_for_model=resized or upscaled_for_sdxl)
    except Exception as exc:
        if not allow_fallback_to_opencv:
            raise
        print(f"[inpaint_with_diffusion] Diffusion inpainting failed, falling back to OpenCV Telea: {exc}")
        raw = opencv_inpaint_fallback(image, mask)
        composited = _composite_preserve_unmasked(image, raw, mask)
        return InpaintResult(raw_generated=raw, composited=composited, mask_used=mask, backend="opencv_telea_fallback", resized_for_model=resized)
