"""Inpainting: hole masks, FLUX2-klein, FLUX Fill, OpenCV fallback."""
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


def _get_hf_token() -> str | None:
    """HuggingFace token for gated models (FLUX.1-Fill-dev, etc.). Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
from pipeline.io import pil_to_np_rgb, np_to_pil_rgb
from pipeline.viz import overlay_mask_on_image, to_pil_mask


def build_hole_mask_from_valid_mask(
    valid_mask: np.ndarray,
    close_px: int = 3,
    dilate_px: int = 1,
    min_area_px: int = 2,
    shrink_px: int | None = None,
    exterior_only: bool = False,
    all_invalid: bool = False,
) -> np.ndarray:
    """Build hole mask using flood-fill to classify interior vs exterior.

    1. Close the valid mask to bridge gaps between sparse rendered points.
    2. ``binary_fill_holes`` fills regions not connected to the image border,
       producing a solid object silhouette.
    3. Interior holes = inside the silhouette but missing valid data.

    If *all_invalid* is True, every pixel without projected geometry is masked
    (both interior holes and exterior background).
    """
    from scipy.ndimage import binary_fill_holes

    valid = np.asarray(valid_mask, dtype=bool)

    if all_invalid:
        hole_mask = (~valid).astype(np.uint8) * 255
    else:
        closed = valid.astype(np.uint8) * 255
        if close_px > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1),
            )
            closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, k)

        silhouette = binary_fill_holes(closed > 0)

        shrink = shrink_px if shrink_px is not None else close_px // 2
        if shrink > 0:
            ek = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * shrink + 1, 2 * shrink + 1),
            )
            silhouette = cv2.erode(
                silhouette.astype(np.uint8), ek,
            ).astype(bool)

        if exterior_only:
            hole_mask = (~silhouette & ~valid).astype(np.uint8) * 255
        else:
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


def prepare_novel_view_inpainting_inputs(
    render: RenderResult,
    close_px: int = 3,
    dilate_px: int = 1,
    min_area_px: int = 2,
    shrink_px: int | None = None,
    exterior_only: bool = False,
    all_invalid: bool = False,
    fill_mask_rgb: tuple[int, int, int] | None = None,
) -> tuple[Image.Image, Image.Image, Image.Image]:
    novel_img = render.image.convert("RGB")
    hole_mask_np = build_hole_mask_from_valid_mask(
        render.valid_mask,
        close_px=close_px,
        dilate_px=dilate_px,
        min_area_px=min_area_px,
        shrink_px=shrink_px,
        exterior_only=exterior_only,
        all_invalid=all_invalid,
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


def _looks_like_flux_fill(model_id: str) -> bool:
    mid = (model_id or "").strip().lower()
    return "flux.1-fill" in mid or "flux-fill" in mid


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

    pipe = Flux2KleinPipeline.from_pretrained(
        model_id, torch_dtype=dtype, token=_get_hf_token()
    )

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
    protect_mask: Image.Image | None = None,
    foreground_threshold: float = 20.0,
    erode_px: int = 2,
    num_inference_steps: int = 8,
    guidance_scale: float = 1.0,
    seed: int = 0,
) -> Image.Image:
    """Enhance image quality with FLUX.2-klein, preserving white background.

    The full image content is passed to Flux for quality enhancement (no
    hole-blanking). A foreground mask built from non-white pixels is used to
    composite the result so white background areas stay strictly unchanged.

    When *protect_mask* is provided (e.g. the hole mask), those pixels are
    additionally excluded from the foreground mask so Flux cannot touch them.
    The foreground mask is eroded by *erode_px* to avoid bleeding into
    white/hole edges.
    """
    device = device or get_device()
    image = image.convert("RGB")
    w, h = image.size

    img_np = np.asarray(image, dtype=np.float32)
    white_distance = np.linalg.norm(img_np - np.array([255.0, 255.0, 255.0]), axis=-1)
    fg_mask_np = (white_distance > foreground_threshold).astype(np.uint8) * 255

    if erode_px > 0:
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        fg_mask_np = cv2.erode(fg_mask_np, erode_k, iterations=1)

    if protect_mask is not None:
        protect_np = np.asarray(protect_mask.convert("L"), dtype=np.uint8)
        if protect_np.shape[:2] != fg_mask_np.shape[:2]:
            protect_pil = protect_mask.convert("L").resize((w, h), Image.Resampling.NEAREST)
            protect_np = np.asarray(protect_pil, dtype=np.uint8)
        fg_mask_np[protect_np > 0] = 0

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


MASK_DILATE_PX = 12  # Overlap into object so VAE 8×8 blocks don't straddle boundary

FLUX_FILL_MODEL = "black-forest-labs/FLUX.1-Fill-dev"


def inpaint_with_flux_fill(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    model_id: str = FLUX_FILL_MODEL,
    device: str | None = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 30.0,
    seed: int = 0,
    allow_fallback_to_opencv: bool = True,
) -> InpaintResult:
    """Inpaint using FLUX.1-Fill-dev — state-of-the-art inpainting (2024+)."""
    try:
        from diffusers import FluxFillPipeline
    except ImportError:
        raise ImportError(
            "FluxFillPipeline requires diffusers>=0.32.0. "
            "Upgrade: pip install -U diffusers"
        ) from None

    device = device or get_device()
    orig = image.convert("RGB")
    mask = mask.convert("L")

    if orig.size != mask.size:
        mask = mask.resize(orig.size, Image.Resampling.NEAREST)

    w, h = orig.size
    FLUX_MIN_SIDE = 1024
    short_side = min(w, h)
    if short_side < FLUX_MIN_SIDE:
        scale = FLUX_MIN_SIDE / short_side
        w, h = int(w * scale), int(h * scale)
        orig = orig.resize((w, h), Image.Resampling.LANCZOS)
        mask = mask.resize((w, h), Image.Resampling.NEAREST)
        print(f"  [inpaint] FLUX Fill: upscaled to {w}x{h} (min side {FLUX_MIN_SIDE})")

    h_aligned = max(8, (h // 8) * 8)
    w_aligned = max(8, (w // 8) * 8)
    pipe_image = orig.resize((w_aligned, h_aligned), Image.Resampling.LANCZOS) if (w_aligned, h_aligned) != (w, h) else orig
    pipe_mask = mask.resize((w_aligned, h_aligned), Image.Resampling.NEAREST) if (w_aligned, h_aligned) != (w, h) else mask

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    cache_key = (model_id, device, "flux_fill")
    if cache_key not in _INPAINT_PIPE_CACHE:
        hf_token = _get_hf_token()
        if not hf_token:
            print("  [inpaint] FLUX.1-Fill-dev is gated. Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN (huggingface-cli login).")
        print(f"  [inpaint] Loading FLUX.1-Fill-dev ({model_id}) ...")
        pipe = FluxFillPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=hf_token,
        ).to(device)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        _INPAINT_PIPE_CACHE[cache_key] = pipe
    pipe = _INPAINT_PIPE_CACHE[cache_key]

    # FLUX.1-Fill: avoid forcing high guidance (washed-out blur); clamp only if very low
    gs = max(guidance_scale, 7.0) if guidance_scale < 5 else guidance_scale

    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = pipe(
        prompt=prompt,
        image=pipe_image,
        mask_image=pipe_mask,
        height=h_aligned,
        width=w_aligned,
        guidance_scale=gs,
        num_inference_steps=num_inference_steps,
        max_sequence_length=512,
        generator=generator,
    )
    raw = result.images[0].convert("RGB")

    if (w_aligned, h_aligned) != (w, h):
        raw = raw.resize((w, h), Image.Resampling.LANCZOS)

    composited = _composite_preserve_unmasked(orig, raw, mask)
    return InpaintResult(
        raw_generated=raw,
        composited=composited,
        mask_used=mask,
        backend="flux_fill",
        resized_for_model=(w_aligned, h_aligned) != (w, h),
    )


def inpaint_holes_individually(
    image: Image.Image,
    mask: Image.Image,
    min_hole_area: int = 16,
    **kwargs: Any,
) -> InpaintResult:
    """Inpaint holes using FLUX Fill or OpenCV fallback. Uses dilated hole mask only."""
    hole_mask_np = (np.asarray(mask.convert("L")) > 0).astype(np.uint8) * 255

    # Dilate hole mask so inpaint region overlaps into object (avoids VAE block seams)
    if MASK_DILATE_PX > 0:
        kernel = np.ones((MASK_DILATE_PX * 2 + 1, MASK_DILATE_PX * 2 + 1), np.uint8)
        dilated_hole_np = cv2.dilate(hole_mask_np, kernel, iterations=1)
    else:
        dilated_hole_np = hole_mask_np

    mask_for_inpaint = Image.fromarray(dilated_hole_np, mode="L")
    mask_pct = (dilated_hole_np > 0).sum() / dilated_hole_np.size * 100
    print(f"  [inpaint] dilated hole mask ({mask_pct:.1f}%)")

    if mask_pct > 85:
        print(f"  [inpaint] mask too large ({mask_pct:.0f}%); using OpenCV fallback")
        raw = opencv_inpaint_fallback(image, mask, radius=5)
        composited = _composite_preserve_unmasked(image, raw, mask)
        return InpaintResult(
            raw_generated=raw, composited=composited, mask_used=mask,
            backend="opencv_telea_fallback", resized_for_model=False,
        )

    result = inpaint_with_diffusion(
        image=image,
        mask=mask_for_inpaint,
        **kwargs,
    )

    # Composite using dilated hole mask so we include the overlap zone (clean blending)
    dilated_hole_pil = Image.fromarray(dilated_hole_np, mode="L")
    composited = _composite_preserve_unmasked(image, result.raw_generated, dilated_hole_pil)

    return InpaintResult(
        raw_generated=result.raw_generated,
        composited=composited,
        mask_used=mask,
        backend=result.backend,
        resized_for_model=result.resized_for_model,
        metadata={"mask_pct": mask_pct},
    )


def inpaint_with_diffusion(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str | None = "blurry, distorted, duplicated structures, changed visible regions",
    model_id: str = FLUX_FILL_MODEL,
    device: str | None = None,
    reference_images: Iterable[Image.Image | str | Path] | None = None,
    backend: str = "auto",
    num_inference_steps: int = 20,
    guidance_scale: float = 6.5,
    strength: float = 0.95,
    padding_mask_crop: int | None = 128,
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
        elif _looks_like_flux_fill(model_id):
            backend = "flux_fill"
        else:
            backend = "opencv_fallback"  # SDXL removed; use FLUX.1-Fill-dev

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

    if backend == "opencv_fallback":
        print("[inpaint_with_diffusion] SDXL removed. Use FLUX.1-Fill-dev or --skip-sdxl for OpenCV.")
        image = image.convert("RGB")
        mask = mask.convert("L")
        raw = opencv_inpaint_fallback(image, mask, radius=5)
        composited = _composite_preserve_unmasked(image, raw, mask)
        return InpaintResult(
            raw_generated=raw, composited=composited, mask_used=mask,
            backend="opencv_telea_fallback", resized_for_model=False,
            metadata={"reason": "sdxl_removed"},
        )

    if backend == "flux_fill":
        try:
            return inpaint_with_flux_fill(
                image=image,
                mask=mask,
                prompt=prompt,
                model_id=model_id,
                device=device,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                allow_fallback_to_opencv=allow_fallback_to_opencv,
            )
        except Exception as exc:
            if not allow_fallback_to_opencv:
                raise
            err_str = str(exc)
            if "401" in err_str or "restricted" in err_str.lower() or "authenticated" in err_str.lower():
                print(
                    "[inpaint_with_diffusion] FLUX.1-Fill is gated. Set HF_TOKEN or run: huggingface-cli login. "
                    "Accept license at https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev"
                )
            print(f"[inpaint_with_diffusion] FLUX.1-Fill failed, falling back to OpenCV Telea: {exc}")
            image = image.convert("RGB")
            mask = mask.convert("L")
            raw = opencv_inpaint_fallback(image, mask)
            composited = _composite_preserve_unmasked(image, raw, mask)
            return InpaintResult(raw_generated=raw, composited=composited, mask_used=mask, backend="opencv_telea_fallback", resized_for_model=False, metadata={"flux_fill_error": str(exc)})

    raise RuntimeError(f"Unknown inpainting backend: {backend}")
