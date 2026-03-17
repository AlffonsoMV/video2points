"""LPIPS and image metrics."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from pipeline._state import _LPIPS_MODEL_CACHE
from pipeline.io import coerce_image_rgb


def load_lpips_model(net: str = "alex", device: str = "cpu", use_cache: bool = True) -> Any:
    import lpips

    cache_key = (net, device)
    if use_cache and cache_key in _LPIPS_MODEL_CACHE:
        return _LPIPS_MODEL_CACHE[cache_key]

    model = lpips.LPIPS(net=net)
    model.eval()
    model.to(device)

    if use_cache:
        _LPIPS_MODEL_CACHE[cache_key] = model
    return model


def compute_image_metrics(
    reference: Image.Image | str | Path,
    candidate: Image.Image | str | Path,
    lpips_net: str = "alex",
    lpips_device: str = "cpu",
) -> dict[str, Any]:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    ref_img = coerce_image_rgb(reference)
    cand_img = coerce_image_rgb(candidate)
    cand_original_size = cand_img.size
    resized_candidate = False
    if cand_img.size != ref_img.size:
        cand_img = cand_img.resize(ref_img.size, Image.Resampling.LANCZOS)
        resized_candidate = True

    ref_np = np.asarray(ref_img, dtype=np.float32) / 255.0
    cand_np = np.asarray(cand_img, dtype=np.float32) / 255.0

    psnr = float(peak_signal_noise_ratio(ref_np, cand_np, data_range=1.0))
    ssim = float(structural_similarity(ref_np, cand_np, channel_axis=2, data_range=1.0))

    lpips_model = load_lpips_model(net=lpips_net, device=lpips_device)
    ref_tensor = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0).to(lpips_device)
    cand_tensor = torch.from_numpy(cand_np).permute(2, 0, 1).unsqueeze(0).to(lpips_device)
    ref_tensor = ref_tensor * 2.0 - 1.0
    cand_tensor = cand_tensor * 2.0 - 1.0
    with torch.no_grad():
        lpips_value = float(lpips_model(ref_tensor, cand_tensor).item())

    return {
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips_value,
        "reference_size": list(ref_img.size),
        "candidate_original_size": list(cand_original_size),
        "candidate_resized_to_reference": resized_candidate,
    }
