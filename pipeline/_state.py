"""Module-level caches and device utilities."""
from __future__ import annotations

import gc
import random

import numpy as np
import torch

_VGGT_MODEL_CACHE: dict[tuple[str, str], object] = {}
_INPAINT_PIPE_CACHE: dict[tuple[str, str, str], object] = {}
_LPIPS_MODEL_CACHE: dict[tuple[str, str], object] = {}


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def clear_torch_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def clear_loaded_model_caches(
    *,
    clear_vggt: bool = True,
    clear_inpaint: bool = True,
    clear_lpips: bool = True,
) -> None:
    if clear_vggt:
        _VGGT_MODEL_CACHE.clear()
    if clear_inpaint:
        _INPAINT_PIPE_CACHE.clear()
    if clear_lpips:
        _LPIPS_MODEL_CACHE.clear()
    clear_torch_cache()


def get_device(prefer_mps: bool = True) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
