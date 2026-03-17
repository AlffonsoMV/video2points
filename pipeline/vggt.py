"""VGGT model loading and reconstruction."""
from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from pipeline._state import _VGGT_MODEL_CACHE, get_device


def _cuda_autocast_context(device: str) -> Any:
    if device == "cuda":
        dtype = torch.float16
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                dtype = torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def load_vggt_model(
    model_id: str = "facebook/VGGT-1B",
    device: str | None = None,
    use_cache: bool = True,
) -> VGGT:
    device = device or get_device()
    cache_key = (model_id, device)
    if use_cache and cache_key in _VGGT_MODEL_CACHE:
        print(f"  [vggt] Using cached model on {device}")
        return _VGGT_MODEL_CACHE[cache_key]

    print(f"  [vggt] Downloading/loading model {model_id} ...")
    t0 = time.time()
    model = VGGT.from_pretrained(model_id)
    print(f"  [vggt] Model loaded in {time.time() - t0:.1f}s. Moving to {device} ...")
    model.eval()
    model.to(device)
    print(f"  [vggt] Ready on {device} ({time.time() - t0:.1f}s total)")

    if use_cache:
        _VGGT_MODEL_CACHE[cache_key] = model
    return model


def _squeeze_batch_tensors(predictions: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in predictions.items():
        if torch.is_tensor(value):
            tensor = value.detach().cpu()
            if tensor.ndim > 0 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            out[key] = tensor
        else:
            out[key] = value
    return out


def _tensor_image_stack_to_nhwc(images_tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    if torch.is_tensor(images_tensor):
        images_tensor = images_tensor.detach().cpu().numpy()
    arr = np.asarray(images_tensor)
    if arr.ndim == 4 and arr.shape[1] == 3:  # S,C,H,W
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected image stack in NCHW or NHWC, got shape {arr.shape}")
    return arr


def run_vggt_reconstruction(
    image_paths: list[str | Path],
    model: VGGT | None = None,
    device: str | None = None,
    model_id: str = "facebook/VGGT-1B",
    preprocess_mode: str = "crop",
) -> dict[str, Any]:
    device = device or get_device()
    image_paths = [Path(p) for p in image_paths]
    if not image_paths:
        raise ValueError("image_paths cannot be empty")

    if model is None:
        model = load_vggt_model(model_id=model_id, device=device)

    print(f"  [vggt] Preprocessing {len(image_paths)} images (mode={preprocess_mode}) ...")
    t0 = time.time()
    images = load_and_preprocess_images([str(p) for p in image_paths], mode=preprocess_mode)
    image_hw = tuple(int(x) for x in images.shape[-2:])
    print(f"  [vggt] Preprocessed to {images.shape} in {time.time() - t0:.1f}s")
    images_dev = images.to(device)

    print(f"  [vggt] Running inference on {device} ...")
    t1 = time.time()
    with torch.no_grad():
        with _cuda_autocast_context(device):
            predictions = model(images_dev)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], image_hw)
    print(f"  [vggt] Inference done in {time.time() - t1:.1f}s")

    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    squeezed = _squeeze_batch_tensors(predictions)
    result: dict[str, Any] = {
        "device": device,
        "image_paths": image_paths,
        "image_hw": image_hw,
        "preprocessed_shape": tuple(int(x) for x in images.shape),
        "preprocess_mode": preprocess_mode,
    }

    for key, value in squeezed.items():
        if torch.is_tensor(value):
            result[key] = value.numpy()
        else:
            result[key] = value

    if "images" in result:
        result["images_nhwc"] = _tensor_image_stack_to_nhwc(result["images"])

    if "depth" in result and "extrinsic" in result and "intrinsic" in result:
        result["world_points_from_depth"] = unproject_depth_map_to_point_map(
            result["depth"], result["extrinsic"], result["intrinsic"]
        )

    return result
