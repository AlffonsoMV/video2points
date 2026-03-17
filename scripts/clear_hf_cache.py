#!/usr/bin/env python3
"""Clear HuggingFace cache for unused models to free disk space.

Usage:
  python -m scripts.clear_hf_cache --dry-run   # show what would be deleted
  python -m scripts.clear_hf_cache            # delete SDXL and other unused caches

Frees space for FLUX.1-Fill-dev (~33GB) by removing:
  - SDXL inpainting (~7GB)
  - With --remove-partial: FLUX.1-Fill-dev cache (use if download failed due to disk full)
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


HF_HUB_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

# Models we no longer use (SDXL removed from pipeline)
UNUSED_MODELS = [
    "models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1",
]

# Partial/incomplete FLUX.1-Fill-dev (from failed downloads when disk full)
FLUX_FILL_CACHE = "models--black-forest-labs--FLUX.1-Fill-dev"


def main() -> None:
    parser = argparse.ArgumentParser(description="Clear HuggingFace cache for unused models")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    parser.add_argument("--remove-partial", action="store_true",
                        help="Also remove incomplete FLUX.1-Fill-dev downloads (use if disk full)")
    args = parser.parse_args()

    if not HF_HUB_CACHE.exists():
        print(f"Cache dir not found: {HF_HUB_CACHE}")
        return

    total_freed = 0
    for name in UNUSED_MODELS:
        path = HF_HUB_CACHE / name
        if path.exists():
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            total_freed += size
            if args.dry_run:
                print(f"Would remove: {path} ({size / 1e9:.1f} GB)")
            else:
                shutil.rmtree(path)
                print(f"Removed: {path} ({size / 1e9:.1f} GB)")

    if args.remove_partial:
        path = HF_HUB_CACHE / FLUX_FILL_CACHE
        if path.exists():
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            total_freed += size
            if args.dry_run:
                print(f"Would remove FLUX.1-Fill-dev cache: {path} ({size / 1e9:.1f} GB)")
            else:
                shutil.rmtree(path)
                print(f"Removed FLUX.1-Fill-dev cache: {path} ({size / 1e9:.1f} GB)")

    if total_freed == 0 and not args.remove_partial:
        print("No unused cache found." if args.dry_run else "Nothing to remove.")
    elif total_freed > 0:
        print(f"\nTotal freed: {total_freed / 1e9:.1f} GB")
        if args.dry_run:
            print("Run without --dry-run to actually delete.")


if __name__ == "__main__":
    main()
