#!/usr/bin/env python3
"""Build the ImageNet-256 SD-VAE latent cache on an interactive GPU node.

This is the direct-execution counterpart of
``scripts/train_slurm/imagenet256_latent_cache_slurm.sh``.  Allocate an
interactive node and activate the ``mmd_flow`` environment before running it;
the script does not call ``sbatch`` or ``srun``.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORAGE_ROOT = Path("/home/rc-chen1/rds/rds-airr-p109-tfgYl93jDnM")


def parse_args() -> argparse.Namespace:
    cache_root = Path(
        os.environ.get("WFLOW_CACHE_ROOT", DEFAULT_STORAGE_ROOT / "cache")
    ).expanduser()

    parser = argparse.ArgumentParser(
        description="Build the ImageNet-256 SD-VAE latent cache directly."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(
            os.environ.get(
                "IMAGENET_PATH", DEFAULT_STORAGE_ROOT / "ILSVRC/Data/CLS-LOC"
            )
        ).expanduser(),
        help="ImageNet root containing train/ and val/.",
    )
    parser.add_argument(
        "--target-path",
        type=Path,
        default=Path(
            os.environ.get(
                "IMAGENET_CACHE_PATH", cache_root / "imagenet256-latents-sdvae"
            )
        ).expanduser(),
        help="Output directory for the six memory-mapped NumPy cache files.",
    )
    parser.add_argument(
        "--vae-path",
        type=Path,
        default=Path(
            os.environ.get("WFLOW_VAE_HF_PATH", cache_root / "sdvae_hf_root")
        ).expanduser(),
        help="Local Hugging Face SD-VAE directory.",
    )
    parser.add_argument(
        "--local-batch-size",
        type=int,
        default=128,
        help="Single-device encoding batch size (default: 128).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="DataLoader worker count (default: 16).",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches prefetched by each DataLoader worker (default: 2).",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pinned DataLoader memory (default: enabled).",
    )
    parser.add_argument(
        "--skip-existing-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Validate and reuse a complete train cache (default: enabled); "
            "pass --no-skip-existing-train to rebuild it."
        ),
    )
    return parser.parse_args()


def require_path(path: Path, description: str, *, directory: bool) -> None:
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise SystemExit(f"Missing {description} {kind}: {path}")


def main() -> None:
    args = parse_args()

    if args.local_batch_size < 1:
        raise SystemExit("--local-batch-size must be at least 1")
    if args.num_workers < 0:
        raise SystemExit("--num-workers cannot be negative")
    if args.prefetch_factor < 1:
        raise SystemExit("--prefetch-factor must be at least 1")

    data_path = args.data_path.resolve()
    target_path = args.target_path.resolve()
    vae_path = args.vae_path.resolve()

    require_path(data_path / "train", "ImageNet train", directory=True)
    require_path(data_path / "val", "ImageNet validation", directory=True)
    require_path(vae_path / "config.json", "SD-VAE config", directory=False)
    require_path(
        vae_path / "diffusion_pytorch_model.safetensors",
        "SD-VAE weights",
        directory=False,
    )
    target_path.mkdir(parents=True, exist_ok=True)

    # Set paths and thread limits before importing torch or project modules.
    os.environ["WFLOW_CACHE_ROOT"] = str(target_path.parent)
    os.environ["WFLOW_VAE_HF_PATH"] = str(vae_path)
    os.environ["IMAGENET_PATH"] = str(data_path)
    os.environ["IMAGENET_CACHE_PATH"] = str(target_path)
    os.environ["PYTHONUNBUFFERED"] = "1"
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"

    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))

    from utils.dist_util import local_device, xpu_is_available

    device = local_device()
    if device.type == "cpu":
        raise SystemExit(
            "No GPU accelerator is available. Run this script inside an "
            "interactive GPU allocation."
        )
    if device.type == "xpu" and not xpu_is_available():
        raise SystemExit(
            "The configured Intel XPU is unavailable. Run this script on an "
            "interactive Dawn GPU node with the mmd_flow environment active."
        )

    from dataset.latent import create_cached_dataset

    print(f"Host:        {socket.gethostname()}")
    print(f"Start time:  {datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"Repository:  {REPO_ROOT}")
    print(f"Data root:   {data_path}")
    print(f"Target root: {target_path}")
    print(f"VAE root:    {vae_path}")
    print(f"Device:      {device}")
    print(f"Workers:     {args.num_workers}")

    create_cached_dataset(
        local_batch_size=args.local_batch_size,
        target_path=str(target_path),
        data_path=str(data_path),
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        skip_existing_train=args.skip_existing_train,
    )

    print(f"End time:    {datetime.now().astimezone().isoformat(timespec='seconds')}")
    print("Cache files:")
    for path in sorted(target_path.iterdir()):
        if path.is_file():
            print(f"  {path.name}: {path.stat().st_size / (1024 ** 3):.2f} GiB")


if __name__ == "__main__":
    main()
