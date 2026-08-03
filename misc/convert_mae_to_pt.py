#!/usr/bin/env python3
"""Convert a cached W-Flow MAE JAX MsgPack artifact to a PyTorch state dict."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.hf import load_mae_torch  # noqa: E402
from utils.env import HF_REPO_ID, HF_ROOT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert one cached MAE MsgPack checkpoint to ema_params.pt."
    )
    parser.add_argument("model", choices=("mae_latent_256", "mae_latent_640"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = Path(HF_ROOT).resolve() / "models/mae/jax" / args.model
    metadata_path = artifact_dir / "metadata.json"
    msgpack_path = artifact_dir / "ema_params.msgpack"
    output_path = artifact_dir / "ema_params.pt"

    if output_path.is_file():
        print(f"PyTorch MAE artifact already exists: {output_path}")
        return
    if not metadata_path.is_file() or not msgpack_path.is_file():
        raise FileNotFoundError(
            f"Expected metadata.json and ema_params.msgpack under {artifact_dir}"
        )

    started = time.perf_counter()
    print(
        f"Converting {msgpack_path} ({msgpack_path.stat().st_size / (1024**3):.2f} GiB) "
        f"to {output_path}",
        flush=True,
    )
    _, state_dict, _ = load_mae_torch(
        args.model,
        repo_id=HF_REPO_ID,
        output_root=HF_ROOT,
    )

    temporary_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    torch.save(
        {name: tensor.detach().cpu() for name, tensor in state_dict.items()},
        temporary_path,
    )
    os.replace(temporary_path, output_path)
    print(
        f"Wrote {output_path} ({output_path.stat().st_size / (1024**3):.2f} GiB) "
        f"in {time.perf_counter() - started:.1f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
