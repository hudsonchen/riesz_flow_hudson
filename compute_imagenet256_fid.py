"""Compute ImageNet-256 FID for a pretrained W-Flow checkpoint on Dawn.

Edit the constants in the ``DAWN SETTINGS`` block when evaluating a different
checkpoint. No command-line arguments are required by this file.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from inference_ours import _load_model, run_eval
from utils.dist_util import (
    barrier,
    cleanup_distributed,
    init_distributed,
    process_index,
)
from utils.env import TORCH_HUB_DIR
from utils.misc import load_config


# ---------------------------------------------------------------------------
# DAWN SETTINGS -- edit these paths/values directly when needed.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent
DAWN_RDS_ROOT = Path("/home/rc-chen1/rds/rds-airr-p109-tfgYl93jDnM")
CHECKPOINT_PATH = (
    DAWN_RDS_ROOT
    / "runs"
    / "imagenet256_B_riesz_4nodes/checkpoints"
    / "state_00004000.pt"
)
# This config has the released B/2 inference architecture and ImageNet-256
# latent/VAE settings. Training-loss fields are not used during evaluation.
CONFIG_PATH = REPO_ROOT / "configs/gen/imagenet256_B_sinkhorn.yaml"
FID_REFERENCE_PATH = (
    DAWN_RDS_ROOT / "cache/wflow_hf_root/stats/jit_in256_stats.npz"
)
CFG_SCALE = 1.19
NUM_SAMPLES = 50_000
GENERATION_BATCH_PER_XPU = 64
SEED = 0
KEEP_SAMPLES = False
WORKDIR = (
    Path(os.environ.get("SLURM_TMPDIR", "/tmp"))
    / "imagenet256_pretrained_fid"
)
RESULT_JSON = (
    DAWN_RDS_ROOT
    / "results/imagenet256_pretrained_fid/fid50000_cfg1.19.json"
)


def _print0(*args, **kwargs) -> None:
    if process_index() == 0:
        print(*args, **kwargs)


def _validate_inputs(
    checkpoint: Path,
    config_path: Path,
    fid_ref: Path,
    num_samples: int,
    gen_bsz: int,
) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing pretrained checkpoint: {checkpoint}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing model config: {config_path}")
    if not fid_ref.is_file():
        raise FileNotFoundError(
            f"Missing ImageNet-256 FID statistics: {fid_ref}\n"
            "Edit FID_REFERENCE_PATH in the DAWN SETTINGS block."
        )
    if num_samples <= 0 or num_samples % 1000 != 0:
        raise ValueError("NUM_SAMPLES must be positive and divisible by 1000 classes")
    if gen_bsz <= 0:
        raise ValueError("GENERATION_BATCH_PER_XPU must be positive")

    config = load_config(str(config_path))
    resolution = int(config.dataset.resolution)
    num_classes = int(config.dataset.get("num_classes", 1000))
    if resolution != 256 or num_classes != 1000:
        raise ValueError(
            "This evaluator requires an ImageNet-256 config with 1000 classes; "
            f"got resolution={resolution}, num_classes={num_classes}"
        )

    with np.load(fid_ref) as stats:
        if not {"mu", "sigma"}.issubset(stats.files):
            raise ValueError(f"FID reference must contain mu and sigma: {fid_ref}")
        mu = stats["mu"]
        sigma = stats["sigma"]
        if mu.shape != (2048,) or sigma.shape != (2048, 2048):
            raise ValueError(
                "Expected 2048-dimensional ImageNet-256 Inception statistics, "
                f"got mu={mu.shape}, sigma={sigma.shape}"
            )

    inception_dir = Path(TORCH_HUB_DIR) / "checkpoints"
    inception_weights = list(
        inception_dir.glob("weights-inception-2015-12-05-*.pth")
    )
    if not inception_weights:
        raise FileNotFoundError(
            f"No cached torch-fidelity Inception weights found in {inception_dir}"
        )


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    if not math.isfinite(CFG_SCALE) or CFG_SCALE <= 0:
        raise ValueError("CFG_SCALE must be finite and positive")

    checkpoint = CHECKPOINT_PATH.expanduser().resolve()
    config_path = CONFIG_PATH.expanduser().resolve()
    fid_ref = FID_REFERENCE_PATH.expanduser().resolve()
    workdir = WORKDIR.expanduser().resolve()
    json_out = RESULT_JSON.expanduser().resolve()

    init_distributed()
    try:
        _validate_inputs(
            checkpoint,
            config_path,
            fid_ref,
            NUM_SAMPLES,
            GENERATION_BATCH_PER_XPU,
        )
        if process_index() == 0:
            workdir.mkdir(parents=True, exist_ok=True)
        barrier()

        _print0(f"Checkpoint:       {checkpoint}")
        _print0(f"Config:           {config_path}")
        _print0(f"FID reference:    {fid_ref}")
        _print0(f"CFG scale:        {CFG_SCALE:g}")
        _print0(f"Number of samples: {NUM_SAMPLES}")
        _print0(f"Batch per XPU:    {GENERATION_BATCH_PER_XPU}")

        model, postprocess_fn, checkpoint_step, device = _load_model(
            str(checkpoint), str(config_path)
        )
        result = run_eval(
            model,
            postprocess_fn,
            str(checkpoint),
            checkpoint_step,
            str(workdir),
            num_samples=NUM_SAMPLES,
            cfg_scale=CFG_SCALE,
            gen_bsz=GENERATION_BATCH_PER_XPU,
            fid_ref=str(fid_ref),
            seed=SEED,
            keep_samples=KEEP_SAMPLES,
            device=device,
        )

        if result is not None:
            result.update(
                {
                    "config": str(config_path),
                    "fid_ref": str(fid_ref),
                    "num_samples": NUM_SAMPLES,
                    "gen_bsz_per_xpu": GENERATION_BATCH_PER_XPU,
                    "seed": SEED,
                }
            )
            _write_json_atomic(json_out, result)
            print(f"\nFID: {float(result['fid']):.6f}")
            print(f"Saved result: {json_out}")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
