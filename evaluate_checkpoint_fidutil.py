from pathlib import Path

import numpy as np
import torch
from PIL import Image

from inference_ours import _load_model, generate_and_save
from utils.env import IMAGENET_FID_NPZ
from utils.fid_util import (
    _compute_frechet_distance,
    _extract_inception_features,
    _load_ref_stats,
)
from utils.dist_util import init_distributed


def main():
    init_distributed()

    config_path = "configs/gen/imagenet64_riesz.yaml"
    ckpt_path = (
        "results_cs_cluster/"
        "imagenet64_riesz_eps1e-6_lr2e-4_trainbs8_pos16_neg8_gen16_acc2/"
        "checkpoints/state_00125100.pt"
    )

    workdir = Path("runs/fid_step125100_fidutil")
    image_dir = workdir / "fid_outputs"
    workdir.mkdir(parents=True, exist_ok=True)

    model, postprocess_fn, step, device = _load_model(
        ckpt_path,
        config_path,
    )

    print("Generating 50,000 images...")
    generate_and_save(
        model,
        postprocess_fn,
        str(image_dir),
        num_samples=50000,
        device_batch_size=64,
        cfg_scale=2.5,
        seed=0,
        device=device,
    )

    paths = sorted(image_dir.glob("*.png"))
    if len(paths) != 50000:
        raise RuntimeError(f"Expected 50000 images, found {len(paths)}")

    print("Loading generated images...")
    images = np.stack(
        [
            np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            for path in paths
        ],
        axis=0,
    )

    print("Extracting Inception features using utils/fid_util.py...")
    features, logits = _extract_inception_features(
        images,
        compute_logits=True,
        batch_size=200,
    )

    mu_fake = np.mean(features, axis=0)
    sigma_fake = np.cov(features, rowvar=False)

    ref = _load_ref_stats("imagenet256")

    fid = _compute_frechet_distance(
        ref["mu"],
        ref["sigma"],
        mu_fake,
        sigma_fake,
    )

    print()
    print("Checkpoint:", ckpt_path)
    print("Step:", step)
    print("Reference:", IMAGENET_FID_NPZ)
    print("FID using utils/fid_util.py:", fid)


if __name__ == "__main__":
    main()
