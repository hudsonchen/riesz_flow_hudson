import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.env import (  # noqa: E402
    HF_REPO_ID,
    HF_ROOT,
    IMAGENET_FID_NPZ,
    TORCH_HUB_DIR,
    VAE_HF_PATH,
)

VAE_DIR = Path(VAE_HF_PATH)
MAE_DIR = Path(HF_ROOT)
FID_STATS_PATH = Path(IMAGENET_FID_NPZ)
TORCH_HUB_PATH = Path(TORCH_HUB_DIR)

MAE_MODELS = ("mae_latent_256", "mae_latent_640")


def required_mae_files(model_name: str) -> list[Path]:
    model_dir = MAE_DIR / "models/mae/jax" / model_name
    return [model_dir / "metadata.json", model_dir / "ema_params.msgpack"]


def ensure_mae(model_name: str) -> list[Path]:
    required = required_mae_files(model_name)
    if all(path.is_file() for path in required):
        print(f"Using existing {model_name} at {required[0].parent}.")
        return required

    print(f"Downloading {model_name} to {required[0].parent} ...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=MAE_DIR,
        allow_patterns=[f"models/mae/jax/{model_name}/*"],
    )
    return required


def ensure_default_assets() -> list[Path]:
    vae_required = [
        VAE_DIR / "config.json",
        VAE_DIR / "diffusion_pytorch_model.safetensors",
    ]
    if all(path.is_file() for path in vae_required):
        print(f"Using existing SD-VAE at {VAE_DIR}.")
    else:
        print(f"Downloading SD-VAE to {VAE_DIR} ...")
        snapshot_download(
            repo_id="stabilityai/sd-vae-ft-mse",
            local_dir=VAE_DIR,
            allow_patterns=[
                "config.json",
                "diffusion_pytorch_model.safetensors",
            ],
        )

    mae_required = []
    for model_name in MAE_MODELS:
        mae_required.extend(ensure_mae(model_name))

    if FID_STATS_PATH.is_file():
        print(f"Using existing ImageNet-256 FID statistics at {FID_STATS_PATH}.")
    else:
        print(f"Downloading ImageNet-256 FID statistics to {FID_STATS_PATH} ...")
        FID_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(
            "https://raw.githubusercontent.com/LTH14/JiT/main/fid_stats/jit_in256_stats.npz",
            FID_STATS_PATH,
        )

    print(f"Downloading torch-fidelity Inception network to {TORCH_HUB_PATH} ...")
    import torch
    from torch_fidelity.utils import create_feature_extractor

    TORCH_HUB_PATH.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(TORCH_HUB_PATH))
    create_feature_extractor(
        "inception-v3-compat",
        ["2048", "logits_unbiased"],
        cuda=False,
    )

    return [
        *vae_required,
        *mae_required,
        FID_STATS_PATH,
        TORCH_HUB_PATH / "checkpoints/weights-inception-2015-12-05-6726825d.pth",
    ]


def verify_files(required_files: list[Path]) -> None:
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        missing_text = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Download finished with missing files:\n{missing_text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download W-Flow prerequisite assets.")
    parser.add_argument(
        "--mae-only",
        choices=MAE_MODELS,
        metavar="MODEL",
        help="download and verify only the selected MAE feature extractor",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mae_only:
        required_files = ensure_mae(args.mae_only)
        verify_files(required_files)
        print(f"{args.mae_only} is ready at {required_files[0].parent}.")
        return

    required_files = ensure_default_assets()
    verify_files(required_files)
    print("SD-VAE, MAE feature extractors, and ImageNet-256 FID assets are ready.")


if __name__ == "__main__":
    main()
