import sys
from pathlib import Path
from urllib.request import urlretrieve

import torch
from huggingface_hub import snapshot_download
from torch_fidelity.utils import create_feature_extractor

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from misc.download import (  # noqa: E402
    HF_ROOT,
    IMAGENET_FID_NPZ,
    TORCH_HUB_DIR,
    VAE_HF_PATH,
)


# The previous downloader fetched the complete W-Flow checkpoint repository and
# initialized the FID network in addition to downloading the training assets.
# It is intentionally disabled for the ImageNet64 Riesz run:
#
# snapshot_download(repo_id="stabilityai/sd-vae-ft-mse", local_dir=VAE_HF_PATH)
# snapshot_download(repo_id=WFLOW_HF_REPO_ID, local_dir=WFLOW_HF_ROOT)
# create_feature_extractor("inception-v3-compat", ["2048", "logits_unbiased"], ...)
# snapshot_download(repo_id=HF_REPO_ID, local_dir=HF_ROOT)


VAE_DIR = Path(VAE_HF_PATH)
MAE_DIR = Path(HF_ROOT)
FID_STATS_PATH = Path(IMAGENET_FID_NPZ)
TORCH_HUB_PATH = Path(TORCH_HUB_DIR)


VAE_REQUIRED = [
    VAE_DIR / "config.json",
    VAE_DIR / "diffusion_pytorch_model.safetensors",
]
if all(path.is_file() for path in VAE_REQUIRED):
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

MAE_REQUIRED = [
    MAE_DIR / "models/mae/jax/mae_latent_256/metadata.json",
    MAE_DIR / "models/mae/jax/mae_latent_256/ema_params.msgpack",
]
if all(path.is_file() for path in MAE_REQUIRED):
    print(f"Using existing mae_latent_256 at {MAE_DIR}.")
else:
    print(f"Downloading mae_latent_256 to {MAE_DIR} ...")
    snapshot_download(
        repo_id="Goodeat/drifting",
        local_dir=MAE_DIR,
        allow_patterns=["models/mae/jax/mae_latent_256/*"],
    )

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
TORCH_HUB_PATH.mkdir(parents=True, exist_ok=True)
torch.hub.set_dir(str(TORCH_HUB_PATH))
create_feature_extractor(
    "inception-v3-compat",
    ["2048", "logits_unbiased"],
    cuda=False,
)

required_files = [
    *VAE_REQUIRED,
    *MAE_REQUIRED,
    FID_STATS_PATH,
    TORCH_HUB_PATH / "checkpoints/weights-inception-2015-12-05-6726825d.pth",
]
missing = [path for path in required_files if not path.is_file()]
if missing:
    missing_text = "\n".join(f"  - {path}" for path in missing)
    raise FileNotFoundError(f"Download finished with missing files:\n{missing_text}")

print("SD-VAE, mae_latent_256, and ImageNet-256 FID assets are ready.")
