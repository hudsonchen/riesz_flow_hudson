from pathlib import Path

from huggingface_hub import snapshot_download


# The previous downloader fetched the complete W-Flow checkpoint repository and
# initialized the FID network in addition to downloading the training assets.
# It is intentionally disabled for the ImageNet64 Riesz run:
#
# snapshot_download(repo_id="stabilityai/sd-vae-ft-mse", local_dir=VAE_HF_PATH)
# snapshot_download(repo_id=WFLOW_HF_REPO_ID, local_dir=WFLOW_HF_ROOT)
# create_feature_extractor("inception-v3-compat", ["2048", "logits_unbiased"], ...)
# snapshot_download(repo_id=HF_REPO_ID, local_dir=HF_ROOT)


REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = REPO_ROOT / ".cache"
VAE_DIR = CACHE_ROOT / "sdvae_hf_root"
MAE_DIR = CACHE_ROOT / "drifting_hf_root"


print(f"Downloading SD-VAE to {VAE_DIR} ...")
snapshot_download(
    repo_id="stabilityai/sd-vae-ft-mse",
    local_dir=VAE_DIR,
    allow_patterns=[
        "config.json",
        "diffusion_pytorch_model.safetensors",
    ],
)

print(f"Downloading mae_latent_256 to {MAE_DIR} ...")
snapshot_download(
    repo_id="Goodeat/drifting",
    local_dir=MAE_DIR,
    allow_patterns=["models/mae/jax/mae_latent_256/*"],
)

required_files = [
    VAE_DIR / "config.json",
    VAE_DIR / "diffusion_pytorch_model.safetensors",
    MAE_DIR / "models/mae/jax/mae_latent_256/metadata.json",
    MAE_DIR / "models/mae/jax/mae_latent_256/ema_params.msgpack",
]
missing = [path for path in required_files if not path.is_file()]
if missing:
    missing_text = "\n".join(f"  - {path}" for path in missing)
    raise FileNotFoundError(f"Download finished with missing files:\n{missing_text}")

print("SD-VAE and mae_latent_256 are ready.")
