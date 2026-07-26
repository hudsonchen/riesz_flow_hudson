from __future__ import annotations

import getpass
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_USER_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "w-flow"
_CACHE_BY_USER = {
    "rc-chen1": Path("/home/rc-chen1/rds/rds-airr-p109-tfgYl93jDnM/cache"),
    "zongchen": REPO_ROOT / ".cache",
}
CACHE_ROOT = Path(
    os.environ.get("WFLOW_CACHE_ROOT", _CACHE_BY_USER.get(getpass.getuser(), _USER_CACHE))
).expanduser()

HF_ROOT = os.environ.get("WFLOW_DRIFTING_HF_ROOT", str(CACHE_ROOT / "drifting_hf_root"))
VAE_HF_PATH = os.environ.get("WFLOW_VAE_HF_PATH", str(CACHE_ROOT / "sdvae_hf_root"))
TORCH_HUB_DIR = os.environ.get("TORCH_HUB_DIR", str(CACHE_ROOT / "torch_hub"))
WFLOW_HF_ROOT = os.environ.get("WFLOW_HF_ROOT", str(CACHE_ROOT / "wflow_hf_root"))
IMAGENET_CACHE_PATH = os.environ.get(
    "IMAGENET_CACHE_PATH", str(CACHE_ROOT / "imagenet256-latents-sdvae")
)
IMAGENET_FID_NPZ = os.environ.get(
    "IMAGENET_FID_NPZ", str(Path(WFLOW_HF_ROOT) / "stats/jit_in256_stats.npz")
)

_SAN_IMAGENET_PATH = Path("/SAN/intelsys/imagenet_mmd_flow/ILSVRC/Data/CLS-LOC")
_DEFAULT_IMAGENET_PATH = (
    _SAN_IMAGENET_PATH
    if _SAN_IMAGENET_PATH.is_dir()
    else Path.home() / "datasets/ILSVRC/Data/CLS-LOC"
)

HF_REPO_ID = "Goodeat/drifting"

WFLOW_HF_REPO_ID = "jiaqihan99/W-Flow"

IMAGENET_PATH = os.environ.get("IMAGENET_PATH", str(_DEFAULT_IMAGENET_PATH))
IMAGENET_PR_NPZ = os.environ.get("IMAGENET_PR_NPZ", "")
