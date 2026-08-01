from __future__ import annotations

import getpass
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_RC_STORAGE_ROOT = Path("/home/rc-chen1/rds/rds-airr-p109-tfgYl93jDnM")
_SAN_STORAGE_ROOT = Path("/SAN/intelsys/imagenet_mmd_flow")
_FALLBACK_CACHE_ROOT = REPO_ROOT / ".cache"
_FALLBACK_IMAGENET_PATH = Path("/home/zongchen/datasets/ILSVRC/Data/CLS-LOC")

_STORAGE_BY_USER = {
    "rc-chen1": {
        "cache": _RC_STORAGE_ROOT / "cache",
        "imagenet": _RC_STORAGE_ROOT / "ILSVRC/Data/CLS-LOC",
    },
    "zongchen": {
        "cache": _SAN_STORAGE_ROOT / "cache",
        "imagenet": _SAN_STORAGE_ROOT / "ILSVRC/Data/CLS-LOC",
    },
}

_USER_STORAGE = _STORAGE_BY_USER.get(getpass.getuser())
if _USER_STORAGE is not None and _USER_STORAGE["imagenet"].is_dir():
    _DEFAULT_CACHE_ROOT = _USER_STORAGE["cache"]
    _DEFAULT_IMAGENET_PATH = _USER_STORAGE["imagenet"]
else:
    _DEFAULT_CACHE_ROOT = _FALLBACK_CACHE_ROOT
    _DEFAULT_IMAGENET_PATH = _FALLBACK_IMAGENET_PATH

_CACHE_OVERRIDE = os.environ.get("WFLOW_CACHE_ROOT")
CACHE_ROOT = Path(_CACHE_OVERRIDE).expanduser() if _CACHE_OVERRIDE else _DEFAULT_CACHE_ROOT

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
IMAGENET64_FID_NPZ = os.environ.get(
    "IMAGENET64_FID_NPZ",
    str(Path(WFLOW_HF_ROOT) / "stats/VIRTUAL_imagenet64_labeled.npz"),
)

HF_REPO_ID = "Goodeat/drifting"

WFLOW_HF_REPO_ID = "jiaqihan99/W-Flow"

IMAGENET_PATH = os.environ.get("IMAGENET_PATH") or str(_DEFAULT_IMAGENET_PATH)
IMAGENET_PR_NPZ = os.environ.get("IMAGENET_PR_NPZ", "")
