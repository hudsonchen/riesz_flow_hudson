from __future__ import annotations

import os
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_CACHE = _PROJECT_ROOT / ".cache"
_USER_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "w-flow"
_DEFAULT_CACHE = _PROJECT_CACHE if os.access(_PROJECT_ROOT, os.W_OK) else _USER_CACHE
_CACHE_ROOT = Path(os.environ.get("WFLOW_CACHE_ROOT", _DEFAULT_CACHE)).expanduser()
_SAN_IMAGENET_PATH = Path("/SAN/intelsys/imagenet_mmd_flow/ILSVRC/Data/CLS-LOC")
_DEFAULT_IMAGENET_PATH = (
    _SAN_IMAGENET_PATH
    if _SAN_IMAGENET_PATH.is_dir()
    else Path.home() / "datasets/ILSVRC/Data/CLS-LOC"
)

HF_REPO_ID = "Goodeat/drifting"
HF_ROOT = os.environ.get("WFLOW_DRIFTING_HF_ROOT", str(_CACHE_ROOT / "drifting_hf_root"))
VAE_HF_PATH = os.environ.get("WFLOW_VAE_HF_PATH", str(_CACHE_ROOT / "sdvae_hf_root"))
TORCH_HUB_DIR = os.environ.get("TORCH_HUB_DIR", str(_CACHE_ROOT / "torch_hub"))

WFLOW_HF_REPO_ID = "jiaqihan99/W-Flow"
WFLOW_HF_ROOT = os.environ.get("WFLOW_HF_ROOT", str(_CACHE_ROOT / "wflow_hf_root"))

IMAGENET_PATH = os.environ.get("IMAGENET_PATH", str(_DEFAULT_IMAGENET_PATH))
IMAGENET_CACHE_PATH = os.environ.get(
    "IMAGENET_CACHE_PATH", str(_CACHE_ROOT / "imagenet256-latents-sdvae")
)
IMAGENET_FID_NPZ = os.environ.get(
    "IMAGENET_FID_NPZ", str(Path(WFLOW_HF_ROOT) / "stats/jit_in256_stats.npz")
)
IMAGENET_PR_NPZ = os.environ.get("IMAGENET_PR_NPZ", "")
