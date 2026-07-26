from __future__ import annotations

import os
from pathlib import Path

from misc.download import (
    HF_ROOT,
    IMAGENET_CACHE_PATH,
    IMAGENET_FID_NPZ,
    TORCH_HUB_DIR,
    VAE_HF_PATH,
    WFLOW_HF_ROOT,
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
