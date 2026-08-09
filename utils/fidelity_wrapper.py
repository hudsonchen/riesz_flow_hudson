"""
Torch-fidelity wrapper that supports .npz reference statistics.

The standard torch-fidelity API doesn't accept pre-computed .npz stats as
input2.  This module re-uses torch-fidelity internals (feature extractor,
FID/ISC computation) but loads the reference mu/sigma from a .npz file
instead of recomputing from raw images.

Reference: https://github.com/LTH14/torch-fidelity
"""

from __future__ import annotations

import multiprocessing
import os
import shutil
import tempfile

import numpy as np
import requests
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.env import TORCH_HUB_DIR

if os.path.isdir(TORCH_HUB_DIR) or not os.path.exists(TORCH_HUB_DIR):
    os.makedirs(TORCH_HUB_DIR, exist_ok=True)
    torch.hub.set_dir(TORCH_HUB_DIR)

from torch_fidelity.helpers import get_kwarg, vassert, vprint
from torch_fidelity.metric_fid import (
    fid_featuresdict_to_statistics_cached,
    fid_inputs_to_metric,
    fid_statistics_to_metric,
)
from torch_fidelity.metric_isc import isc_featuresdict_to_metric
from torch_fidelity.utils import (
    create_feature_extractor,
    extract_featuresdict_from_input_id_cached,
    get_cacheable_input_name,
    prepare_input_from_id,
)


def url_to_path(url_or_path: str) -> str:
    """Download a URL to a local temp file, or return a local path as-is."""
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        filename = url_or_path.split("/")[-1]
        cache_dir = os.path.join(tempfile.gettempdir(), "fid_ref_cache")
        os.makedirs(cache_dir, exist_ok=True)
        local_path = os.path.join(cache_dir, filename)
        if not os.path.exists(local_path):
            vprint(True, f"Downloading {url_or_path} to {local_path}...")
            response = requests.get(url_or_path, stream=True)
            with open(local_path, "wb") as f:
                shutil.copyfileobj(response.raw, f)
        return local_path
    return url_or_path


def _extract_featuresdict_xpu(input_id, feat_extractor, **kwargs):
    """Extract torch-fidelity features on Intel XPU from a dataset input."""
    dataset = prepare_input_from_id(input_id, **kwargs)
    if not isinstance(dataset, Dataset):
        raise TypeError("XPU fidelity extraction currently requires a dataset input")

    batch_size = min(get_kwarg("batch_size", kwargs), len(dataset))
    save_cpu_ram = get_kwarg("save_cpu_ram", kwargs)
    verbose = get_kwarg("verbose", kwargs)
    num_workers = 0 if save_cpu_ram else min(4, 2 * multiprocessing.cpu_count())
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    chunks = None
    with tqdm(
        disable=not verbose,
        leave=False,
        unit="samples",
        total=len(dataset),
        desc="Processing samples",
    ) as progress, torch.no_grad():
        for batch in dataloader:
            batch = batch.to("xpu")
            features = feat_extractor(batch)
            features = feat_extractor.convert_features_tuple_to_dict(features)
            features = {key: [value.cpu()] for key, value in features.items()}
            if chunks is None:
                chunks = features
            else:
                chunks = {key: chunks[key] + features[key] for key in chunks}
            progress.update(batch.shape[0])

    if chunks is None:
        raise RuntimeError("No samples were available for XPU feature extraction")
    return {key: torch.cat(values, dim=0) for key, values in chunks.items()}


def calculate_metrics(**kwargs) -> dict:
    """Compute FID / ISC using torch-fidelity internals.

    Accepts the same keyword arguments as ``torch_fidelity.calculate_metrics``
    with one extension: *input2* may be a path (or URL) to a ``.npz`` file
    containing pre-computed ``mu`` and ``sigma`` arrays.  In that case the
    reference statistics are loaded directly instead of being recomputed from
    images.
    """
    kwargs.update({
        "feature_extractor": "inception-v3-compat",
        "feature_layer_isc": "logits_unbiased",
        "feature_layer_fid": "2048",
    })

    verbose = get_kwarg("verbose", kwargs)
    input1, input2 = get_kwarg("input1", kwargs), get_kwarg("input2", kwargs)

    have_isc = get_kwarg("isc", kwargs)
    have_fid = get_kwarg("fid", kwargs)
    have_kid = get_kwarg("kid", kwargs)

    vassert(
        have_isc or have_fid or have_kid,
        "At least one of 'isc', 'fid', 'kid' metrics must be specified",
    )

    metrics = {}

    if not (have_isc or have_fid or have_kid):
        return metrics

    feature_extractor = get_kwarg("feature_extractor", kwargs)
    feature_layer_isc, feature_layer_fid, feature_layer_kid = (None,) * 3
    feature_layers = set()
    if have_isc:
        feature_layer_isc = get_kwarg("feature_layer_isc", kwargs)
        feature_layers.add(feature_layer_isc)
    if have_fid:
        feature_layer_fid = get_kwarg("feature_layer_fid", kwargs)
        feature_layers.add(feature_layer_fid)
    if have_kid:
        feature_layer_kid = get_kwarg("feature_layer_kid", kwargs)
        feature_layers.add(feature_layer_kid)

    use_xpu = bool(kwargs.pop("xpu", False))
    if use_xpu and not torch.xpu.is_available():
        raise RuntimeError("XPU fidelity extraction requested, but torch.xpu is unavailable")

    feat_extractor = create_feature_extractor(
        feature_extractor, list(feature_layers), **kwargs
    )
    if use_xpu:
        feat_extractor = feat_extractor.to("xpu")

    if (not use_xpu) and (not have_isc) and have_fid and (not have_kid):
        metric_fid = fid_inputs_to_metric(feat_extractor, **kwargs)
        metrics.update(metric_fid)
    else:
        vprint(verbose, "Extracting features from input1")
        if use_xpu:
            featuresdict_1 = _extract_featuresdict_xpu(1, feat_extractor, **kwargs)
        else:
            featuresdict_1 = extract_featuresdict_from_input_id_cached(
                1, feat_extractor, **kwargs
            )

        if have_isc:
            metric_isc = isc_featuresdict_to_metric(
                featuresdict_1, feature_layer_isc, **kwargs
            )
            metrics.update(metric_isc)

        if have_fid:
            cacheable_input1_name = get_cacheable_input_name(1, **kwargs)
            fid_stats_1 = fid_featuresdict_to_statistics_cached(
                featuresdict_1,
                cacheable_input1_name,
                feat_extractor,
                feature_layer_fid,
                **kwargs,
            )

            ref_path = url_to_path(input2)
            x = np.load(ref_path)
            # torch-fidelity 0.4 computes the feature mean in the extractor's
            # dtype (normally float32), while standard OpenAI reference batches
            # store both FID statistics as float64.  Its metric helper requires
            # exact dtype equality, so normalize both sides to the customary
            # double precision used for the covariance/FID calculation.
            fid_stats_1 = {
                "mu": np.asarray(fid_stats_1["mu"], dtype=np.float64),
                "sigma": np.asarray(fid_stats_1["sigma"], dtype=np.float64),
            }
            fid_stats_2 = {
                "mu": np.asarray(x["mu"], dtype=np.float64),
                "sigma": np.asarray(x["sigma"], dtype=np.float64),
            }
            metric_fid = fid_statistics_to_metric(
                fid_stats_1, fid_stats_2, get_kwarg("verbose", kwargs)
            )
            metrics.update(metric_fid)

    return metrics
