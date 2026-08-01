"""Latent cache dataset and cache builder for ImageNet release workflows.

Cache format (6 memory-mapped numpy files):
    cache_root/
        train_moments.npy       # (N, 32, 32, 4) float32
        train_moments_flip.npy  # (N, 32, 32, 4) float32
        train_targets.npy       # (N,) int64
        val_moments.npy
        val_moments_flip.npy
        val_targets.npy

Build:
    python -m dataset.latent --data-path /path/to/imagenet --target-path /path/to/cache

The dataset reads via np.load(mmap_mode='r') for zero-copy random access.
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader, is_image_file
from tqdm import tqdm

from utils.env import IMAGENET_CACHE_PATH, IMAGENET_PATH


# ---------------------------------------------------------------------------
# Dataset (reader)
# ---------------------------------------------------------------------------

class LatentDataset(Dataset):
    """Memory-mapped latent cache dataset.

    Accepts either:
      - ``root`` pointing to the cache directory + ``split`` name, OR
      - ``root`` pointing to ``{cache_root}/{split}`` (legacy-compatible call).

    The files must be named ``{split}_moments.npy``, etc.  When ``root``
    already contains files with the split prefix we use them directly;
    otherwise we look one level up.
    """

    def __init__(self, root: str, split: str | None = None):
        root = str(root)

        if split is not None:
            base, prefix = root, split
        else:
            base = os.path.dirname(root)
            prefix = os.path.basename(root)

        def _path(name: str) -> str:
            return os.path.join(base, f"{prefix}_{name}.npy")

        if not os.path.isfile(_path("moments")):
            base = root
            prefix = "train" if "train" in root else "val"

        self.moments: np.ndarray = np.load(_path("moments"), mmap_mode="r")
        self.moments_flip: np.ndarray = np.load(_path("moments_flip"), mmap_mode="r")
        self.targets: np.ndarray = np.load(_path("targets"), mmap_mode="r")
        assert len(self.moments) == len(self.targets)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        if torch.rand(1).item() < 0.5:
            m = np.array(self.moments[index])
        else:
            m = np.array(self.moments_flip[index])
        return m, int(self.targets[index])


# ---------------------------------------------------------------------------
# Cache builder helpers
# ---------------------------------------------------------------------------

def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def _center_crop_256(image: Image.Image) -> Image.Image:
    return center_crop_arr(image, 256)


class _FlatImageNetValidationDataset(Dataset):
    """Load a flat ILSVRC validation split using its XML annotations."""

    def __init__(
        self,
        image_dir: str,
        annotation_dir: str,
        class_to_idx: dict[str, int],
        transform,
    ) -> None:
        image_paths = sorted(
            path
            for path in Path(image_dir).iterdir()
            if path.is_file() and is_image_file(str(path))
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found in flat validation directory: {image_dir}")

        samples: list[tuple[str, int]] = []
        for image_path in image_paths:
            annotation_path = Path(annotation_dir) / f"{image_path.stem}.xml"
            if not annotation_path.is_file():
                raise FileNotFoundError(
                    f"Missing validation annotation for {image_path.name}: {annotation_path}"
                )

            try:
                annotation = ET.parse(annotation_path).getroot()
            except ET.ParseError as exc:
                raise ValueError(
                    f"Could not parse validation annotation {annotation_path}: {exc}"
                ) from exc

            wnids = {
                element.text.strip()
                for element in annotation.findall("object/name")
                if element.text and element.text.strip()
            }
            if len(wnids) != 1:
                raise ValueError(
                    f"Expected one class in {annotation_path}, found {sorted(wnids)}"
                )

            wnid = next(iter(wnids))
            if wnid not in class_to_idx:
                raise ValueError(
                    f"Validation class {wnid!r} from {annotation_path} is not present "
                    "in the training class folders"
                )
            samples.append((str(image_path), class_to_idx[wnid]))

        self.samples = samples
        self.imgs = samples
        self.targets = [target for _, target in samples]
        self.classes = sorted(class_to_idx, key=class_to_idx.get)
        self.class_to_idx = dict(class_to_idx)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = default_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _find_ilsvrc_val_annotation_dir(data_path: str) -> Path:
    """Find annotations paired with ``Data/CLS-LOC/val`` in ILSVRC."""
    data_root = Path(data_path)
    candidates = (
        data_root.parent.parent / "Annotations" / "CLS-LOC" / "val",
        data_root / "Annotations" / "CLS-LOC" / "val",
        data_root.parent / "Annotations" / "CLS-LOC" / "val",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "The validation images are flat, so their ILSVRC XML annotations are "
        f"required to recover class labels. Tried: {tried}"
    )


def _build_source_dataset(
    *,
    data_path: str,
    split: str,
    transform,
    train_class_to_idx: dict[str, int] | None,
):
    split_dir = os.path.join(data_path, split)
    has_class_dirs = any(path.is_dir() for path in Path(split_dir).iterdir())

    if split != "val" or has_class_dirs:
        dataset = datasets.ImageFolder(split_dir, transform=transform)
        if split == "val" and train_class_to_idx is not None:
            if dataset.class_to_idx != train_class_to_idx:
                raise ValueError(
                    "Validation class folders do not match the training class folders"
                )
        return dataset

    if train_class_to_idx is None:
        raise ValueError(
            "The training class folders are required to label a flat validation split"
        )

    annotation_dir = _find_ilsvrc_val_annotation_dir(data_path)
    print(f"[val] flat image directory; loading labels from {annotation_dir}")
    return _FlatImageNetValidationDataset(
        image_dir=split_dir,
        annotation_dir=str(annotation_dir),
        class_to_idx=train_class_to_idx,
        transform=transform,
    )


def _cache_split_is_complete(
    *,
    moments_path: str,
    flip_path: str,
    targets_path: str,
    expected_targets: list[int],
    latent_shape: tuple[int, int, int],
) -> tuple[bool, str]:
    """Check that an existing split cache has the expected arrays and labels."""
    paths = (moments_path, flip_path, targets_path)
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        return False, "missing " + ", ".join(os.path.basename(path) for path in missing)

    n_samples = len(expected_targets)
    try:
        moments = np.load(moments_path, mmap_mode="r")
        moments_flip = np.load(flip_path, mmap_mode="r")
        targets = np.load(targets_path, mmap_mode="r")
    except (OSError, ValueError) as exc:
        return False, f"could not load cache arrays: {exc}"

    expected_moments_shape = (n_samples, *latent_shape)
    if moments.shape != expected_moments_shape or moments.dtype != np.float32:
        return False, (
            f"{os.path.basename(moments_path)} has shape/dtype "
            f"{moments.shape}/{moments.dtype}, expected "
            f"{expected_moments_shape}/float32"
        )
    if moments_flip.shape != expected_moments_shape or moments_flip.dtype != np.float32:
        return False, (
            f"{os.path.basename(flip_path)} has shape/dtype "
            f"{moments_flip.shape}/{moments_flip.dtype}, expected "
            f"{expected_moments_shape}/float32"
        )
    if targets.shape != (n_samples,) or targets.dtype != np.int64:
        return False, (
            f"{os.path.basename(targets_path)} has shape/dtype "
            f"{targets.shape}/{targets.dtype}, expected {(n_samples,)}/int64"
        )

    # Comparing labels with ImageFolder catches a preallocated cache whose
    # encoding job stopped before all training batches were written.
    if not np.array_equal(targets, np.asarray(expected_targets, dtype=np.int64)):
        return False, f"{os.path.basename(targets_path)} does not match the source dataset"

    return True, "all three arrays match the source dataset"


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

def create_cached_dataset(
    local_batch_size: int,
    target_path: str,
    data_path: str,
    *,
    num_workers: int = 8,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    skip_existing_train: bool = False,
) -> None:
    """Encode ImageNet images into VAE latents and write memory-mapped .npy files.

    Runs on a single GPU.  For multi-GPU encoding, launch separate processes
    that each handle a different split or shard and concatenate afterwards.
    """
    from dataset.vae import vae_enc_decode

    from utils.dist_util import local_device
    device = local_device()
    encode_fn, _ = vae_enc_decode(replicate_params=False)

    Path(target_path).mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Lambda(_center_crop_256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    train_class_to_idx: dict[str, int] | None = None
    for split in ("train", "val"):
        split_dir = os.path.join(data_path, split)
        if not os.path.isdir(split_dir):
            print(f"Skipping {split}: {split_dir} does not exist")
            continue

        ds = _build_source_dataset(
            data_path=data_path,
            split=split,
            transform=transform,
            train_class_to_idx=train_class_to_idx,
        )
        if split == "train":
            train_class_to_idx = dict(ds.class_to_idx)
        n_samples = len(ds)
        print(f"[{split}] {n_samples} images")

        latent_shape = (32, 32, 4)

        moments_path = os.path.join(target_path, f"{split}_moments.npy")
        flip_path = os.path.join(target_path, f"{split}_moments_flip.npy")
        targets_path = os.path.join(target_path, f"{split}_targets.npy")

        if split == "train" and skip_existing_train:
            cache_complete, reason = _cache_split_is_complete(
                moments_path=moments_path,
                flip_path=flip_path,
                targets_path=targets_path,
                expected_targets=ds.targets,
                latent_shape=latent_shape,
            )
            if cache_complete:
                print(f"[train] existing cache is complete ({reason}); skipping train")
                continue
            print(f"[train] existing cache is not complete ({reason}); rebuilding train")

        mm_moments = np.lib.format.open_memmap(
            moments_path, mode="w+", dtype=np.float32,
            shape=(n_samples, *latent_shape),
        )
        mm_flip = np.lib.format.open_memmap(
            flip_path, mode="w+", dtype=np.float32,
            shape=(n_samples, *latent_shape),
        )
        mm_targets = np.lib.format.open_memmap(
            targets_path, mode="w+", dtype=np.int64,
            shape=(n_samples,),
        )

        loader = DataLoader(
            ds,
            batch_size=local_batch_size,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=(prefetch_factor if num_workers > 0 else None),
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=num_workers > 0,
        )

        write_idx = 0
        with tqdm(
            total=n_samples,
            desc=f"encode:{split}",
            unit="image",
            dynamic_ncols=True,
        ) as progress:
            for step, (images, labels) in enumerate(loader):
                images = images.to(device)

                with torch.no_grad():
                    # Use identical RNG state for both normal and flipped encode
                    # to match JAX's functional RNG semantics (same noise for both).
                    rng = torch.Generator(device=device)
                    rng.manual_seed(step)
                    latents = encode_fn(images, rng=rng).detach().cpu().numpy()

                    rng_flip = torch.Generator(device=device)
                    rng_flip.manual_seed(step)
                    latents_flip = encode_fn(
                        torch.flip(images, dims=(3,)), rng=rng_flip
                    ).detach().cpu().numpy()

                bs = latents.shape[0]
                end_idx = write_idx + bs
                mm_moments[write_idx:end_idx] = latents
                mm_flip[write_idx:end_idx] = latents_flip
                mm_targets[write_idx:end_idx] = labels.numpy().astype(np.int64)
                write_idx = end_idx
                progress.update(bs)

        mm_moments.flush()
        mm_flip.flush()
        mm_targets.flush()
        del mm_moments, mm_flip, mm_targets

        print(f"[{split}] wrote {write_idx} samples to {target_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ImageNet latent cache (memory-mapped .npy files)."
    )
    parser.add_argument(
        "--data-path", default=IMAGENET_PATH,
        help="ImageNet root containing train/ and val/.",
    )
    parser.add_argument(
        "--target-path", default=IMAGENET_CACHE_PATH or "latent_cache",
        help="Output directory for .npy cache files.",
    )
    parser.add_argument(
        "--local-batch-size", type=int, default=128,
        help="Encoding batch size.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=8,
        help="DataLoader worker count.",
    )
    parser.add_argument(
        "--prefetch-factor", type=int, default=2,
        help="DataLoader prefetch factor when num_workers > 0.",
    )
    parser.add_argument(
        "--pin-memory", action="store_true",
        help="Enable DataLoader pin_memory.",
    )
    parser.add_argument(
        "--skip-existing-train", action="store_true",
        help=(
            "Skip train encoding when all three train cache arrays have the "
            "expected shapes, dtypes, and source labels."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    create_cached_dataset(
        local_batch_size=args.local_batch_size,
        target_path=args.target_path,
        data_path=args.data_path,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        skip_existing_train=args.skip_existing_train,
    )


if __name__ == "__main__":
    main()
