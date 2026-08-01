"""Evaluate the latest checkpoint from a finished ImageNet training run."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from inference_ours import _load_model, run_eval
from utils.dist_util import barrier, init_distributed, process_index
from utils.env import IMAGENET64_FID_NPZ, IMAGENET_FID_NPZ
from utils.misc import load_config


def _print0(*args, **kwargs) -> None:
    if process_index() == 0:
        print(*args, **kwargs)


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Checkpoint name must be state_<step>.pt, got: {path}") from exc


def _resolve_checkpoint(run_dir: Path | None, ckpt_arg: str) -> Path:
    if ckpt_arg:
        checkpoint = Path(ckpt_arg).expanduser().resolve()
    else:
        if run_dir is None:
            raise ValueError("Provide --run-dir or --ckpt")
        candidates = sorted(
            (run_dir / "checkpoints").glob("state_*.pt"),
            key=_checkpoint_step,
        )
        if not candidates:
            raise FileNotFoundError(f"No state_*.pt files under {run_dir / 'checkpoints'}")
        checkpoint = candidates[-1].resolve()

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    return checkpoint


def _parse_cfg_scales(raw: str, configured_scales) -> list[float]:
    if raw.strip():
        values = [float(value) for value in raw.replace(",", " ").split()]
    else:
        values = [float(value) for value in configured_scales]

    if not values:
        raise ValueError("At least one CFG scale is required")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"CFG scales must be finite and positive, got: {values}")
    return list(dict.fromkeys(values))


def _validate_reference(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing FID reference statistics: {path}\n"
            "Set IMAGENET64_FID_NPZ or IMAGENET_FID_NPZ to the matching file."
        )
    with np.load(path) as data:
        if not {"mu", "sigma"}.issubset(data.files):
            raise ValueError(f"FID reference must contain mu and sigma arrays: {path}")
        if data["mu"].ndim != 1 or data["sigma"].shape != (data["mu"].size, data["mu"].size):
            raise ValueError(f"Invalid FID reference array shapes in: {path}")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _is_complete_result(
    path: Path,
    *,
    checkpoint: Path,
    step: int,
    cfg_scale: float,
    num_samples: int,
    fid_ref: Path,
) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return (
            Path(payload["ckpt"]).resolve() == checkpoint
            and int(payload["step"]) == step
            and float(payload["cfg_scale"]) == cfg_scale
            and int(payload["num_samples"]) == num_samples
            and Path(payload["fid_ref"]).resolve() == fid_ref
            and math.isfinite(float(payload["fid"]))
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate class-balanced samples from a finished run's EMA checkpoint "
            "and compute FID-50K separately from training."
        )
    )
    parser.add_argument(
        "--run-dir",
        default="",
        help="Training workdir. The newest checkpoints/state_*.pt is selected by default.",
    )
    parser.add_argument("--ckpt", default="", help="Optional explicit state_*.pt override.")
    parser.add_argument("--config", required=True, help="Training config used by the checkpoint.")
    parser.add_argument(
        "--cfg-scales",
        default="",
        help="Comma- or space-separated scales; defaults to train.cfg_list in the config.",
    )
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--gen-bsz", type=int, default=64, help="Generation batch per GPU.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--fid-ref",
        default="",
        help="Reference .npz override; otherwise selected from the config resolution.",
    )
    parser.add_argument(
        "--result-dir",
        default="",
        help="Output directory; defaults to <run-dir>/fid.",
    )
    parser.add_argument("--keep-samples", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse valid JSON results for this exact checkpoint/reference/sample count.",
    )
    return parser


def main() -> None:
    init_distributed()
    args = build_parser().parse_args()

    if args.num_samples <= 0 or args.num_samples % 1000 != 0:
        raise ValueError("--num-samples must be positive and divisible by 1000 classes")
    if args.gen_bsz <= 0:
        raise ValueError("--gen-bsz must be positive")

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config: {config_path}")
    config = load_config(str(config_path))

    resolution = int(config.dataset.resolution)
    if resolution not in (64, 256):
        raise ValueError(f"Only ImageNet-64 and ImageNet-256 are supported, got {resolution}")

    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    checkpoint = _resolve_checkpoint(run_dir, args.ckpt)
    filename_step = _checkpoint_step(checkpoint)

    if args.result_dir:
        result_dir = Path(args.result_dir).expanduser().resolve()
    elif run_dir is not None:
        result_dir = run_dir / "fid"
    else:
        result_dir = checkpoint.parent.parent / "fid"

    default_ref = IMAGENET64_FID_NPZ if resolution == 64 else IMAGENET_FID_NPZ
    fid_ref = Path(args.fid_ref or default_ref).expanduser().resolve()
    _validate_reference(fid_ref)

    cfg_scales = _parse_cfg_scales(args.cfg_scales, config.train.cfg_list)
    if process_index() == 0:
        result_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    _print0(f"Config:       {config_path}")
    _print0(f"Resolution:   {resolution}")
    _print0(f"Checkpoint:   {checkpoint}")
    _print0(f"Reference:    {fid_ref}")
    _print0(f"CFG scales:   {cfg_scales}")
    _print0(f"Samples/CFG:  {args.num_samples}")
    _print0(f"Result dir:   {result_dir}")

    model, postprocess_fn, checkpoint_step, device = _load_model(
        str(checkpoint), str(config_path)
    )
    if checkpoint_step >= 0 and checkpoint_step != filename_step:
        raise ValueError(
            f"Checkpoint payload step {checkpoint_step} does not match filename step {filename_step}"
        )
    step = filename_step if checkpoint_step < 0 else checkpoint_step

    completed_results = []
    for cfg_scale in cfg_scales:
        cfg_text = f"{cfg_scale:g}"
        result_path = result_dir / (
            f"fid{args.num_samples // 1000}k_step{step:08d}_cfg{cfg_text}.json"
        )
        already_complete = args.skip_existing and _is_complete_result(
            result_path,
            checkpoint=checkpoint,
            step=step,
            cfg_scale=cfg_scale,
            num_samples=args.num_samples,
            fid_ref=fid_ref,
        )
        if already_complete:
            _print0(f"Skipping completed result: {result_path}")
            if process_index() == 0:
                completed_results.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue

        _print0(f"Evaluating CFG {cfg_text} ...")
        cfg_workdir = result_dir / "work" / f"cfg{cfg_text}"
        result = run_eval(
            model,
            postprocess_fn,
            str(checkpoint),
            step,
            str(cfg_workdir),
            num_samples=args.num_samples,
            cfg_scale=cfg_scale,
            gen_bsz=args.gen_bsz,
            fid_ref=str(fid_ref),
            seed=args.seed,
            keep_samples=args.keep_samples,
            device=device,
        )
        if result is not None:
            result.update(
                {
                    "config": str(config_path),
                    "resolution": resolution,
                    "num_samples": args.num_samples,
                    "gen_bsz_per_gpu": args.gen_bsz,
                    "seed": args.seed,
                    "fid_ref": str(fid_ref),
                }
            )
            _write_json(result_path, result)
            completed_results.append(result)
            _print0(f"Saved result: {result_path}")

    if process_index() == 0:
        finite_results = [item for item in completed_results if math.isfinite(float(item["fid"]))]
        best = min(finite_results, key=lambda item: float(item["fid"])) if finite_results else None
        summary = {
            "checkpoint": str(checkpoint),
            "step": step,
            "config": str(config_path),
            "resolution": resolution,
            "num_samples": args.num_samples,
            "fid_ref": str(fid_ref),
            "results": completed_results,
            "best": best,
        }
        summary_path = result_dir / f"fid{args.num_samples // 1000}k_step{step:08d}_summary.json"
        _write_json(summary_path, summary)
        print(f"Saved summary: {summary_path}")
        if best is not None:
            print(f"Best FID: {float(best['fid']):.6f} at CFG {float(best['cfg_scale']):g}")


if __name__ == "__main__":
    main()
