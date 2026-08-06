from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from absl import logging as absl_logging
from PIL import Image

from utils.dist_util import accelerator_synchronize, process_index

# Abseil defaults to warning-only output when this project is launched through
# argparse rather than absl.app.  Enable INFO so startup/checkpoint timing from
# log_for_0() is not silently discarded.
absl_logging.set_verbosity(absl_logging.INFO)


def is_rank_zero() -> bool:
    return process_index() == 0


def log_for_0(msg, *args, **kwargs):
    if is_rank_zero():
        absl_logging.info(msg, *args, **kwargs)


def log_for_all(msg):
    absl_logging.info("[Rank %s] %s", process_index(), msg)


class TrainingTimeTracker:
    """Persist synchronized training-loop wall time across resumed sessions."""

    def __init__(self, workdir: str, initial_step: int, total_steps: int) -> None:
        self.path = Path(workdir).resolve() / "log" / "training_time.json"
        self.initial_step = int(initial_step)
        self.total_steps = int(total_steps)
        self.session_started_at_utc = datetime.now(timezone.utc).isoformat()
        self._started_at: Optional[float] = None
        self._accumulated_before_session = self._load_accumulated_seconds()

    def _load_accumulated_seconds(self) -> float:
        if not is_rank_zero() or self.initial_step <= 0 or not self.path.exists():
            return 0.0
        try:
            previous = json.loads(self.path.read_text(encoding="utf-8"))
            previous_step = int(previous.get("completed_steps", -1))
            previous_seconds = float(previous.get("accumulated_training_seconds", 0.0))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            log_for_0("Could not read prior training time from %s: %s", self.path, exc)
            return 0.0
        if 0 <= previous_step <= self.initial_step and previous_seconds >= 0.0:
            return previous_seconds
        log_for_0(
            "Ignoring training time in %s because its completed_steps=%d is incompatible with resume step=%d",
            self.path,
            previous_step,
            self.initial_step,
        )
        return 0.0

    def start(self) -> None:
        accelerator_synchronize()
        self._started_at = time.perf_counter()

    def save(self, completed_steps: int, status: str) -> Optional[Dict[str, Any]]:
        if self._started_at is None:
            raise RuntimeError("TrainingTimeTracker.start() must be called before save()")
        if status not in {"running", "complete"}:
            raise ValueError(f"Unsupported training-time status: {status}")

        accelerator_synchronize()
        session_seconds = time.perf_counter() - self._started_at
        accumulated_seconds = self._accumulated_before_session + session_seconds
        if not is_rank_zero():
            return None

        payload: Dict[str, Any] = {
            "schema_version": 1,
            "scope": "training_loop_including_checkpoint_and_evaluation",
            "status": status,
            "session_initial_step": self.initial_step,
            "completed_steps": int(completed_steps),
            "target_steps": self.total_steps,
            "session_training_seconds": session_seconds,
            "accumulated_training_seconds": accumulated_seconds,
            "accumulated_training_hours": accumulated_seconds / 3600.0,
            "session_started_at_utc": self.session_started_at_utc,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, self.path)
        return payload


class WandbLogger:
    def __init__(self) -> None:
        self.step = 0
        self.use_wandb = True
        self.log_every_k = 1
        self._buffer: Dict[str, float] = {}
        self._count: Dict[str, int] = {}
        self.offline_dir = Path("log")
        self._wandb = None

    def set_logging(
        self,
        project: Optional[str] = None,
        config: Optional[Any] = None,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        use_wandb: bool = True,
        offline_dir: str = "log",
        workdir: Optional[str] = None,
        log_every_k: int = 1,
        allow_resume: bool = True,
        **kwargs,
    ) -> None:
        self.use_wandb = bool(use_wandb)
        self.log_every_k = int(log_every_k)
        workdir_path = Path(workdir).resolve() if workdir else None
        resolved_offline_dir = workdir_path / "log" if (workdir_path is not None and not self.use_wandb) else Path(offline_dir)
        self.offline_dir = resolved_offline_dir
        self.offline_dir.mkdir(parents=True, exist_ok=True)

        if not is_rank_zero():
            return

        if self.use_wandb:
            import wandb

            self._wandb = wandb
            default_run_id = ""
            if workdir_path is not None:
                default_run_id = hashlib.sha1(str(workdir_path).encode("utf-8")).hexdigest()[:16]
            run_id = kwargs.pop("run_id", None) or default_run_id
            allow_resume = allow_resume and os.environ.get("WANDB_RESUME", "").lower() != "never"
            init_kwargs = dict(project=project, entity=entity, name=name, config=config, reinit=True)
            if allow_resume:
                init_kwargs["resume"] = "allow"
                if run_id:
                    init_kwargs["id"] = run_id
            init_kwargs.update(kwargs)
            wandb.init(**init_kwargs)

    def set_step(self, step: int) -> None:
        self.step = int(step)

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        reduced = {k: (self._buffer[k] / max(1, self._count.get(k, 1))) for k in self._buffer.keys()}
        if self._wandb is not None:
            self._wandb.log(reduced, step=self.step)
        else:
            p = self.offline_dir / "metrics.jsonl"
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": self.step, **reduced}, ensure_ascii=False) + "\n")
        self._buffer.clear()
        self._count.clear()

    def log_dict(self, d: Dict[str, Any]) -> None:
        if not is_rank_zero():
            return
        reduced = {}
        for k, v in d.items():
            if torch.is_tensor(v):
                v = float(v.detach().float().mean().item())
            elif isinstance(v, np.ndarray):
                v = float(np.asarray(v).mean())
            if isinstance(v, (int, float, np.floating, np.integer)):
                reduced[k] = float(v)
        for k, v in reduced.items():
            self._buffer[k] = self._buffer.get(k, 0.0) + float(v)
            self._count[k] = self._count.get(k, 0) + 1
        if self.log_every_k <= 1 or (self.step % self.log_every_k == 0):
            self._flush_buffer()

    def log_dict_dir(self, prefix: str, d: Dict[str, Any]) -> None:
        self.log_dict({f"{prefix}/{k}": v for k, v in d.items()})

    @staticmethod
    def _normalize_images(images) -> np.ndarray:
        if torch.is_tensor(images):
            arr = images.detach().cpu().numpy()
        else:
            arr = np.asarray(images)
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.ndim != 4:
            raise ValueError(f"Expected image batch with 3 or 4 dims, got shape {arr.shape}")
        if arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (0, 2, 3, 1))
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.shape[-1] != 3:
            raise ValueError(f"Expected channel-last image batch with 3 channels, got shape {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        return arr

    @staticmethod
    def _make_grid_image(images: np.ndarray, rows: int = 8) -> Image.Image:
        rows = max(1, int(rows))
        pil_imgs = [Image.fromarray(img) for img in images]
        cols = max(1, int(math.ceil(len(pil_imgs) / rows)))
        w, h = pil_imgs[0].size
        total = rows * cols
        if len(pil_imgs) < total:
            blank = Image.new("RGB", (w, h), color=(0, 0, 0))
            pil_imgs += [blank] * (total - len(pil_imgs))
        grid = Image.new("RGB", (cols * w, rows * h))
        for idx, img in enumerate(pil_imgs):
            row = idx % rows
            col = idx // rows
            grid.paste(img, (col * w, row * h))
        return grid

    def log_image(self, name: str, images) -> None:
        if not is_rank_zero():
            return
        arr = self._normalize_images(images)
        grid_img = self._make_grid_image(arr)
        if self._wandb is not None:
            self._wandb.log({name: [self._wandb.Image(img) for img in arr]}, step=self.step)
            self._wandb.log({f"{name}_grid": self._wandb.Image(grid_img)}, step=self.step)
            return
        out = self.offline_dir / "images"
        out.mkdir(parents=True, exist_ok=True)
        grid_img.save(out / f"{name.replace('/', '_')}_step{self.step}.jpg", format="JPEG")

    def finish(self) -> None:
        self._flush_buffer()
        if self._wandb is not None and is_rank_zero():
            self._wandb.finish()


class NullLogger:
    @staticmethod
    def log_dict(*args, **kwargs):
        return None

    @staticmethod
    def log_image(*args, **kwargs):
        return None

    @staticmethod
    def finish(*args, **kwargs):
        return None
