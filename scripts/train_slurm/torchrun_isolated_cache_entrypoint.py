#!/usr/bin/env python3
"""Start one torchrun worker with rank-isolated compiler caches."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    job_id = _required_env("SLURM_JOB_ID")
    local_rank = _required_env("LOCAL_RANK")
    node_id = _required_env("SLURM_NODEID")

    temporary_root = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp"
    cache_root = Path(
        os.environ.get(
            "WFLOW_COMPILE_CACHE_ROOT",
            str(Path(temporary_root) / f"wflow-compile-{job_id}"),
        )
    )
    rank_root = cache_root / f"node-{node_id}" / f"rank-{local_rank}"
    triton_cache = rank_root / "triton"
    inductor_cache = rank_root / "inductor"
    triton_cache.mkdir(parents=True, exist_ok=True)
    inductor_cache.mkdir(parents=True, exist_ok=True)

    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)

    print(
        f"Compiler caches for local rank {local_rank}: "
        f"TRITON_CACHE_DIR={triton_cache}, "
        f"TORCHINDUCTOR_CACHE_DIR={inductor_cache}",
        flush=True,
    )

    repository = Path(os.environ.get("REPO_DIR", Path(__file__).resolve().parents[2]))
    train_script = repository / "train.py"
    if not train_script.is_file():
        raise FileNotFoundError(f"Missing training entrypoint: {train_script}")

    os.execv(
        sys.executable,
        [sys.executable, "-u", str(train_script), *sys.argv[1:]],
    )


if __name__ == "__main__":
    main()
