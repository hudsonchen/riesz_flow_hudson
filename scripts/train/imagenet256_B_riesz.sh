#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${WFLOW_REPO_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
cd "$REPO_DIR"

NGPU=${NGPU:-1}
MASTER_PORT=${MASTER_PORT:-6667}
CONFIG=${CONFIG:-configs/gen/imagenet256_B_riesz.yaml}
RUN_NAME=${RUN_NAME:-imagenet256_B_riesz}
WORKDIR=${WORKDIR:-runs/$RUN_NAME}
DRIFT_COMPILE=${DRIFT_COMPILE:-0}
DRIFT_FEAT_CHUNK=${DRIFT_FEAT_CHUNK:-1}

if [[ ! -f "$CONFIG" ]]; then
  echo "Error: config not found: $CONFIG" >&2
  exit 1
fi

echo "GPUs:       $NGPU"
echo "Config:     $CONFIG"
echo "Workdir:    $WORKDIR"
echo "Repository: $REPO_DIR"

DRIFT_COMPILE="$DRIFT_COMPILE" \
DRIFT_FEAT_CHUNK="$DRIFT_FEAT_CHUNK" \
NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
torchrun \
  --nnodes=1 \
  --nproc_per_node="$NGPU" \
  --master_addr=127.0.0.1 \
  --master_port="$MASTER_PORT" \
  train.py \
  --config "$CONFIG" \
  --workdir "$WORKDIR"
