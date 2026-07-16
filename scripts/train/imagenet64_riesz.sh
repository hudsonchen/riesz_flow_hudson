#!/usr/bin/env bash
set -euo pipefail

NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
if [ "$NGPU" -lt 1 ]; then
    NGPU=1
fi

MASTER_PORT=${MASTER_PORT:-6668}
CONFIG=configs/gen/imagenet64_riesz.yaml
WORKDIR=${WORKDIR:-runs/imagenet64_riesz_direct}

DRIFT_COMPILE=${DRIFT_COMPILE:-0} \
DRIFT_FEAT_CHUNK=${DRIFT_FEAT_CHUNK:-1} \
NCCL_DEBUG=WARN \
torchrun \
    --nproc_per_node="$NGPU" \
    --master_port="$MASTER_PORT" \
    train.py \
    --config "$CONFIG" \
    --workdir "$WORKDIR"
