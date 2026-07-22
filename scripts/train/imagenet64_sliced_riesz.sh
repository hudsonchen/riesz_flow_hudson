#!/usr/bin/env bash
set -euo pipefail

NGPU=${NGPU:-1}

MASTER_PORT=${MASTER_PORT:-6668}
CONFIG=configs/gen/imagenet64_sliced_riesz.yaml
WORKDIR=${WORKDIR:-runs/imagenet64_sliced_riesz_pilot_direct}

DRIFT_COMPILE=${DRIFT_COMPILE:-0} \
DRIFT_FEAT_CHUNK=${DRIFT_FEAT_CHUNK:-1} \
NCCL_DEBUG=WARN \
torchrun \
    --nproc_per_node="$NGPU" \
    --master_port="$MASTER_PORT" \
    train.py \
    --config "$CONFIG" \
    --workdir "$WORKDIR"
