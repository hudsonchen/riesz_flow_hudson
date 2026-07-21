#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../accelerator_count.sh"
NGPU=${NGPU:-$(accelerator_count)}

MASTER_PORT=${MASTER_PORT:-6667}
CONFIG=configs/gen/imagenet64_pilot.yaml
WORKDIR=${WORKDIR:-runs/imagenet64_pilot}

# Compilation of the full multi-scale MAE/Sinkhorn graph can be slow and has
# triggered illegal CUDA accesses on some consumer GPUs. Enable it explicitly
# with DRIFT_COMPILE=1 after validating the eager run.
DRIFT_COMPILE=${DRIFT_COMPILE:-0} \
DRIFT_FEAT_CHUNK=${DRIFT_FEAT_CHUNK:-1} \
NCCL_DEBUG=WARN \
torchrun \
    --nproc_per_node="$NGPU" \
    --master_port="$MASTER_PORT" \
    train.py \
    --config "$CONFIG" \
    --workdir "$WORKDIR"
