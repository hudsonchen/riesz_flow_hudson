#!/usr/bin/env bash
set -euo pipefail

NGPU=${NGPU:-1}

MASTER_PORT=${MASTER_PORT:-6667}
CONFIG=configs/gen/imagenet64_pilot.yaml
RUN_NAME=${RUN_NAME:-imagenet64_mmd_ot_debiased_r0p05_sink10_lr2e-4_trainbs8_pos16_neg8_gen16_acc2}
WORKDIR=${WORKDIR:-runs/$RUN_NAME}

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
