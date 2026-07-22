#!/usr/bin/env bash
#$ -N imagenet64_riesz
#$ -cwd
#$ -V
#$ -l gpu=true,gpu_type=h100
#$ -pe gpu 8
#$ -l tmem=10G
#$ -l h_rt=1:00:00
#$ -R y
#$ -j y

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${W_FLOW_REPO_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}
CONDA_ENV=${CONDA_ENV:-/home/zongchen/miniconda3/envs/mmd_flow_hudson}

cd "$REPO_DIR"
export PATH="$CONDA_ENV/bin:$PATH"

export NGPU=${NGPU:-8}
export MASTER_PORT=${MASTER_PORT:-6668}
export CONFIG=${CONFIG:-configs/gen/imagenet64_riesz.yaml}
export WORKDIR=${WORKDIR:-/SAN/intelsys/imagenet_mmd_flow/}
export DRIFT_COMPILE=${DRIFT_COMPILE:-0}
export DRIFT_FEAT_CHUNK=${DRIFT_FEAT_CHUNK:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

echo "Job:        ${JOB_ID:-N/A}"
echo "Node:       ${HOSTNAME:-N/A}"
echo "GPUs:       $NGPU"
echo "Config:     $CONFIG"
echo "Workdir:    $WORKDIR"
echo "Repository: $REPO_DIR"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NGPU" \
  --master_port="$MASTER_PORT" \
  train.py \
  --config "$CONFIG" \
  --workdir "$WORKDIR"
