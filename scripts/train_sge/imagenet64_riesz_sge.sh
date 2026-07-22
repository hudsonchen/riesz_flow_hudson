#!/usr/bin/env bash
#$ -N imagenet64_riesz
#$ -P aihub_ucl
#$ -cwd
#$ -V
#$ -l gpu=true,gpu_type=h100
#$ -pe gpu 8
#$ -l tmem=10G
#$ -l h_rt=1:00:00
#$ -R y
#$ -j y
#$ -o /home/zongchen/

set -euo pipefail

REPO_DIR=${RIESZ_FLOW_REPO_DIR:-/home/zongchen/riesz_flow_hudson}
CONDA_ENV=${CONDA_ENV:-/home/zongchen/miniconda3/envs/mmd_flow_hudson}

cd "$REPO_DIR"
export PATH="$CONDA_ENV/bin:$PATH"

if [[ ! -f train.py ]]; then
  echo "Error: train.py was not found in repository directory: $REPO_DIR" >&2
  echo "Set RIESZ_FLOW_REPO_DIR if the repository is installed elsewhere." >&2
  exit 1
fi

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
  --nnodes=1 \
  --nproc_per_node="$NGPU" \
  --master_addr=127.0.0.1 \
  --master_port="$MASTER_PORT" \
  train.py \
  --config "$CONFIG" \
  --workdir "$WORKDIR"
