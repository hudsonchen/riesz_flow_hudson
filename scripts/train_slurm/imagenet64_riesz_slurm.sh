#!/usr/bin/env bash
#SBATCH --job-name=imagenet64_riesz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=8:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

cd "${REPO_DIR:-/path/to/W-Flow}"
source "${ENV_ACTIVATE:-/path/to/venv/bin/activate}"

NGPU=${NGPU:-${SLURM_GPUS_ON_NODE:-8}}
MASTER_PORT=${MASTER_PORT:-6668}
CONFIG=${CONFIG:-configs/gen/imagenet64_riesz.yaml}
WORKDIR=${WORKDIR:-/path/to/workdir/imagenet64_riesz}

mkdir -p "$WORKDIR"

DRIFT_COMPILE=${DRIFT_COMPILE:-0} \
DRIFT_FEAT_CHUNK=${DRIFT_FEAT_CHUNK:-1} \
NCCL_DEBUG=${NCCL_DEBUG:-WARN} \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NGPU" \
    --master_port="$MASTER_PORT" \
    train.py \
    --config "$CONFIG" \
    --workdir "$WORKDIR"
