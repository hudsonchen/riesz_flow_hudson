#!/usr/bin/env bash
#SBATCH --job-name=imagenet64_riesz
#SBATCH --account=aihub_ucl
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=${REPO_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}
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

echo "Job:       ${SLURM_JOB_ID:-N/A}"
echo "Node:      ${SLURM_JOB_NODELIST:-N/A}"
echo "GPUs:      $NGPU"
echo "Config:    $CONFIG"
echo "Workdir:   $WORKDIR"
echo "Repository: $REPO_DIR"

srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-64}" \
  --kill-on-bad-exit=1 \
  torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NGPU" \
    --master_port="$MASTER_PORT" \
    train.py \
    --config "$CONFIG" \
    --workdir "$WORKDIR"
