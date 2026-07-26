#!/usr/bin/env bash
#SBATCH --job-name=imagenet256_latents
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
PYTHON=${PYTHON:-python}

DATA_PATH=${DATA_PATH:-/home/rc-chen1/rds/rds-airr-p109-tfgYl93jDnM/ILSVRC/Data/CLS-LOC/}
TARGET_PATH=${TARGET_PATH:-/home/rc-chen1/rds/rds-airr-p109-tfgYl93jDnM/cache/drifting_hf_root/models/mae/jax/mae_latent_256/imagenet256-latents-sdvae}
LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-8}

cd "$REPO_DIR"

if [[ ! -f dataset/latent.py ]]; then
  echo "Error: dataset/latent.py was not found in repository: $REPO_DIR" >&2
  echo "Submit from the W-Flow checkout or set REPO_DIR." >&2
  exit 1
fi

echo "Job:         ${SLURM_JOB_ID:-N/A}"
echo "Node:        ${HOSTNAME:-N/A}"
echo "Repository:  $REPO_DIR"
echo "Data:        $DATA_PATH"
echo "Target:      $TARGET_PATH"
echo "Batch size:  $LOCAL_BATCH_SIZE"
echo "Workers:     $NUM_WORKERS"

"$PYTHON" -m dataset.latent \
  --data-path "$DATA_PATH" \
  --target-path "$TARGET_PATH" \
  --local-batch-size "$LOCAL_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --pin-memory
