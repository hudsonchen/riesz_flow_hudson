#!/bin/bash -l
#SBATCH --job-name=riesz_rms_ablation_imagenet256
#SBATCH --account=airr-p109-dawn-gpu
#SBATCH --partition=pvc9
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=4
#SBATCH --time=1-12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/rc-chen1/riesz_flow_hudson}
export REPO_DIR
export SUBMIT_SCRIPT=${SUBMIT_SCRIPT:-$0}
export CONFIG=${CONFIG:-configs/gen/imagenet256_ablation_riesz_rms.yaml}
export RUN_NAME=${RUN_NAME:-imagenet256_ablation_riesz_frozen_velocity_rms_2nodes}
export MAE_MODEL=${MAE_MODEL:-mae_latent_256}

test -f "${REPO_DIR}/riesz_rms.py" || {
    echo "Missing RMS Riesz loss: ${REPO_DIR}/riesz_rms.py" >&2
    exit 1
}

exec "${REPO_DIR}/scripts/train_slurm/imagenet256_B_riesz_slurm.sh"
