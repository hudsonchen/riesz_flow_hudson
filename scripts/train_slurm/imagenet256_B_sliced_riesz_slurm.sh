#!/bin/bash -l
#SBATCH --job-name=sliced_riesz_B_imagenet256
#SBATCH --account=airr-p109-dawn-gpu
#SBATCH --partition=pvc9
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

REPO_DIR=${REPO_DIR:-/home/rc-chen1/riesz_flow_hudson}
export REPO_DIR
export SUBMIT_SCRIPT=${SUBMIT_SCRIPT:-$0}
export CONFIG=${CONFIG:-configs/gen/imagenet256_B_sliced_riesz.yaml}
export RUN_NAME=${RUN_NAME:-imagenet256_B_sliced_riesz}

exec "${REPO_DIR}/scripts/train_slurm/imagenet256_B_riesz_slurm.sh"
