#!/bin/bash -l
#SBATCH --job-name=imagenet256_B_sinkhorn
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

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export SUBMIT_SCRIPT=${SUBMIT_SCRIPT:-$0}
export CONFIG=${CONFIG:-configs/gen/imagenet256_B_sinkhorn.yaml}
export RUN_NAME=${RUN_NAME:-imagenet256_B_sinkhorn}

exec "${SCRIPT_DIR}/imagenet256_B_riesz_slurm.sh"
