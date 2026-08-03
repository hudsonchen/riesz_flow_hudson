#!/bin/bash -l
#SBATCH --job-name=imagenet256_fid
#SBATCH --account=airr-p109-dawn-gpu
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this file with sbatch." >&2
    exit 1
fi

set +u
module purge
module load rhel9/default-dawn
module load intelpython-conda
conda activate mmd_flow
set -u

REPO_DIR=${REPO_DIR:-/home/rc-chen1/riesz_flow_hudson}
RDS_ROOT=${RDS_ROOT:-/home/rc-chen1/rds/rds-airr-p109-tfgYl93jDnM}

cd "$REPO_DIR"
test -f compute_imagenet256_fid.py || {
    echo "Missing evaluator: ${REPO_DIR}/compute_imagenet256_fid.py" >&2
    exit 1
}

export WFLOW_CACHE_ROOT=${WFLOW_CACHE_ROOT:-"${RDS_ROOT}/cache"}
export WFLOW_HF_ROOT=${WFLOW_HF_ROOT:-"${WFLOW_CACHE_ROOT}/wflow_hf_root"}
export WFLOW_DRIFTING_HF_ROOT=${WFLOW_DRIFTING_HF_ROOT:-"${WFLOW_CACHE_ROOT}/drifting_hf_root"}
export WFLOW_VAE_HF_PATH=${WFLOW_VAE_HF_PATH:-"${WFLOW_CACHE_ROOT}/sdvae_hf_root"}
export TORCH_HUB_DIR=${TORCH_HUB_DIR:-"${WFLOW_CACHE_ROOT}/torch_hub"}
export IMAGENET_FID_NPZ=${IMAGENET_FID_NPZ:-"${WFLOW_HF_ROOT}/stats/jit_in256_stats.npz"}
export PYTHONUNBUFFERED=1

# One process performs XPU generation and CPU Inception feature extraction.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-24}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-24}

echo "Host:       $(hostname)"
echo "Start time: $(date)"
echo "Repository: $REPO_DIR"
echo "Device:     one Dawn PVC"

srun --mpi=none python compute_imagenet256_fid.py

echo "End time:   $(date)"
