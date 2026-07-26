#!/bin/bash -l
#SBATCH --job-name=imagenet256_latents
#SBATCH --account=airr-p109-dawn-gpu
#SBATCH --partition=pvc9

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00

#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module purge
module load rhel9/default-dawn

# Activate the existing environment.
source "${HOME}/.conda/etc/profile.d/conda.sh"
conda activate mmd_flow

cd /home/rc-chen1/riesz_flow_hudson

export WFLOW_CACHE_ROOT=/home/rc-chen1/rds/rds-airr-p109-tfgYl93jDnM/cache
export PYTHONUNBUFFERED=1

# Prevent each DataLoader worker from spawning many CPU threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

DATA_ROOT=/home/rc-chen1/rds/rds-airr-p109-tfgYl93jDnM/ILSVRC/Data/CLS-LOC
TARGET_ROOT="${WFLOW_CACHE_ROOT}/imagenet256-latents-sdvae"
VAE_ROOT="${WFLOW_CACHE_ROOT}/sdvae_hf_root"

# Preflight checks.
test -d "${DATA_ROOT}/train" || {
    echo "Missing ImageNet train directory: ${DATA_ROOT}/train"
    exit 1
}

test -d "${DATA_ROOT}/val" || {
    echo "Missing ImageNet val directory: ${DATA_ROOT}/val"
    exit 1
}

test -f "${VAE_ROOT}/config.json" || {
    echo "Missing SD-VAE config: ${VAE_ROOT}/config.json"
    exit 1
}

test -f "${VAE_ROOT}/diffusion_pytorch_model.safetensors" || {
    echo "Missing SD-VAE weights: ${VAE_ROOT}/diffusion_pytorch_model.safetensors"
    exit 1
}

mkdir -p "${TARGET_ROOT}"

echo "Host:        $(hostname)"
echo "Start time:  $(date)"
echo "Data root:   ${DATA_ROOT}"
echo "Target root: ${TARGET_ROOT}"
echo "VAE root:    ${VAE_ROOT}"
echo "CPUs:        ${SLURM_CPUS_PER_TASK}"

python - <<'PY'
from utils.env import VAE_HF_PATH, IMAGENET_CACHE_PATH
from utils.dist_util import local_device

print("VAE_HF_PATH        =", VAE_HF_PATH)
print("IMAGENET_CACHE_PATH=", IMAGENET_CACHE_PATH)
print("Device             =", local_device())
PY

srun python -u -m dataset.latent \
    --data-path "${DATA_ROOT}" \
    --target-path "${TARGET_ROOT}" \
    --local-batch-size 128 \
    --num-workers 16 \
    --prefetch-factor 2 \
    --pin-memory

echo "End time: $(date)"

ls -lh "${TARGET_ROOT}"
