#!/bin/bash -l
#SBATCH --job-name=imagenet256_fid_nb
#SBATCH --account=airr-p109-dawn-gpu
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "This script must be submitted through Slurm:" >&2
    printf '  sbatch %q\n' "$0" >&2
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
NOTEBOOK=${NOTEBOOK:-"${REPO_DIR}/notebooks/fid_imagenet256.ipynb"}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_DIR}/artifacts/fid_imagenet256/notebooks"}
OUTPUT_NAME=${OUTPUT_NAME:-"fid_imagenet256_${SLURM_JOB_ID}.ipynb"}
KERNEL_NAME=${KERNEL_NAME:-python3}

cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR"
test -f "$NOTEBOOK" || { echo "Missing notebook: $NOTEBOOK" >&2; exit 1; }
python -c 'import jupyter, nbconvert, torch, tqdm; assert hasattr(torch, "xpu") and torch.xpu.is_available(), "Intel XPU is unavailable"' || {
    echo "The environment needs Jupyter, nbconvert, tqdm, and an XPU-enabled PyTorch." >&2
    exit 1
}

export WFLOW_CACHE_ROOT=${WFLOW_CACHE_ROOT:-"${RDS_ROOT}/cache"}
export WFLOW_HF_ROOT=${WFLOW_HF_ROOT:-"${WFLOW_CACHE_ROOT}/wflow_hf_root"}
export WFLOW_DRIFTING_HF_ROOT=${WFLOW_DRIFTING_HF_ROOT:-"${WFLOW_CACHE_ROOT}/drifting_hf_root"}
export WFLOW_VAE_HF_PATH=${WFLOW_VAE_HF_PATH:-"${WFLOW_CACHE_ROOT}/sdvae_hf_root"}
export TORCH_HUB_DIR=${TORCH_HUB_DIR:-"${WFLOW_CACHE_ROOT}/torch_hub"}
export IMAGENET_FID_NPZ=${IMAGENET_FID_NPZ:-"${WFLOW_HF_ROOT}/stats/jit_in256_stats.npz"}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-24}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-24}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

echo "Host:            $(hostname)"
echo "Start time:      $(date)"
echo "Repository:      $REPO_DIR"
echo "Notebook:        $NOTEBOOK"
echo "FID reference:   $IMAGENET_FID_NPZ"
echo "Executed output: ${OUTPUT_DIR}/${OUTPUT_NAME}"

jupyter nbconvert \
    --execute \
    --to notebook \
    --ExecutePreprocessor.timeout=-1 \
    --ExecutePreprocessor.kernel_name="$KERNEL_NAME" \
    --output-dir="$OUTPUT_DIR" \
    --output="$OUTPUT_NAME" \
    "$NOTEBOOK"

echo "End time: $(date)"
