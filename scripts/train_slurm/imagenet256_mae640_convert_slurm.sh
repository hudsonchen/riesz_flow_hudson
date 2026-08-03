#!/bin/bash -l
#SBATCH --job-name=convert_mae640_pt
#SBATCH --account=airr-p109-dawn-gpu
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "This script must be submitted through Slurm:"
    printf '  sbatch %q\n' "${SUBMIT_SCRIPT:-$0}"
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
export WFLOW_CACHE_ROOT=${WFLOW_CACHE_ROOT:-"${RDS_ROOT}/cache"}
export WFLOW_DRIFTING_HF_ROOT=${WFLOW_DRIFTING_HF_ROOT:-"${WFLOW_CACHE_ROOT}/drifting_hf_root"}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}

cd "$REPO_DIR"

MAE_ROOT="${WFLOW_DRIFTING_HF_ROOT}/models/mae/jax/mae_latent_640"
test -f "${MAE_ROOT}/metadata.json" || {
    echo "Missing MAE-640 metadata: ${MAE_ROOT}/metadata.json" >&2
    exit 1
}
test -f "${MAE_ROOT}/ema_params.msgpack" || {
    echo "Missing MAE-640 MsgPack weights: ${MAE_ROOT}/ema_params.msgpack" >&2
    exit 1
}

echo "Host:       $(hostname)"
echo "Start time: $(date)"
echo "Repository: ${REPO_DIR}"
echo "MAE source: ${MAE_ROOT}/ema_params.msgpack"
echo "MAE output: ${MAE_ROOT}/ema_params.pt"

python -u misc/convert_mae_to_pt.py mae_latent_640

test -s "${MAE_ROOT}/ema_params.pt" || {
    echo "Conversion did not produce a non-empty ${MAE_ROOT}/ema_params.pt" >&2
    exit 1
}

echo "Converted MAE-640 artifact:"
ls -lh "${MAE_ROOT}/ema_params.pt"
echo "End time: $(date)"
