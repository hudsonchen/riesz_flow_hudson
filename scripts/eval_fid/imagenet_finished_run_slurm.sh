#!/bin/bash -l
#SBATCH --job-name=imagenet_fid50k
#SBATCH --account=airr-p109-dawn-gpu
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "This script must be submitted through Slurm:"
    printf '  sbatch %q\n' "$0"
    exit 1
fi

# Intel oneAPI module/Conda hooks can reference unset variables. Relax nounset
# only while those external shell hooks run, then restore strict mode.
set +u
module purge
module load rhel9/default-dawn
module load intelpython-conda
conda activate mmd_flow
set -u

REPO_DIR=${REPO_DIR:-/home/rc-chen1/riesz_flow_hudson}
RDS_ROOT=${RDS_ROOT:-/home/rc-chen1/rds/rds-airr-p109-tfgYl93jDnM}
TARGET=${TARGET:-imagenet256}

case "$TARGET" in
    imagenet64)
        CONFIG=${CONFIG:-configs/gen/imagenet64_riesz.yaml}
        RUN_DIR=${RUN_DIR:-"${RDS_ROOT}/runs/imagenet64_riesz"}
        ;;
    imagenet256)
        CONFIG=${CONFIG:-configs/gen/imagenet256_B_riesz.yaml}
        RUN_DIR=${RUN_DIR:-"${RDS_ROOT}/runs/imagenet256_B_riesz"}
        ;;
    *)
        echo "TARGET must be imagenet64 or imagenet256, got: ${TARGET}" >&2
        exit 1
        ;;
esac

cd "$REPO_DIR"

export WFLOW_CACHE_ROOT=${WFLOW_CACHE_ROOT:-"${RDS_ROOT}/cache"}
export WFLOW_DRIFTING_HF_ROOT=${WFLOW_DRIFTING_HF_ROOT:-"${WFLOW_CACHE_ROOT}/drifting_hf_root"}
export WFLOW_HF_ROOT=${WFLOW_HF_ROOT:-"${WFLOW_CACHE_ROOT}/wflow_hf_root"}
export WFLOW_VAE_HF_PATH=${WFLOW_VAE_HF_PATH:-"${WFLOW_CACHE_ROOT}/sdvae_hf_root"}
export IMAGENET_FID_NPZ=${IMAGENET_FID_NPZ:-"${WFLOW_HF_ROOT}/stats/jit_in256_stats.npz"}
export IMAGENET64_FID_NPZ=${IMAGENET64_FID_NPZ:-"${WFLOW_HF_ROOT}/stats/VIRTUAL_imagenet64_labeled.npz"}
export PYTHONUNBUFFERED=1

# Prevent metric extraction and image writing from oversubscribing the CPUs.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

NGPU=${NGPU:-4}
MASTER_PORT=${MASTER_PORT:-6670}
NUM_SAMPLES=${NUM_SAMPLES:-50000}
GEN_BSZ=${GEN_BSZ:-64}
SEED=${SEED:-0}
RESULT_DIR=${RESULT_DIR:-"${RUN_DIR}/fid"}
CFG_SCALES=${CFG_SCALES:-}
CKPT=${CKPT:-}
KEEP_SAMPLES=${KEEP_SAMPLES:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}

test -f "$CONFIG" || {
    echo "Missing config: $CONFIG" >&2
    exit 1
}
test -d "$RUN_DIR" || {
    echo "Missing training run directory: $RUN_DIR" >&2
    exit 1
}

if [[ "$TARGET" == "imagenet64" ]]; then
    FID_REF=$IMAGENET64_FID_NPZ
else
    FID_REF=$IMAGENET_FID_NPZ
fi
test -f "$FID_REF" || {
    echo "Missing ${TARGET} FID reference statistics: ${FID_REF}" >&2
    exit 1
}

EVAL_ARGS=(
    --run-dir "$RUN_DIR"
    --config "$CONFIG"
    --num-samples "$NUM_SAMPLES"
    --gen-bsz "$GEN_BSZ"
    --seed "$SEED"
    --fid-ref "$FID_REF"
    --result-dir "$RESULT_DIR"
)
if [[ -n "$CFG_SCALES" ]]; then
    EVAL_ARGS+=(--cfg-scales "$CFG_SCALES")
fi
if [[ -n "$CKPT" ]]; then
    EVAL_ARGS+=(--ckpt "$CKPT")
fi
if [[ "$KEEP_SAMPLES" == "1" ]]; then
    EVAL_ARGS+=(--keep-samples)
fi
if [[ "$SKIP_EXISTING" == "1" ]]; then
    EVAL_ARGS+=(--skip-existing)
fi

mkdir -p "$RESULT_DIR"

echo "Host:             $(hostname)"
echo "Start time:       $(date)"
echo "Repository:       ${REPO_DIR}"
echo "Target:           ${TARGET}"
echo "Config:           ${CONFIG}"
echo "Training run:     ${RUN_DIR}"
echo "Checkpoint:       ${CKPT:-latest in the run}"
echo "FID reference:    ${FID_REF}"
echo "CFG scales:       ${CFG_SCALES:-from config train.cfg_list}"
echo "Samples per CFG:  ${NUM_SAMPLES}"
echo "Batch per GPU:    ${GEN_BSZ}"
echo "GPUs:             ${NGPU}"
echo "Result directory: ${RESULT_DIR}"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NGPU" \
    --master_port="$MASTER_PORT" \
    evaluate_checkpoint_fidutil.py \
    "${EVAL_ARGS[@]}"

echo "End time: $(date)"
