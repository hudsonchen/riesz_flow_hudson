#!/bin/bash -l
#SBATCH --job-name=riesz_imagenet64
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

# Match the Dawn environment used by the ImageNet-256 launcher. Intel oneAPI
# module/Conda hooks can reference unset variables, so relax nounset only while
# those external shell hooks run.
set +u
module purge
module load rhel9/default-dawn
module load intelpython-conda
conda activate mmd_flow
set -u

REPO_DIR=${REPO_DIR:-/home/rc-chen1/riesz_flow_hudson}
RDS_ROOT=${RDS_ROOT:-/home/rc-chen1/rds/rds-airr-p109-tfgYl93jDnM}

cd "$REPO_DIR"

export WFLOW_CACHE_ROOT="${RDS_ROOT}/cache"
export IMAGENET_PATH="${RDS_ROOT}/ILSVRC/Data/CLS-LOC"
export PYTHONUNBUFFERED=1

# Prevent each DataLoader worker from spawning many CPU threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

NGPU=${NGPU:-${SLURM_GPUS_ON_NODE:-4}}
MASTER_PORT=${MASTER_PORT:-6668}
CONFIG=${CONFIG:-configs/gen/imagenet64_riesz.yaml}
WORKDIR=${WORKDIR:-"${RDS_ROOT}/runs/imagenet64_riesz_ps_1"}

mkdir -p "$WORKDIR"

echo "Host:          $(hostname)"
echo "Start time:    $(date)"
echo "Repository:    ${REPO_DIR}"
echo "ImageNet root: ${IMAGENET_PATH}"
echo "Cache root:    ${WFLOW_CACHE_ROOT}"
echo "Workdir:       ${WORKDIR}"
echo "GPUs:          ${NGPU}"

DRIFT_COMPILE=${DRIFT_COMPILE:-0} \
DRIFT_FEAT_CHUNK=${DRIFT_FEAT_CHUNK:-1} \
NCCL_DEBUG=${NCCL_DEBUG:-WARN} \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NGPU" \
    --master_port="$MASTER_PORT" \
    train.py \
    --config "$CONFIG" \
    --workdir "$WORKDIR"

echo "End time: $(date)"
