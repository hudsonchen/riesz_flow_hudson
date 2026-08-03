#!/bin/bash -l
#SBATCH --job-name=riesz_B_imagenet256
#SBATCH --account=airr-p109-dawn-gpu
#SBATCH --partition=pvc9
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=1-12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "This script must be submitted through Slurm:"
    printf '  sbatch %q\n' "${SUBMIT_SCRIPT:-$0}"
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
TRAIN_SLURM_DIR=${TRAIN_SLURM_DIR:-"${REPO_DIR}/scripts/train_slurm"}

cd "$REPO_DIR"

export WFLOW_CACHE_ROOT=${WFLOW_CACHE_ROOT:-"${RDS_ROOT}/cache"}
export WFLOW_DRIFTING_HF_ROOT=${WFLOW_DRIFTING_HF_ROOT:-"${WFLOW_CACHE_ROOT}/drifting_hf_root"}
export IMAGENET_PATH=${IMAGENET_PATH:-"${RDS_ROOT}/ILSVRC/Data/CLS-LOC"}
export IMAGENET_CACHE_PATH=${IMAGENET_CACHE_PATH:-"${WFLOW_CACHE_ROOT}/imagenet256-latents-sdvae"}
export PYTHONUNBUFFERED=1

# Prevent each DataLoader worker from spawning many CPU threads.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

NNODES=${SLURM_JOB_NUM_NODES:-4}
NGPU=${SLURM_GPUS_ON_NODE:-4}
MASTER_PORT=${MASTER_PORT:-6669}
CONFIG=${CONFIG:-configs/gen/imagenet256_B_riesz.yaml}
RUN_NAME=${RUN_NAME:-imagenet256_B_riesz_4nodes}
WORKDIR=${WORKDIR:-"${RDS_ROOT}/runs/${RUN_NAME}"}
RANK_ENTRYPOINT=${RANK_ENTRYPOINT:-"${TRAIN_SLURM_DIR}/torchrun_isolated_cache_entrypoint.py"}

mapfile -t NODE_HOSTS < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
MASTER_ADDR=${MASTER_ADDR:-"${NODE_HOSTS[0]}"}

test -f "$CONFIG" || {
    echo "Missing config: $CONFIG" >&2
    exit 1
}
test -f "$RANK_ENTRYPOINT" || {
    echo "Missing torchrun rank entrypoint: $RANK_ENTRYPOINT" >&2
    exit 1
}

for cache_file in \
    train_moments.npy \
    train_moments_flip.npy \
    train_targets.npy \
    val_moments.npy \
    val_moments_flip.npy \
    val_targets.npy; do
    test -f "${IMAGENET_CACHE_PATH}/${cache_file}" || {
        echo "Missing ImageNet-256 latent cache file: ${IMAGENET_CACHE_PATH}/${cache_file}" >&2
        exit 1
    }
done

MAE_ROOT="${WFLOW_DRIFTING_HF_ROOT}/models/mae/jax/mae_latent_640"
test -f "${MAE_ROOT}/metadata.json" || {
    echo "Missing mae_latent_640 metadata: ${MAE_ROOT}/metadata.json" >&2
    exit 1
}
if [[ -f "${MAE_ROOT}/ema_params.pt" ]]; then
    MAE_WEIGHT_FILE=ema_params.pt
elif [[ -f "${MAE_ROOT}/ema_params.msgpack" ]]; then
    MAE_WEIGHT_FILE=ema_params.msgpack
else
    echo "Missing mae_latent_640 weights under: ${MAE_ROOT}" >&2
    exit 1
fi

mkdir -p "$WORKDIR"

echo "Host:           $(hostname)"
echo "Start time:     $(date)"
echo "Repository:     ${REPO_DIR}"
echo "Config:         ${CONFIG}"
echo "Run name:       ${RUN_NAME}"
echo "Cache root:     ${WFLOW_CACHE_ROOT}"
echo "Latent cache:   ${IMAGENET_CACHE_PATH}"
echo "MAE-640:        ${MAE_ROOT}"
echo "MAE weights:    ${MAE_WEIGHT_FILE}"
echo "Workdir:        ${WORKDIR}"
echo "Nodes:          ${NNODES}"
echo "GPUs per node:  ${NGPU}"
echo "Total GPUs:     $((NNODES * NGPU))"
echo "Master address: ${MASTER_ADDR}"
echo "Master port:    ${MASTER_PORT}"

export REPO_DIR RDS_ROOT NNODES NGPU MASTER_ADDR MASTER_PORT CONFIG RUN_NAME WORKDIR RANK_ENTRYPOINT
export SHARED_MAE_ROOT="${MAE_ROOT}" MAE_WEIGHT_FILE
export DRIFT_COMPILE=${DRIFT_COMPILE:-1}
export DRIFT_FEAT_CHUNK=${DRIFT_FEAT_CHUNK:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# Slurm starts one torchrun agent per node; torchrun then starts one XCCL
# worker per XPU.  Do not let oneCCL's MPI/Hydra defaults attach those child
# workers to the Slurm PMI world.
export CCL_PROCESS_LAUNCHER=torchrun
export CCL_ATL_TRANSPORT=ofi

srun \
    --mpi=none \
    --export=ALL \
    --ntasks="$NNODES" \
    --ntasks-per-node=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-64}" \
    --kill-on-bad-exit=1 \
    bash -c '
        set -euo pipefail
        cd "$REPO_DIR"

        echo "Host: $(hostname), node rank: ${SLURM_NODEID}"

        # There is one srun task per node at this point. Stage the large MAE
        # artifact once, before torchrun creates four local worker processes.
        # The workers then read node-local storage instead of all reading RDS.
        node_tmpdir=${SLURM_TMPDIR:-/tmp}
        local_hf_root="${node_tmpdir}/wflow-drifting-hf-root"
        local_mae_root="${local_hf_root}/models/mae/jax/mae_latent_640"
        local_mae_weight="${local_mae_root}/${MAE_WEIGHT_FILE}"
        shared_mae_weight="${SHARED_MAE_ROOT}/${MAE_WEIGHT_FILE}"
        required_bytes=$(stat -c %s "$shared_mae_weight")

        if [[ ! -f "${local_mae_root}/metadata.json" || ! -f "$local_mae_weight" || \
              $(stat -c %s "$local_mae_weight" 2>/dev/null || echo 0) -ne $required_bytes ]]; then
            available_bytes=$(df -B1 --output=avail "$node_tmpdir" | tail -n 1 | tr -d " ")
            reserve_bytes=$((2 * 1024 * 1024 * 1024))
            if (( available_bytes < required_bytes + reserve_bytes )); then
                echo "Insufficient node-local space on $(hostname): need $((required_bytes + reserve_bytes)) bytes, have ${available_bytes}" >&2
                exit 1
            fi

            if [[ -e "$local_mae_root" ]]; then
                echo "Incomplete pre-existing node-local MAE directory: ${local_mae_root}" >&2
                exit 1
            fi

            stage_tmp=$(mktemp -d "${node_tmpdir}/wflow-mae-stage.XXXXXX")
            staged_mae_root="${stage_tmp}/mae_latent_640"
            mkdir -p "$staged_mae_root" "$(dirname "$local_mae_root")"
            stage_start=$SECONDS
            echo "[$(date --iso-8601=seconds)] Staging MAE-640 on $(hostname): ${shared_mae_weight} -> ${local_mae_weight}"
            cp -p "${SHARED_MAE_ROOT}/metadata.json" "$staged_mae_root/metadata.json"
            cp -p "$shared_mae_weight" "$staged_mae_root/$MAE_WEIGHT_FILE"

            staged_bytes=$(stat -c %s "$staged_mae_root/$MAE_WEIGHT_FILE")
            if (( staged_bytes != required_bytes )); then
                echo "Node-local MAE size mismatch on $(hostname): expected ${required_bytes}, got ${staged_bytes}" >&2
                exit 1
            fi

            mv "$staged_mae_root" "$local_mae_root"
            rmdir "$stage_tmp"
            echo "[$(date --iso-8601=seconds)] MAE-640 staged on $(hostname) in $((SECONDS - stage_start)) s (${required_bytes} bytes)"
        else
            echo "[$(date --iso-8601=seconds)] Reusing staged MAE-640 on $(hostname): ${local_mae_weight}"
        fi

        export WFLOW_DRIFTING_HF_ROOT="$local_hf_root"
        echo "Node-local WFLOW_DRIFTING_HF_ROOT=${WFLOW_DRIFTING_HF_ROOT}"

        torchrun \
            --nnodes="$NNODES" \
            --nproc_per_node="$NGPU" \
            --node_rank="$SLURM_NODEID" \
            --master_addr="$MASTER_ADDR" \
            --master_port="$MASTER_PORT" \
            "$RANK_ENTRYPOINT" \
            --config "$CONFIG" \
            --workdir "$WORKDIR"
    '

echo "End time: $(date)"
