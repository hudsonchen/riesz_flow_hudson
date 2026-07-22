#!/usr/bin/env bash
#$ -N imagenet64_riesz
#$ -P aihub_ucl
#$ -cwd
#$ -V
#$ -l gpu=true,gpu_type=h100
#$ -pe gpu 2
#$ -l tmem=10G
#$ -l h_rt=1:00:00
#$ -R y
#$ -j y
#$ -o /home/zongchen/

set -euo pipefail

REPO_DIR=${RIESZ_FLOW_REPO_DIR:-/home/zongchen/riesz_flow_hudson}
SHARED_WFLOW_CACHE=${SHARED_WFLOW_CACHE:-/home/zongchen/riesz_flow_hudson/.cache}

eval "$(/home/zongchen/miniconda3/condabin/conda shell.bash hook)"
conda activate mmd_flow

cd "$REPO_DIR"

export WFLOW_VAE_HF_PATH=${WFLOW_VAE_HF_PATH:-$SHARED_WFLOW_CACHE/sdvae_hf_root}
export WFLOW_DRIFTING_HF_ROOT=${WFLOW_DRIFTING_HF_ROOT:-$SHARED_WFLOW_CACHE/drifting_hf_root}
export TORCH_HUB_DIR=${TORCH_HUB_DIR:-$SHARED_WFLOW_CACHE/torch_hub}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

if [[ ! -f train.py ]]; then
  echo "Error: train.py was not found in repository directory: $REPO_DIR" >&2
  echo "Set RIESZ_FLOW_REPO_DIR if the repository is installed elsewhere." >&2
  exit 1
fi

if [[ ! -f "$WFLOW_VAE_HF_PATH/config.json" ]]; then
  echo "Error: SD-VAE config not found at $WFLOW_VAE_HF_PATH/config.json" >&2
  exit 1
fi

MAE_METADATA="$WFLOW_DRIFTING_HF_ROOT/models/mae/jax/mae_latent_256/metadata.json"
if [[ ! -f "$MAE_METADATA" ]]; then
  echo "Error: latent MAE metadata not found at $MAE_METADATA" >&2
  exit 1
fi

export NGPU=${NGPU:-2}
export MASTER_PORT=${MASTER_PORT:-6668}
export CONFIG=${CONFIG:-configs/gen/imagenet64_riesz.yaml}
export WORKDIR=${WORKDIR:-/SAN/intelsys/imagenet_mmd_flow/}
export DRIFT_COMPILE=${DRIFT_COMPILE:-0}
export DRIFT_FEAT_CHUNK=${DRIFT_FEAT_CHUNK:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

echo "Job:        ${JOB_ID:-N/A}"
echo "Node:       ${HOSTNAME:-N/A}"
echo "SGE slots:  ${NSLOTS:-unknown}"
echo "CPU cores:  $(nproc)"
echo "GPUs:       $NGPU"
echo "Config:     $CONFIG"
echo "Workdir:    $WORKDIR"
echo "Repository: $REPO_DIR"
echo "VAE:        $WFLOW_VAE_HF_PATH"
echo "MAE root:   $WFLOW_DRIFTING_HF_ROOT"

torchrun \
  --nnodes=1 \
  --nproc_per_node="$NGPU" \
  --master_addr=127.0.0.1 \
  --master_port="$MASTER_PORT" \
  train.py \
  --config "$CONFIG" \
  --workdir "$WORKDIR"
