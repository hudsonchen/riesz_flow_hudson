set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/../accelerator_count.sh"
NGPU=${NGPU:-$(accelerator_count)}

MASTER_PORT=${MASTER_PORT:-6667}

CONFIG=configs/gen/ablation_ot_1node.yaml
EXP_NAME=ablation_ot_1node

WORKDIR=/path/to/workdir/$EXP_NAME
WANDB_PROJECT=YOUR_WANDB_PROJECT
WANDB_NAME=$EXP_NAME

DRIFT_COMPILE=1 \
NCCL_DEBUG=WARN \
WANDB_PROJECT=$WANDB_PROJECT \
WANDB_NAME=$WANDB_NAME \
torchrun \
    --nproc_per_node="$NGPU" \
    --master_port="$MASTER_PORT" \
    train.py \
    --config "$CONFIG" \
    --workdir "$WORKDIR"


echo "finished!"
