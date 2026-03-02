#!/bin/bash
# Submit 5 fixed-embedding training jobs (one per embedding mode).
#
# Usage:
#   bash submit_fixed_embedding.sh <checkpoint_dir>
#
# Example:
#   bash submit_fixed_embedding.sh models/cylinderhmm/20260206_135142_L4_d8_full_noLN
#
# The script reads config.yaml from the checkpoint directory and uses
# best_model.pt as the checkpoint for the "trained" embedding mode.

CKPT_DIR=${1:?Usage: bash submit_fixed_embedding.sh <checkpoint_dir>}

# Validate inputs
CONFIG="${CKPT_DIR}/config.yaml"
BEST_MODEL="${CKPT_DIR}/best_model.pt"

if [ ! -f "$CONFIG" ]; then
    echo "Error: config.yaml not found in ${CKPT_DIR}"
    exit 1
fi

if [ ! -f "$BEST_MODEL" ]; then
    echo "Error: best_model.pt not found in ${CKPT_DIR}"
    exit 1
fi

# Cluster resources
GPUS=1
GPU_QUEUE=${2:-gpulong}
NCORES=2
MEM_PER_CORE_GB=5

MODES="trained pmi random onehot"

for MODE in $MODES; do
    EXTRA_ARGS=""
    if [ "$MODE" = "trained" ]; then
        EXTRA_ARGS="--checkpoint_path ${BEST_MODEL}"
    fi

    addqueue -q $GPU_QUEUE --gpus $GPUS \
        -o "out/fixed_${MODE}_%j.out" \
        -s -n $NCORES -m $MEM_PER_CORE_GB \
        /usr/bin/python3 src/train_fixed_embedding.py \
            --embedding_mode $MODE \
            --config "$CONFIG" \
            $EXTRA_ARGS

    echo "Submitted: ${MODE}"
done

echo ""
echo "Submitted 5 fixed-embedding jobs using config from ${CKPT_DIR}"
