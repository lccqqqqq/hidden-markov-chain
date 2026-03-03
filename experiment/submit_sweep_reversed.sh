#!/bin/bash
# Usage:
#   1. Create sweep:    wandb sweep config/sweeps/sweep_config_reversed.yaml
#   2. Launch agents:   bash experiment/submit_sweep_reversed.sh <sweep-id> [num_agents]
#
# Example:
#   wandb sweep config/sweeps/sweep_config_reversed.yaml    # prints: Created sweep with ID: abc123
#   bash experiment/submit_sweep_reversed.sh your-entity/pcfg/abc123 6

SWEEP_ID=${1:?Usage: bash experiment/submit_sweep_reversed.sh <sweep-id> [num_agents]}
NUM_AGENTS=${2:-1}

GPUS=1
GPU_QUEUE=gpulong
NCORES=2
MEM_PER_CORE_GB=5

for i in $(seq 1 $NUM_AGENTS); do
    addqueue -q $GPU_QUEUE --gpus $GPUS \
        -o out/cylinder_graph_hmm_reversed_sweep_agent_${i}_%j.out \
        -s -n $NCORES -m $MEM_PER_CORE_GB \
        /usr/bin/env TRAIN_CONFIG=config/base_config_reversed.yaml \
        /usr/bin/python3 -m wandb agent $SWEEP_ID
done

echo "Submitted $NUM_AGENTS reversed sweep agents for $SWEEP_ID"
