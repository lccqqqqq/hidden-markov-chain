#!/bin/bash
# Usage:
#   1. Create sweep:    wandb sweep sweep_config.yaml
#   2. Launch agents:   bash submit_sweep.sh <sweep-id> [num_agents]
#
# Example:
#   wandb sweep sweep_config.yaml          # prints: Created sweep with ID: abc123
#   bash submit_sweep.sh your-entity/pcfg/abc123 4

SWEEP_ID=${1:?Usage: bash submit_sweep.sh <sweep-id> [num_agents]}
NUM_AGENTS=${2:-1}

GPUS=1
GPU_QUEUE=gpulong
NCORES=2
MEM_PER_CORE_GB=5

for i in $(seq 1 $NUM_AGENTS); do
    addqueue -q $GPU_QUEUE --gpus $GPUS \
        -o out/cylinder_graph_hmm_sweep_agent_${i}_%j.out \
        -s -n $NCORES -m $MEM_PER_CORE_GB \
        /usr/bin/python3 -m wandb agent $SWEEP_ID
done

echo "Submitted $NUM_AGENTS sweep agents for $SWEEP_ID"
