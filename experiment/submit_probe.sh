#!/bin/bash
# Run linear probes on a trained model directory via the cluster.
# Usage: bash experiment/submit_probe.sh <model_dir>
#   e.g. bash experiment/submit_probe.sh models/mess3/20260306_150920_L4_d64_H1_full_LN/

MODEL_DIR=${1:?Usage: bash submit_probe.sh <model_dir>}

# Read seq_length from the model's config
SEQ_LENGTH=$(python3 -c "
import yaml
with open('${MODEL_DIR}/config.yaml') as f:
    cfg = yaml.safe_load(f)
print(cfg['model']['n_ctx'])
")

QUEUE=long
NCORES=1
MEM_PER_CORE_GB=60

echo "Submitting probe on ${MODEL_DIR} with seq_length=${SEQ_LENGTH}"

addqueue -c "linear probe" -q $QUEUE \
    -s -n $NCORES -m $MEM_PER_CORE_GB \
    -o out/probe_%j.out \
    /usr/bin/python3 -u src/run_linear_probe.py \
        --model-dir "$MODEL_DIR" \
        --seq-length "$SEQ_LENGTH" \
        --device cpu \
        --save-data
