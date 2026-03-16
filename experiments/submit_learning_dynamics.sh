#!/bin/bash
# Submit multi-seed forward vs reversed training experiment.
# 20 seeds x 2 directions = 40 jobs on gpulong + kocsisgpu queues.
#
# Usage:
#   bash experiments/submit_learning_dynamics.sh           # submit all 40 jobs
#   bash experiments/submit_learning_dynamics.sh --dry-run # print commands only

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

N_SEEDS=20
OUT_DIR="out/learning_dynamics"
CONFIG_FWD="config/base_config.yaml"
CONFIG_REV="config/base_config_reversed.yaml"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN ==="
fi

mkdir -p "$OUT_DIR" out

# Alternate between gpulong and kocsisgpu to spread load
QUEUES=(gpulong kocsisgpu)
job_idx=0

for DIRECTION in forward reversed; do
    if [[ "$DIRECTION" == "forward" ]]; then
        CONFIG="$CONFIG_FWD"
    else
        CONFIG="$CONFIG_REV"
    fi

    for SEED in $(seq 0 $((N_SEEDS - 1))); do
        CURVE_PATH="${OUT_DIR}/curve_${DIRECTION}_seed${SEED}.json"

        # Skip if output already exists
        if [[ -f "$CURVE_PATH" ]]; then
            echo "SKIP: ${DIRECTION} seed=${SEED} (already exists)"
            continue
        fi

        QUEUE="${QUEUES[$((job_idx % ${#QUEUES[@]}))]}"
        COMMENT="learn_dyn ${DIRECTION} s${SEED} ~30min"

        CMD="addqueue -c \"${COMMENT}\" -q ${QUEUE} --gpus 1 \
    -s -n 2 -m 5 \
    -o out/learn_dyn_${DIRECTION}_s${SEED}_%j.out \
    /usr/bin/python3 src/train.py \
        --config ${CONFIG} \
        --model_seed ${SEED} \
        --loss_curve_path ${CURVE_PATH}"

        if $DRY_RUN; then
            echo "$CMD"
        else
            eval "$CMD"
            echo "Submitted: ${DIRECTION} seed=${SEED} -> ${QUEUE}"
        fi

        job_idx=$((job_idx + 1))
    done
done

echo ""
echo "Total jobs submitted: ${job_idx}"
echo "Output curves -> ${OUT_DIR}/curve_<direction>_seed<N>.json"
echo "Job logs      -> out/learn_dyn_<direction>_s<N>_<jobid>.out"
echo ""
echo "After all jobs finish, run:"
echo "  python src/learning_dynamics_experiment.py --mode collect"
echo "  python src/analyze_learning_dynamics.py"
