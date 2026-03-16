#!/bin/bash
# Full pipeline: data prep → train 40 models → collect + analyze
# for the n_ctx=4 learning dynamics experiment.
#
# Submit as a single GPU job:
#   addqueue -c "ctx4 learning dynamics 40 runs" -q gpulong --gpus 1 \
#       -s -n 2 -m 8 -o out/learning_dynamics_ctx4_%j.out \
#       bash experiments/run_learning_dynamics_ctx4.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

N_SEEDS=20
OUT_DIR="out/learning_dynamics_ctx4"
CONFIG_FWD="config/ctx4_config.yaml"
CONFIG_REV="config/ctx4_config_reversed.yaml"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

echo "$(timestamp) === Phase 1: Data Preparation ==="

python3 -c "
import torch, os, numpy as np

for src_dir, dst_dir in [
    ('data/datasets/cylinder_graph_hmm', 'data/datasets/cylinder_graph_hmm_ctx4'),
    ('data/datasets/cylinder_graph_hmm_reversed', 'data/datasets/cylinder_graph_hmm_reversed_ctx4'),
]:
    for split in ['train', 'test']:
        src = os.path.join(src_dir, split, 'observations.pt')
        dst_path = os.path.join(dst_dir, split, 'observations.pt')

        if os.path.exists(dst_path):
            t = torch.load(dst_path, weights_only=True)
            print(f'SKIP {dst_path} (already exists, shape={t.shape})')
            continue

        data = torch.load(src, weights_only=True)  # (N, 17)
        print(f'Loaded {src}: shape={data.shape}')

        # Extract 3 non-overlapping windows of length 5: [0:5], [5:10], [10:15]
        windows = [data[:, 0:5], data[:, 5:10], data[:, 10:15]]
        combined = torch.cat(windows, dim=0)  # (3*N, 5)

        # Shuffle
        perm = torch.randperm(combined.shape[0])
        combined = combined[perm]

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        torch.save(combined, dst_path)
        print(f'Saved {dst_path}: shape={combined.shape}')

    # Copy metadata if it exists
    meta_src = os.path.join(src_dir, 'metadata.json')
    meta_dst = os.path.join(dst_dir, 'metadata.json')
    if os.path.exists(meta_src) and not os.path.exists(meta_dst):
        import shutil
        shutil.copy2(meta_src, meta_dst)
        print(f'Copied metadata to {meta_dst}')

print('Data preparation complete.')
"

echo "$(timestamp) === Phase 2: Reorganize old figures ==="

if [[ -d "figures/learning_dynamics" && ! -d "figures/learning_dynamics_ctx16" ]]; then
    mv figures/learning_dynamics figures/learning_dynamics_ctx16
    echo "Renamed figures/learning_dynamics -> figures/learning_dynamics_ctx16"
elif [[ -d "figures/learning_dynamics_ctx16" ]]; then
    echo "SKIP: figures/learning_dynamics_ctx16 already exists"
else
    echo "SKIP: no figures/learning_dynamics to rename"
fi

echo "$(timestamp) === Phase 3: Training 40 models ==="

mkdir -p "$OUT_DIR"
job_idx=0
total_jobs=$((2 * N_SEEDS))

for DIRECTION in forward reversed; do
    if [[ "$DIRECTION" == "forward" ]]; then
        CONFIG="$CONFIG_FWD"
    else
        CONFIG="$CONFIG_REV"
    fi

    for SEED in $(seq 0 $((N_SEEDS - 1))); do
        job_idx=$((job_idx + 1))
        CURVE_PATH="${OUT_DIR}/curve_${DIRECTION}_seed${SEED}.json"

        if [[ -f "$CURVE_PATH" ]]; then
            echo "$(timestamp) [${job_idx}/${total_jobs}] SKIP ${DIRECTION} seed=${SEED} (already exists)"
            continue
        fi

        echo "$(timestamp) [${job_idx}/${total_jobs}] Training ${DIRECTION} seed=${SEED}..."
        python3 src/train.py \
            --config "$CONFIG" \
            --model_seed "$SEED" \
            --loss_curve_path "$CURVE_PATH"
        echo "$(timestamp) [${job_idx}/${total_jobs}] Done ${DIRECTION} seed=${SEED}"
    done
done

echo "$(timestamp) === Phase 4: Collect and Analyze ==="

python3 src/learning_dynamics_experiment.py \
    --mode collect \
    --out_dir "$OUT_DIR" \
    --n_seeds "$N_SEEDS"

python3 src/analyze_learning_dynamics.py \
    --curves "${OUT_DIR}/all_curves.json" \
    --fig_dir "figures/learning_dynamics_ctx4" \
    --output "${OUT_DIR}/learning_dynamics_ctx4_analysis.json"

echo "$(timestamp) === All done ==="
echo "Curves:   ${OUT_DIR}/"
echo "Figures:  figures/learning_dynamics_ctx4/"
echo "Analysis: ${OUT_DIR}/learning_dynamics_ctx4_analysis.json"
