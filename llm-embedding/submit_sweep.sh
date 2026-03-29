#!/bin/bash
# Submit bilinear Deep Sets sweep over random model seeds.
# 3 configs × 20 seeds = 60 lightweight CPU jobs.

cd /mnt/users/clin/workspace/hidden-markov-chain/llm-embedding

COMMON_ARGS="--N 50000 --epochs 500 --lr 0.01 --batch_size 512"

# Config 1: A=2.0, p0=0.3, L=50
for seed in $(seq 0 19); do
    addqueue -c "deepsets A2 p0.3 L50 s${seed}" -q long \
        -s -n 1 -m 4 \
        -o out/cfg1_seed${seed}_%j.out \
        /usr/bin/python3 train.py --mode gaussian --A 2.0 --p0 0.3 --L 50 \
        ${COMMON_ARGS} --model_seed ${seed} --save_dir results/gaussian_A2.0_p0.3_L50
done

# Config 2: A=2.0, p0=0.3, L=200
for seed in $(seq 0 19); do
    addqueue -c "deepsets A2 p0.3 L200 s${seed}" -q long \
        -s -n 1 -m 4 \
        -o out/cfg2_seed${seed}_%j.out \
        /usr/bin/python3 train.py --mode gaussian --A 2.0 --p0 0.3 --L 200 \
        ${COMMON_ARGS} --model_seed ${seed} --save_dir results/gaussian_A2.0_p0.3_L200
done

# Config 3: A=2.0, p0=0.5, L=50
for seed in $(seq 0 19); do
    addqueue -c "deepsets A2 p0.5 L50 s${seed}" -q long \
        -s -n 1 -m 4 \
        -o out/cfg3_seed${seed}_%j.out \
        /usr/bin/python3 train.py --mode gaussian --A 2.0 --p0 0.5 --L 50 \
        ${COMMON_ARGS} --model_seed ${seed} --save_dir results/gaussian_A2.0_p0.5_L50
done

echo "Submitted 60 jobs (3 configs × 20 seeds)"
