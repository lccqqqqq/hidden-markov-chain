#!/bin/bash
# Submit Bayes-optimal gap analysis at k=16 with 50k samples
# CPU job, ~hours runtime

mkdir -p out/bayes_optimal_k16

addqueue -q long \
    -o out/bayes_optimal_k16/bayes_optimal_k16_%j.out \
    -s -m 5 \
    /usr/bin/python3 src/reverse_hmm_analysis.py \
        --num_samples 50000 \
        --max_context 16 \
        --output out/bayes_optimal_k16/reverse_hmm_analysis_k16.json
