addqueue -q short -n 6 -m 5 \
    -o out/iterative_probe_checkpoints/probe_ckpt_%j.out \
    /usr/bin/python3 src/iterative_probe_multi_checkpoint.py \
        --model-dir models/mess3/20260306_150920_L4_d64_H1_full_LN/ \
        --seq-length 10 --batch-size 10000 \
        --max-iterations 15 --r2-threshold 0.01 \
        --device cpu --use-mpi --force
