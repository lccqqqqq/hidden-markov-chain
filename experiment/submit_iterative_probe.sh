addqueue -q short \
    -o out/iterative_probe/iterative_probe_%j.out \
    -s -m 5 \
    /usr/bin/python3 src/run_iterative_probe.py \
        --model-dir models/mess3/20260306_150920_L4_d64_H1_full_LN/ \
        --seq-length 10 --batch-size 10000 \
        --max-iterations 32 --r2-threshold -1 \
        --device cpu --force
