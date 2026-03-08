addqueue -q gpulong -m 8 \
    -o out/dense_checkpoint_training/train_%j.out \
    /usr/bin/python3 src/train.py --config config/mess3_dense_checkpoints.yaml
