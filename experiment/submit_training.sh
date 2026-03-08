GPUS=1
GPU_QUEUE=heraclesgpu
# GPU_TYPE=a100with80gb
NCORES=2
MEM_PER_CORE_GB=5

addqueue -q $GPU_QUEUE --gpus $GPUS \
    -o out/mess3_new/mess3_new_%j.out \
    -s -n $NCORES -m $MEM_PER_CORE_GB \
    /usr/bin/python3 src/train.py --config config/base_config_mess3.yaml
