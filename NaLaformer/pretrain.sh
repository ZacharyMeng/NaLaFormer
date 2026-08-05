#!/bin/bash

# Usage:
#   export DATA=/path/to/imagenet
#   export CUDA_VISIBLE_DEVICES=0,1,2,3
#   bash pretrain.sh

DATA=${DATA:-/path/to/imagenet}
OUTPUT=${OUTPUT:-./output}
NPROC=${NPROC:-8}
PYTHON=${PYTHON:-python}

$PYTHON -m torch.distributed.launch --nproc_per_node=$NPROC --use_env main.py \
    --warmup-epochs 5 \
    --model NALAFORMER_XT \
    --data-path $DATA \
    --num_workers 16 \
    --batch-size 128 \
    --drop-path 0.05 \
    --epochs 300 \
    --model-ema \
    --dist-eval \
    --output_dir $OUTPUT
