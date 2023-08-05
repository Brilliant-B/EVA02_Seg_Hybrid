#!/usr/bin/env bash

CONFIG="configs/eva2_hybrid/Segmenter_EVA02_large_24_512_slide_80k.py"
GPUS=4

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=${GPUS} --master_port=-29500 \
    $(dirname "$0")/train.py ${CONFIG} --launcher pytorch ${@:3}
