#!/usr/bin/env bash

GPUS=4
CONFIGS="configs/eva2_hybrid/Segmenter_EVA02_large_24_512_slide_80k.py"

python -m torch.distributed.launch --nproc_per_node=${GPUS} \
    --use_env train.py --launcher pytorch \
    ${CONFIGS} --seed 0 --deterministic --gpus ${GPUS} \
    # --load-from workbench/iter_60000.pth