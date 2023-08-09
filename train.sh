#!/usr/bin/env bash

GPU=0
NAME="eva2_segmenter"
WORK_DIR="/hy-tmp/workbench/${NAME}/train/"
RESUME=""
LOAD="/hy-tmp/workbench/eva2_segmenter/train/neck_linear_finetune_0/100.pth"

python train.py --seed 0 --deterministic --gpu-ids ${GPU} \
 --work-dir ${WORK_DIR} \
# --resume-from ${RESUME} \
# --load-from ${LOAD}
# --no-validate

