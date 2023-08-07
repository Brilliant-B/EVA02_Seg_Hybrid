#!/usr/bin/env bash

GPU=0
NAME="eva2_segmenter"
WORK_DIR="workbench/${NAME}/train/"
RESUME="workbench/${NAME}/train/finetune_1/iter_14000.pth"
LOAD=""

python train.py --seed 0 --deterministic --gpu-ids ${GPU} \
 --work-dir ${WORK_DIR} \
#  --resume-from ${RESUME} \
#  --no-validate
#  --load-from ${LOAD}
 
