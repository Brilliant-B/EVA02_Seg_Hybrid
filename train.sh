#!/usr/bin/env bash

GPU=0
NAME="eva2_segmenter"
WORK_DIR="/hy-tmp/workbench/${NAME}/train/"
RESUME=""
LOAD=""

python train.py --seed 0 --deterministic --gpu-ids ${GPU} \
 --work-dir ${WORK_DIR} \
#  --resume-from ${RESUME} \
#  --no-validate
#  --load-from ${LOAD}
