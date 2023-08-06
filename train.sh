#!/usr/bin/env bash

GPU=0
NAME="eva2_segmenter"
CONFIGS="configs/Hyper_Control.py"
WORK_DIR="workbench/${NAME}/train/"
RESUME="workbench/${NAME}/train/finetune_1/iter_20000.pth"
LOAD=""

python train.py --config ${CONFIGS} --seed 0 --deterministic --gpu-ids ${GPU} --eval \
 --work-dir ${WORK_DIR} \
#  --resume-from ${RESUME} \
#  --no-validate
#  --load-from ${LOAD}
 
