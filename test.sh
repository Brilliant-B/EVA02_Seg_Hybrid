#!/usr/bin/env bash

GPU=0
NAME="eva2_segmenter"
CONFIGS="configs/Hyper_Control.py"
CKPT="workbench/eva2_segmenter/train/finetune_1/iter_14000.pth"
WORK_DIR="workbench/${NAME}/eval/"
SHOW_DIR="workbench/${NAME}/eval/inference/"

python test.py --eval mIoU --gpu-id ${GPU} \
 --config ${CONFIGS} \
 --checkpoint ${CKPT} \
 --work-dir ${WORK_DIR} \
 --show-dir ${SHOW_DIR} \
#  --no-validate
#  --load-from ${LOAD}
 
