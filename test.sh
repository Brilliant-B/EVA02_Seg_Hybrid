#!/usr/bin/env bash

GPU=0
NAME="eva2_segmenter"
CONFIGS="configs/linear_neck.py"
CKPT="/hy-tmp/workbench/eva2_segmenter/train/neck_linear_finetune_0/100.pth"
WORK_DIR="/hy-tmp/workbench/${NAME}/eval/"
SHOW_DIR="/hy-tmp/workbench/${NAME}/eval/inference/"

python test.py --eval mIoU --gpu-id ${GPU} \
 --config ${CONFIGS} \
 --checkpoint ${CKPT} \
 --work-dir ${WORK_DIR} \
 --show-dir ${SHOW_DIR} \
#  --no-validate
#  --load-from ${LOAD}
