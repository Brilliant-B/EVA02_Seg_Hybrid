#!/usr/bin/env bash

GPU=0
NAME="eva2_segmenter"
CONFIGS="configs/eva2_hybrid/Segmenter_EVA02_large_24_512_slide_80k.py"
CKPT="workbench/eva2_segmenter/train/finetune_1/iter_20000.pth"
WORK_DIR="workbench/${NAME}/eval/"
SHOW_DIR="workbench/${NAME}/eval/inference/"

python test.py --seed 0 --deterministic --gpu-ids ${GPU} \
 --config ${CONFIGS} \
 --checkpoint ${CKPT} \
 --work-dir ${WORK_DIR} \
 --show_dir ${SHOW_DIR} \
#  --no-validate
#  --load-from ${LOAD}
 
