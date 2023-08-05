#!/usr/bin/env bash

GPU=0
CONFIGS="configs/eva2_hybrid/Segmenter_EVA02_large_24_512_slide_80k.py"
WORK_DIR="workbench/eva2_segmenter/"
LOAD=""
RESUME=""

python train.py --config ${CONFIGS} --seed 0 --deterministic --gpu-ids ${GPU} \
 --work-dir ${WORK_DIR} \
#  --load-from ${LOAD}
