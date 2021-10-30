#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0,1

DATASET_DIR="../pascal-mibi-early-dataset-notest-with-negative"
OUTPUT_DIR="./train-mibi-early-with-negative"

mkdir ${OUTPUT_DIR}
SNAPSHOP_PATH="./${OUTPUT_DIR}/snapshots"
TENSORBOARD_DIR="./${OUTPUT_DIR}/logs"
ERR_LOG="./${OUTPUT_DIR}/train.err"
STD_LOG="./${OUTPUT_DIR}/train.log"

keras_retinanet/bin/train.py \
 --random-transform \
 --epochs 100 \
 --batch-size 8 \
 --steps 1250 \
 --multi-gpu 2 \
 --multi-gpu-force \
 --multiprocessing \
 --workers 4 \
 --compute-val-loss \
 --image-min-side 800 \
 --image-max-side 1333 \
 --snapshot-path $SNAPSHOP_PATH \
 --tensorboard-dir $TENSORBOARD_DIR \
 pascal-mibi $DATASET_DIR 2>>$ERR_LOG >>$STD_LOG
