#!/bin/bash

# export CUDA_VISIBLE_DEVICES=1

MODEL_ID=$1 # delay|early
DATA_ID=$2  # delay|early
SCORE_THRESHOLD=$3 # -1.0
IOU_THRESHOLD=$4 # 0.3

PATH_PREFIX="./train-mibi-${MODEL_ID}-with-negative/evaluate/"
mkdir $PATH_PREFIX

MODEL="./train-mibi-${MODEL_ID}-with-negative/${MODEL_ID}-best-mAP.h5"
TEST_NAME="${MODEL_ID}model-${DATA_ID}data"
DATASET_DIR_TEST="../pascal-mibi-${DATA_ID}-dataset-test-all"
DATASET_DIR_VAL="../pascal-mibi-${DATA_ID}-dataset-notest-with-negative"

# evaluate with validation dataset for FROC
keras_retinanet/bin/evaluate.py --convert-model \
 --mode FROC \
 --iou-threshold ${IOU_THRESHOLD} \
 --score-threshold ${SCORE_THRESHOLD} \
 --max-detections 1000 \
 pascal-mibi --evaluate-target-split val \
 $DATASET_DIR_VAL $MODEL | tee ${PATH_PREFIX}evaluate-${TEST_NAME}-val-froc-iou${IOU_THRESHOLD}-score${SCORE_THRESHOLD}.log

# evaluate with test dataset for mAP
keras_retinanet/bin/evaluate.py --convert-model \
 --mode mAP \
 --iou-threshold ${IOU_THRESHOLD} \
 --score-threshold ${SCORE_THRESHOLD} \
 --max-detections 1000 \
 pascal-mibi --evaluate-target-split test_all \
 $DATASET_DIR_TEST $MODEL | tee ${PATH_PREFIX}evaluate-${TEST_NAME}-test-all-map-iou${IOU_THRESHOLD}-score${SCORE_THRESHOLD}.log

# evaluate with test dataset for FROC
keras_retinanet/bin/evaluate.py --convert-model \
 --mode FROC \
 --iou-threshold ${IOU_THRESHOLD} \
 --score-threshold ${SCORE_THRESHOLD} \
 --max-detections 1000 \
 pascal-mibi --evaluate-target-split test_all \
 $DATASET_DIR_TEST $MODEL | tee ${PATH_PREFIX}evaluate-${TEST_NAME}-test-all-froc-iou${IOU_THRESHOLD}-score${SCORE_THRESHOLD}.log

# evaluate with test dataset for binary test
keras_retinanet/bin/evaluate.py --convert-model \
 --mode TEST \
 --iou-threshold ${IOU_THRESHOLD} \
 --score-threshold 0.8 \
 pascal-mibi --evaluate-target-split test_all \
 $DATASET_DIR_TEST $MODEL | tee ${PATH_PREFIX}evaluate-${TEST_NAME}-test-all-test-iou${IOU_THRESHOLD}-score0.8.log

# create image output for test dataset with scores
keras_retinanet/bin/evaluate.py --convert-model \
 --mode SCORE \
 --iou-threshold ${IOU_THRESHOLD} \
 --score-threshold ${SCORE_THRESHOLD} \
 --save-path ${PATH_PREFIX}evaluate-mibi-${TEST_NAME}-output-test_all-nost \
 pascal-mibi --evaluate-target-split test_all \
 $DATASET_DIR_TEST $MODEL | tee ${PATH_PREFIX}evaluate-${TEST_NAME}-score-test_all-nost-iou${IOU_THRESHOLD}-score${SCORE_THRESHOLD}.log

# create image output for validation dataset with scores
keras_retinanet/bin/evaluate.py --convert-model \
 --mode SCORE \
 --iou-threshold ${IOU_THRESHOLD} \
 --score-threshold ${SCORE_THRESHOLD} \
 --save-path ${PATH_PREFIX}evaluate-mibi-${TEST_NAME}-output-val-nost \
 pascal-mibi --evaluate-target-split val \
 $DATASET_DIR_VAL $MODEL | tee ${PATH_PREFIX}evaluate-${TEST_NAME}-score-val-nost-iou${IOU_THRESHOLD}-score${SCORE_THRESHOLD}.log
