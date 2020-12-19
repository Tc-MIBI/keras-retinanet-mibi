#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

MODEL_ID=$1 # both|delay|early
DATA_ID=$2  # both|delay|early
PATH_PREFIX="./evaluate-mibi/"

MODEL="./snapshots/${MODEL_ID}-best-performing.h5"
TEST_NAME="${MODEL_ID}model-${DATA_ID}data"
DATASET_DIR_TEST="../pascal-mibi-${DATA_ID}-dataset-test-all"
DATASET_DIR_VAL="../pascal-mibi-${DATA_ID}-dataset-notest"

mkdir $PATH_PREFIX

# evaluate with validation dataset for FROC @IoU=0.1
keras_retinanet/bin/evaluate.py --convert-model \
 --mode FROC \
 --iou-threshold 0.1 \
 --score-threshold -1.0 \
 --max-detections 1000 \
 pascal-mibi --evaluate-target-split val \
 $DATASET_DIR_VAL $MODEL | tee ${PATH_PREFIX}evaluate-mibi-${TEST_NAME}-val-froc-iou0.1.log

# evaluate with validation dataset for FROC @IoU=0.3
keras_retinanet/bin/evaluate.py --convert-model \
 --mode FROC \
 --iou-threshold 0.3 \
 --score-threshold -1.0 \
 --max-detections 1000 \
 pascal-mibi --evaluate-target-split val \
 $DATASET_DIR_VAL $MODEL | tee ${PATH_PREFIX}evaluate-mibi-${TEST_NAME}-val-froc-iou0.3.log

# evaluate with test dataset for mAP @IoU=0.1
keras_retinanet/bin/evaluate.py --convert-model \
 --mode mAP \
 --iou-threshold 0.1 \
 --score-threshold -1.0 \
 --max-detections 1000 \
 pascal-mibi --evaluate-target-split test_all \
 $DATASET_DIR_TEST $MODEL | tee ${PATH_PREFIX}evaluate-mibi-${TEST_NAME}-test-all-map-iou0.1.log

# evaluate with test dataset for mAP @IoU=0.3
keras_retinanet/bin/evaluate.py --convert-model \
 --mode mAP \
 --iou-threshold 0.3 \
 --score-threshold -1.0 \
 --max-detections 1000 \
 pascal-mibi --evaluate-target-split test_all \
 $DATASET_DIR_TEST $MODEL | tee ${PATH_PREFIX}evaluate-mibi-${TEST_NAME}-test-all-map-iou0.3.log

# evaluate with test dataset for FROC @IoU=0.1
keras_retinanet/bin/evaluate.py --convert-model \
 --mode FROC \
 --iou-threshold 0.1 \
 --score-threshold -1.0 \
 --max-detections 1000 \
 pascal-mibi --evaluate-target-split test_all \
 $DATASET_DIR_TEST $MODEL | tee ${PATH_PREFIX}evaluate-mibi-${TEST_NAME}-test-all-froc-iou0.1.log

# evaluate with test dataset for FROC @IoU=0.3
keras_retinanet/bin/evaluate.py --convert-model \
 --mode FROC \
 --iou-threshold 0.3 \
 --score-threshold -1.0 \
 --max-detections 1000 \
 pascal-mibi --evaluate-target-split test_all \
 $DATASET_DIR_TEST $MODEL | tee ${PATH_PREFIX}evaluate-mibi-${TEST_NAME}-test-all-froc-iou0.3.log

# evaluate with test dataset for binary test
keras_retinanet/bin/evaluate.py --convert-model \
 --mode TEST \
 --iou-threshold 0.3 \
 --score-threshold 0.8 \
 pascal-mibi --evaluate-target-split test_all \
 $DATASET_DIR_TEST $MODEL | tee ${PATH_PREFIX}evaluate-mibi-${TEST_NAME}-test-all-test.log

# create image output for test dataset with scores
keras_retinanet/bin/evaluate.py --convert-model \
 --mode SCORE \
 --iou-threshold 0.3 \
 --score-threshold -1.0 \
 --save-path ${PATH_PREFIX}evaluate-mibi-${TEST_NAME}-output-test_all-nost \
 pascal-mibi --evaluate-target-split test_all \
 $DATASET_DIR_TEST $MODEL | tee ${PATH_PREFIX}evaluate-mibi-${TEST_NAME}-score-test_all-nost.log

# create image output for validation dataset with scores
keras_retinanet/bin/evaluate.py --convert-model \
 --mode SCORE \
 --iou-threshold 0.3 \
 --score-threshold -1.0 \
 --save-path ${PATH_PREFIX}evaluate-mibi-${TEST_NAME}-output-val-nost \
 pascal-mibi --evaluate-target-split val \
 $DATASET_DIR_VAL $MODEL | tee ${PATH_PREFIX}evaluate-mibi-${TEST_NAME}-score-val-nost.log
