#!/bin/bash

SCORE_THRESHOLD=-1.0
IOU_THRESHOLD=0.3

 ./evaluate-mibi.sh delay delay ${SCORE_THRESHOLD} ${IOU_THRESHOLD}
 ./evaluate-mibi.sh early early ${SCORE_THRESHOLD} ${IOU_THRESHOLD}
