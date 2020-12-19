#!/bin/bash

 ./evaluate-mibi.sh delay early
 ./evaluate-mibi.sh delay delay
 ./evaluate-mibi.sh delay both

 ./evaluate-mibi.sh early early
 ./evaluate-mibi.sh early delay
 ./evaluate-mibi.sh early both

 ./evaluate-mibi.sh both early
 ./evaluate-mibi.sh both delay
 ./evaluate-mibi.sh both both
