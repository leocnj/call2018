#!/usr/bin/env bash

DATA=../ml_exp/inputs

for TASK in text
do
echo $TASK

TA_17=${DATA}/y17_train_${TASK}.csv
TA_18=${DATA}/y18_train_A_${TASK}.csv
TS_17=${DATA}/y17_test_${TASK}.csv
echo ${TS_17}
MODEL=TPOT
#
# using debug version to skip many CVs.
python runexp_pair_debug.py --train ${TA_18} ${TA_17} --test ${TS_17} -t ${MODEL} --fit
done