#!/usr/bin/env bash

# model: LR, SVC, and Ensemble
DATA=../ml_exp/inputs
TASK1=asr
TASK2=2best

echo $TASK

TA_17=${DATA}/y17_train_${TASK1}.csv
TA_18=${DATA}/y18_train_A_${TASK1}.csv

# test:  y17_test_text.csv
TS_17=${DATA}/y17_test_${TASK2}.csv
echo ${TS_17}
for MODEL in LR SVC Ensemble
do
  echo $MODEL
  python runexp_pair.py --train ${TA_18} ${TA_17} --test ${TS_17} -t ${MODEL} --fit
done
