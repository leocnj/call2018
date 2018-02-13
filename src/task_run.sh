#!/usr/bin/env bash

# model: LR, SVC, and Ensemble
DATA=../ml_exp/inputs
TASK=$1

echo $TASK

TA_17=${DATA}/y17_train_${TASK}.csv
TA_18=${DATA}/y18_train_A_${TASK}.csv

# test:  y17_test_text.csv
TS_17=${DATA}/y17_test_${TASK}.csv
echo ${TS_17}
for MODEL in LR SVC Ensemble
do
  echo $MODEL
  python runexp_pair.py --train ${TA_18} ${TA_17} --test ${TS_17} -t ${MODEL} --fit
done

# test: y18_train_BC_text
TA_18_B=${DATA}/y18_train_B_${TASK}.csv
TA_18_C=${DATA}/y18_train_C_${TASK}.csv
echo ${TA_18_B} ${TA_18_C}
for MODEL in LR SVC Ensemble
do
  echo $MODEL
  python runexp_pair.py --train ${TA_18} ${TA_17} --test ${TA_18_B} ${TA_18_C} -t ${MODEL} --fit
done

# xval
python runexp_xval.py --train ${TA_17} ${TA_18} ${TA_18_B} ${TA_18_C} -t Ensemble --onlyA