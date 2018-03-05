#!/usr/bin/env bash

DATA=../ml_exp/inputs
PIPE=../ml_exp/pipe

MODEL=Ensemble

# 3 2017 + ABC, text, 0.5
THRES=0.5
TASK=text
TASK2=text

TA_17=${DATA}/y17_train_${TASK}.csv
TA_18_A=${DATA}/y18_train_A_${TASK}.csv
TA_18_B=${DATA}/y18_train_B_${TASK}.csv
TA_18_C=${DATA}/y18_train_C_${TASK}.csv
TA_18="$TA_18_A $TA_18_B $TA_18_C"
echo ${TA_18}

TS_18=${DATA}/y18_test_${TASK2}.csv
MODEL_FILE=${PIPE}/y18_train_A_${TASK}-y18_train_B_${TASK}-y18_train_C_${TASK}-y17_train_${TASK}-${MODEL}.pkl

echo $MODEL_FILE
python runexp_train.py --train ${TA_18} ${TA_17} -t ${MODEL} --fit
python runexp_eval.py --model_file $MODEL_FILE --test ${TS_18} --year 2018_text --thres $THRES