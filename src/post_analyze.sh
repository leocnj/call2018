#!/usr/bin/env bash

DATA=../ml_exp/inputs
PIPE=../ml_exp/pipe
PRED=../ml_exp/preds
MODEL=Ensemble

# 3 2017 + ABC, text, 0.5
EXPM="Expm 3: 2017 + 2018ABC, Ensemble, Text, 0.5"
echo $EXPM
RUN_PY=0

THRES=0.5
TASK=text
TASK2=text
TA_17=${DATA}/y17_train_${TASK}.csv
TA_18_A=${DATA}/y18_train_A_${TASK}.csv
TA_18_B=${DATA}/y18_train_B_${TASK}.csv
TA_18_C=${DATA}/y18_train_C_${TASK}.csv
TA_18="$TA_18_A $TA_18_B $TA_18_C"
TS_18=${DATA}/y18_test_${TASK2}.csv

MODEL_FILE=${PIPE}/y18_train_A_${TASK}-y18_train_B_${TASK}-y18_train_C_${TASK}-y17_train_${TASK}-${MODEL}.pkl
CSV=${PRED}/y18_train_A_${TASK}-y18_train_B_${TASK}-y18_train_C_${TASK}-y17_train_${TASK}-${MODEL}_y18_test_${TASK2}_t${THRES}.csv
if [ $RUN_PY -eq 1 ]; then
  python runexp_train.py --train ${TA_18} ${TA_17} -t ${MODEL} --fit
  python runexp_eval.py --model_file $MODEL_FILE --test ${TS_18} --year 2018_text --thres $THRES
  python comp_result.py $CSV
fi