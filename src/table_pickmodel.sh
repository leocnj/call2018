#!/usr/bin/env bash

DATA=../ml_exp/inputs
PIPE=../ml_exp/pipe
PRED=../ml_exp/preds
RUN_PY=1


# 2017 text task
# compare five models.
one_row () {
    MODEL=$1
    THRES=$2
    TASK=$3
    TASK2=$4
    echo "************************************"

    TA_17=${DATA}/y17_train_${TASK}.csv
    TS_17=${DATA}/y17_test_${TASK2}.csv

    MODEL_FILE=${PIPE}/y17_train_${TASK}-${MODEL}.pkl
    # 2017 test
    CSV=${PRED}/y17_train_${TASK}-${MODEL}_y17_test_${TASK2}_t${THRES}.csv
    if [ $RUN_PY -eq 1 ]; then
      python runexp_train.py --train ${TA_17} -t ${MODEL} --fit
      python runexp_eval.py --model_file $MODEL_FILE --test ${TS_17} --year 2017 --thres $THRES
      python comp_result.py $CSV --year 2017
    fi
   echo "************************************"
}

for MODEL in LR SVC RF XGB Ensemble
do
    echo $MODEL
    one_row $MODEL 0.5 asr asr
    one_row $MODEL 0.35 asr asr
done
