#!/usr/bin/env bash

DATA=../ml_exp/inputs
PIPE=../ml_exp/pipe
PRED=../ml_exp/preds
MODEL=Ensemble
RUN_PY=1

one_row () {
    TASK=$1
    TASK2=$2
    THRES=$3
    echo "************************************"
    echo $THRES

    TA_17=${DATA}/y17_train_${TASK}.csv
    TA_18_A=${DATA}/y18_train_A_${TASK}.csv
    TA_18="$TA_18_A"
    TS_17=${DATA}/y17_test_${TASK2}.csv
    TS_18=${DATA}/y18_test_${TASK2}.csv

    MODEL_FILE=${PIPE}/y18_train_A_${TASK}-y17_train_${TASK}-${MODEL}.pkl
    # 2017 test
    CSV=${PRED}/y18_train_A_${TASK}-y17_train_${TASK}-${MODEL}_y17_test_${TASK2}_t${THRES}.csv
    if [ $RUN_PY -eq 1 ]; then
      python runexp_train.py --train ${TA_18} ${TA_17} -t ${MODEL} --fit
      python runexp_eval.py --model_file $MODEL_FILE --test ${TS_17} --year 2017 --thres $THRES
      python comp_result.py $CSV --year 2017
    fi
    # 2018 test
    CSV=${PRED}/y18_train_A_${TASK}-y17_train_${TASK}-${MODEL}_y18_test_${TASK2}_t${THRES}.csv
    if [ $RUN_PY -eq 1 ]; then
      python runexp_train.py --train ${TA_18} ${TA_17} -t ${MODEL} --fit
      python runexp_eval.py --model_file $MODEL_FILE --test ${TS_18} --year 2018_text --thres $THRES
      python comp_result.py $CSV
    fi
    echo "************************************"
}

# Table 3
for THRES in 0.5 0.4 0.35
do
    echo $THRES
    one_row text text $THRES
done

# Table 4
for THRES in 0.5 0.4 0.35
do
    echo $THRES
    one_row asr asr $THRES
done

# Table 5
for THRES in 0.5 0.4 0.35
do
    echo $THRES
    one_row asr 2best $THRES
done
