#!/usr/bin/env bash

DATA=../ml_exp/inputs

MODEL=Ensemble
THRES=$1
echo $MODEL $THRES

# text ask
TASK=text
TASK2=text
echo $TASK
TA_17=${DATA}/y17_train_${TASK}.csv
TA_18=${DATA}/y18_train_A_${TASK}.csv

TS_17=${DATA}/y17_test_${TASK2}.csv
TS_18=${DATA}/y18_test_${TASK2}.csv
MODEL_FILE=../ml_exp/pipe/y18_train_A_${TASK}-y17_train_${TASK}-${MODEL}.pkl

echo $MODEL_FILE
python runexp_train.py --train ${TA_18} ${TA_17} -t ${MODEL} --fit
# show D on y17_test
python runexp_eval.py --model_file $MODEL_FILE --test ${TS_17} --year 2017 --thres $THRES
python runexp_eval.py --model_file $MODEL_FILE --test ${TS_18} --year 2018 --thres $THRES

# asr task
TASK=asr
TASK2=2best
echo $TASK
TA_17=${DATA}/y17_train_${TASK}.csv
TA_18=${DATA}/y18_train_A_${TASK}.csv

TS_17=${DATA}/y17_test_${TASK2}.csv
TS_18=${DATA}/y18_test_${TASK2}.csv
MODEL_FILE=../ml_exp/pipe/y18_train_A_${TASK}-y17_train_${TASK}-${MODEL}.pkl

echo $MODEL_FILE
python runexp_train.py --train ${TA_18} ${TA_17} -t ${MODEL} --fit
# show D on y17_test
python runexp_eval.py --model_file $MODEL_FILE --test ${TS_17} --year 2017 --thres $THRES
python runexp_eval.py --model_file $MODEL_FILE --test ${TS_18} --year 2018 --thres $THRES

# asr task
TASK=asr
TASK2=asr
echo $TASK
TA_17=${DATA}/y17_train_${TASK}.csv
TA_18=${DATA}/y18_train_A_${TASK}.csv

TS_17=${DATA}/y17_test_${TASK2}.csv
TS_18=${DATA}/y18_test_${TASK2}.csv
MODEL_FILE=../ml_exp/pipe/y18_train_A_${TASK}-y17_train_${TASK}-${MODEL}.pkl

echo $MODEL_FILE
python runexp_train.py --train ${TA_18} ${TA_17} -t ${MODEL}
# show D on y17_test
python runexp_eval.py --model_file $MODEL_FILE --test ${TS_17} --year 2017 --thres $THRES
python runexp_eval.py --model_file $MODEL_FILE --test ${TS_18} --year 2018 --thres $THRES
