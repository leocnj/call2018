#!/usr/bin/env bash

# ctm already contains Id column, not need to get
python ../../src/parse_ctm.py --ctm raw/align-conf.ctm  --header --no-getId > 2017_test_asr.csv

python ../../src/parse_ctm.py --ctm raw/00-align-conf-20-0.5.ctm  --header > 2018_train_asr.csv
python ../../src/parse_ctm.py --ctm raw/01-align-conf-20-0.5.ctm  >> 2018_train_asr.csv
python ../../src/parse_ctm.py --ctm raw/02-align-conf-20-0.5.ctm  >> 2018_train_asr.csv
python ../../src/parse_ctm.py --ctm raw/03-align-conf-20-0.5.ctm  >> 2018_train_asr.csv
python ../../src/parse_ctm.py --ctm raw/04-align-conf-20-0.5.ctm  >> 2018_train_asr.csv

#WERs are:
#
#Training set:
#
#0) WER: 13.46
#1) WER:  9.32
#2) WER: 26.80
#3)WER:  11.68
#4) WER: 12.80
#Weighted Average WER: 12.4
#
#Wer for test set is: 13.64%

# ST2 train 6698 utterances  ASR contains 6090
# ST1 test   995 utterances  ASR contains 995
