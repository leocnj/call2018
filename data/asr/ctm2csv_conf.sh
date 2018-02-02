#!/usr/bin/env bash

D=with_confidence
echo $D

# ctm already contains Id column, not need to get
python ../../src/parse_ctm.py --ctm raw/align-conf.ctm  --header --no-getId --conf > ${D}/2017_test_asr.csv
python ../../src/parse_ctm.py --ctm raw/ed2_train_cv.ctm  --header --conf> ${D}/2018_train_asr.csv

# using ed2_train_cv.ctm instead of individual ctm
# python ../../src/parse_ctm.py --ctm raw/00-align-conf-20-0.5.ctm  --header > 2018_train_asr.csv
# python ../../src/parse_ctm.py --ctm raw/01-align-conf-20-0.5.ctm  >> 2018_train_asr.csv
# python ../../src/parse_ctm.py --ctm raw/02-align-conf-20-0.5.ctm  >> 2018_train_asr.csv
# python ../../src/parse_ctm.py --ctm raw/03-align-conf-20-0.5.ctm  >> 2018_train_asr.csv
# python ../../src/parse_ctm.py --ctm raw/04-align-conf-20-0.5.ctm  >> 2018_train_asr.csv

# WERs are:
#
# Training set:
#
# 0) WER: 13.46
# 1) WER:  9.32
# 2) WER: 26.80
# 3) WER:  11.68
# 4) WER: 12.80
# Weighted Average WER: 12.4%
#
# WER for test set is: 13.64%

# ST2 train 6698 utterances  ASR contains 6090
# ST1 test   995 utterances  ASR contains 995


# 1/28/2018
# 2017_train 2-fold CV
#
# And the WERS are:  17.85%, and  11.80%
#
# A pretty wide variation

python ../../src/parse_ctm.py --ctm raw/ed1_train_cv.ctm  --header --no-getId --conf > ${D}/2017_train_asr.csv