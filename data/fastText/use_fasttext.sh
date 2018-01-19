#!/usr/bin/env bash

SRC=./ref_and_df17train.txt
MODEL=df17

# Train a word2vec model
#
# skipgram
# dim 50
#
fasttext skipgram -input ${SRC} -output ${MODEL} -dim 50