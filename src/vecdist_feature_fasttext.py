import fastText  # use FB's official python binding
import pickle
import numpy as np
from scipy import spatial
import csv
import pandas as pd
from multiprocessing import cpu_count, Pool

# from gensim.models import Word2Vec, FastText         # Gensim
from gensim.models.wrappers.fasttext import FastText   # C++

from utils import get_logger
# logger = get_logger(__name__, simple=True)

# load grammar
with open('../data/processed/data.pkl', 'rb') as pf:
    objs = pickle.load(pf)

grammar_dic = objs[0]
df_17_train = objs[1]
df_17_test = objs[2]
df_18_A_train = objs[3]
df_18_B_train = objs[4]
df_18_C_train = objs[5]

# load fasttext word2vec model
model_ft = fastText.load_model('../data/fastText/model/wiki.simple.bin')
model = FastText.load_fasttext_format('../data/fastText/model/wiki.simple')

def cossim_np(a, b):
    return 1 - spatial.distance.cosine(a, b)

def get_sent_vec(sent):
    return get_sent_vec_gs(sent)

# can support OOV word output.
def get_sent_vec_ft(sent):
    return model_ft.get_sentence_vector(sent)

# model[word] will raise error if word is not in vocab
# have to use words in the vocab
def get_sent_vec_gs(sent):
    vecs = [model[word] for word in sent.split() if word in model.wv.vocab]
    return np.mean(vecs, axis=0)


def cossim_feature(sent, sents_gr):
    vec_dists = [cossim_np(get_sent_vec(sent), get_sent_vec(sent_gr))
                 for sent_gr in sents_gr]
    return np.mean(vec_dists), np.max(vec_dists)

# word mover distance (WMD)
def wmd_feature(sent, sents_gr):
    vec_dists = [model.wmdistance(sent, sent_gr) for sent_gr in sents_gr]
    return np.mean(vec_dists), np.min(vec_dists)

from tqdm import tqdm

def extract_vec_features(df_in):
    feats = []
    # for idx in tqdm(range(len(df_in.index))):   # move tqdm outside in MP condition
    #     row = df_in.iloc[idx,:]
    for _, row in tqdm(df_in.iterrows()):
        id = row['Id']
        # !!! always use RecResult
        sent = row['RecResult'].rstrip()
        sents_in_grammar = grammar_dic.get(row['Prompt'])
        if sents_in_grammar is not None:
            cos_mean, cos_max = cossim_feature(sent, sents_in_grammar)
            wmd_mean, wmd_min = wmd_feature(sent, sents_in_grammar)
            # logger.info('{}: {} {} {} {}'.format(sent, cos_mean, cos_max, wmd_mean, wmd_min))
            feats.append([id, cos_mean, cos_max, wmd_mean, wmd_min])
    return pd.DataFrame.from_records(feats,
                                     columns=['Id', 'cos_mean', 'cos_max', 'wmd_mean', 'wmd_min'])


def output(feats, out_csv):
    with open(out_csv, 'w', newline='') as op:
        feats.to_csv(op, index=False)


def parallel(data, func):
    data_split = np.array_split(data, cores)
    pool = Pool(cores)
    data = pd.concat(pool.imap_unordered(func, data_split))
    pool.close()
    pool.join()
    return data

import parmap

def parallel2(data, func):
    data_split = np.array_split(data, cores)
    # pool = Pool(cores)
    # data = pd.concat(pool.imap_unordered(func, data_split))
    data = pd.concat(parmap.map(func, data_split, pm_pbar=True))
    # pool.close()
    # pool.join()
    return data



def test():
    a = ['a room for four days',
         'I want to pay by card',
         'I came from a small village']
    b = ['a room four two nights',
         'can I pay by my credit card',
         'toy story two is boring']

    for x, y in zip(a,b):
        cossim = cossim_np(get_sent_vec(x), get_sent_vec(y))
        wmd = model.wmdistance(x, y)
        print('{} vs {}: {} {}'.format(x, y, cossim, wmd))

if __name__ == '__main__':

    cores = cpu_count()

    # test()
    # extract_vec_features(df_17_train, '../data/processed/df17_train_fasttext.csv')
    # 1/17/2018 re-run after getting 2017 test set's ASR outputs.
    # feats = extract_vec_features(df_17_test)
    feats = parallel2(df_17_test, extract_vec_features)  # multi-processor
    output(feats, '../data/processed/df17_test_fasttext.csv')
    # extract_vec_features(df_18_A_train, '../data/processed/df18_A_train_fasttext.csv')
    # extract_vec_features(df_18_B_train, '../data/processed/df18_B_train_fasttext.csv')
    # extract_vec_features(df_18_C_train, '../data/processed/df18_C_train_fasttext.csv')
