import pickle
import numpy as np
from utils import *
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import optunity
import optunity.metrics
"""
1/19/2018   using 40 features, default SVC (w/ RBF kernel) can generate D about 4
            after adding D tuning, when choose prob_cutoff as 0.28134, D can be 5.15


"""

label_back = lambda x: 'correct' if x==1 else 'incorrect'

def get_D_based_thresholds(LANG_T, lang_prob, y, print=False):
    scores = init_scores()
    for i, p_lang in enumerate(lang_prob):
        if p_lang[1] >= LANG_T:
            decision = 'accept'
        else:
            decision = 'reject'
        score_decision(decision, label_back(get_langauge_y(y)[i]), label_back(get_meaning_y(y)[i]), scores)
    if print:
        print_scores(scores)
    return get_D(scores)

MEANING_DIM = 24
def get_meaning_X(X):
    return X[:,0:MEANING_DIM]

def get_meaning_y(y):
    return y[:,0]

def get_langauge_X(X):
    return X[:, MEANING_DIM:]  # using -1 will lose the last column.

def get_langauge_y(y):
    return y[:,1]


if __name__ == '__main__':

 # year17 data
    with open('../data/processed/numpy/year17_withHuy.pkl', 'rb') as pf:
        objs = pickle.load(pf)
        year17_lang_train_X = objs[0]
        year17_lang_train_y = objs[1]
        year17_lang_test_X = objs[2]
        year17_lang_test_y = objs[3]

        year17_meaning_train_X = objs[4]
        year17_meaning_train_y = objs[5]
        year17_meaning_test_X = objs[6]
        year17_meaning_test_y = objs[7]

        year17_train_X = np.concatenate((year17_meaning_train_X, year17_lang_train_X), axis=1)
        year17_train_y = np.column_stack((year17_meaning_train_y, year17_lang_train_y))
        year17_test_X = np.concatenate((year17_meaning_test_X, year17_lang_test_X), axis=1)
        year17_test_y = np.column_stack((year17_meaning_test_y, year17_lang_test_y))


    logger = get_logger(__name__, simple=True)


    @optunity.cross_validated(x=year17_train_X, y=year17_train_y, num_folds=5)
    def tune_Dscore(x_train, y_train, x_test, y_test, L_thres=0.5):
        model_lang = SVC(probability=True)
        model_lang.fit(get_langauge_X(x_train), get_langauge_y(y_train))
        D_score = get_D_based_thresholds(L_thres,
                                         model_lang.predict_proba(get_langauge_X(x_test)),
                                         y_test)
        return D_score

    @timeit
    # this can use optunity's multi-thread and smarter solver
    def tune_D_smart():
        thres_grid = {'L_thres': [0.25, 0.35]}
        optm_thres, info, _ = optunity.maximize_structured(tune_Dscore,
                                                           num_evals=20,
                                                           pmap=optunity.pmap,
                                                           search_space=thres_grid)
        print("Optimal parameters" + str(optm_thres))
        print("best D after CV running: %2.3f" % info.optimum)
        return (optm_thres['L_thres'])

    best_L = tune_D_smart()
    #
    # 1/19/2018
    #
    # best_L 0.5  D = 4.04
    # best_L 0.3  D = 4.58
    # It looks that we need tune best_L
    # best_L = 0.3
    #
    # when tuning D on L_thres [0.25, 0.75] for 50 times, we get L_thres = 0.251273
    # D can reach 5.76, however, cRj is 0.04
    # After focusing on [0.25, 0.35] for 20 times, get D 5.15 with a cRj 0.05
    #

    model_lang = SVC(probability=True)
    model_lang.fit(year17_lang_train_X, year17_lang_train_y)
    probs_l = model_lang.predict_proba(year17_lang_test_X)

    y_true, y_pred = year17_lang_test_y, model_lang.predict(year17_lang_test_X)

    print(accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))

    D_score = get_D_based_thresholds(best_L,
                                     probs_l,
                                     year17_test_y, True)
