import pickle
import numpy as np
import optunity
import optunity.metrics
from utils import *
from train_model import define_model
import pandas as pd

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


# !!! update after adding features.
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


    with open('../data/processed/numpy/year17_models.pkl', 'rb') as pf:
        objs = pickle.load(pf)
        # RF
        param_lang = objs[0]
        param_meaning = objs[1]
        # # XGB
        param_lang = objs[2]
        param_meaning = objs[3]
        # SVM
        param_lang = objs[4]
        param_meaning = objs[5]

    logger = get_logger(__name__, simple=True)


    """
    Step 2:
    
    The following codes are for tuning the thresholds to map probs output by both language and meaning 
    SVM models to decisions (accept vs. reject).
    
    Note that due to D computation doesn't use numpy, its running speed is in fact slow. For example,
    compared with the time used for model training in the Step 1, tuning D uses 3 to 4 times long time.
    """
    @optunity.cross_validated(x=year17_train_X, y=year17_train_y, num_folds=5)
    def tune_Dscore(x_train, y_train, x_test, y_test, L_thres=0.5, M_thres=0.5):
        model_lang = define_model(**param_lang)
        model_meaning = define_model(**param_meaning)
        model_lang.fit(get_langauge_X(x_train), get_langauge_y(y_train))
        model_meaning.fit(get_meaning_X(x_train), get_meaning_y(y_train))
        D_score = get_D_based_thresholds(L_thres,
                                         model_lang.predict_proba(get_langauge_X(x_test)),
                                         y_test)
        return D_score


    @timeit
    # this can use optunity's multi-thread and smarter solver
    def tune_D_smart():
        thres_grid = {'L_thres' :  [0.35, 0.75]}
        optm_thres, info, _ = optunity.maximize_structured(tune_Dscore,
                                                           num_evals=100,
                                                           pmap=optunity.pmap,
                                                           search_space=thres_grid)
        print("Optimal parameters" + str(optm_thres))
        print("best D after CV running: %2.3f" % info.optimum)
        return(optm_thres['L_thres'])

    # best_L = tune_D_smart()
    best_L = 0.5

    """
    Step 3:
    
    Use the optimized models and D-related thresholds to generate predictions on the 2017 test data set.
    """
    model_lang = define_model(**param_lang)
    model_meaning = define_model(**param_meaning)
    model_lang.fit(get_langauge_X(year17_train_X), get_langauge_y(year17_train_y))
    model_meaning.fit(get_meaning_X(year17_train_X), get_meaning_y(year17_train_y))

    probs_l = model_lang.predict_proba(get_langauge_X(year17_test_X))
    probs_m = model_meaning.predict_proba(get_meaning_X(year17_test_X))

    # For analyzing prediction quality
    df_view = pd.DataFrame({'pb_l': probs_l[:, 1],
                        'pb_m': probs_m[:, 1],
                        'label_l': year17_lang_test_y,
                        'label_m': year17_meaning_test_y
                        })
    df_view.to_csv('../data/debug/17test_pred.csv', index=False)

    D_score = get_D_based_thresholds(best_L,
                                     probs_l,
                                     year17_test_y, True)
