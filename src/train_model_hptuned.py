#
# train_model_hptuned.py
#
# - Tune hyper parameters (hp)
#
#
#


import pickle
import argparse

import numpy as np
from hyperopt import fmin, tpe, Trials
from hyperopt import hp
from hyperopt import space_eval
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from utils import *

SEED = 1


def define_model(**params):
    if params['model'] == 'RF':
        n_estimators = params['n_estimators']
        max_features = params['max_features']
        model = RandomForestClassifier(n_estimators=n_estimators,
                                       max_features=max_features, random_state=SEED)
    elif params['model'] == 'XGB':
        n_estimators = params['n_estimators']
        max_depth = params['max_depth']
        model = XGBClassifier(n_estimators=n_estimators,
                              max_depth=max_depth)
    elif params['model'] == 'SVM':
        C = params['C']
        kernel = params['kernel']
        model = SVC(C=C, kernel=kernel, probability=True, random_state=SEED)
    elif params['model'] == 'LR':
        C = params['C']
        model = LogisticRegression(C=C, random_state=SEED)
    elif params['model'] == 'kNN':
        k = params['k']
        model = KNeighborsClassifier(n_neighbors=k)
    else:
        pass
    return model


def show_trials(trials, space):
    """
    display hyperopt running details
    :param trials:
    :param space:
    :return:
    """
    for trial in trials.trials:
        params = trial['misc']['vals']
        dict_ = {}
        for k, v in params.items():
            dict_[k] = v[0]  # convert list to int to use space_eval
        params = space_eval(space, dict_)
        score = 1 - trial['result']['loss']
        print('{} => {}'.format(params, score))


@timeit
def find_optm_params(objective, space, max_evals=20):
    trials = Trials()
    best = fmin(objective,
                space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)
    best_params = space_eval(space, best)
    show_trials(trials, space)
    return best_params


if __name__ == '__main__':

    with open('../data/processed/numpy/year17_withHuy.pkl', 'rb') as pf:
        objs = pickle.load(pf)
        lang_train_X = objs[0]
        lang_train_y = objs[1]
        lang_test_X = objs[2]
        lang_test_y = objs[3]

        meaning_train_y = objs[4]
        meaning_test_y = objs[5]

        train_y = np.column_stack((meaning_train_y, lang_train_y))
        test_y = np.column_stack((meaning_test_y, lang_test_y))

    # skip lots of sklearn depreciation warnings
    import warnings

    warnings.filterwarnings("ignore")

    logger = get_logger(__name__, simple=True)

    # to use same CV data splitting
    shuffle = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    scoring_metric = 'roc_auc'


    # define inside main to easily access data
    def D_objective(params):
        model = define_model(**params)
        cv_score, _, _ = cross_val_D(model,
                                     lang_train_X, train_y,
                                     cv=shuffle)
        return 1 - cv_score.mean()


    def ML_objective(params):
        model = define_model(**params)
        cv_score = cross_val_score(model,
                                   lang_train_X, lang_train_y,
                                   cv=shuffle,
                                   scoring=scoring_metric, n_jobs=-1)
        return 1 - cv_score.mean()


    shuffle = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    shuffle_inEval = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED + 1024)


    def cv_acc(model):
        cv_score = cross_val_score(model,
                                   lang_train_X, lang_train_y,
                                   cv=shuffle_inEval,
                                   scoring='accuracy', n_jobs=-1)
        acc_test = accuracy_score(lang_test_y, model.predict(lang_test_X))
        print('Acc mean on train: {:2.4f}\tAcc on test: {:2.4f}'.format(cv_score.mean(), acc_test))


    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--model_type', type=str, help='model type',
                        required=True)
    args = parser.parse_args()
    model_type = args.model_type
    print('model type: {}'.format(model_type))

    if model_type == 'RF':
        # model = RandomForestClassifier(random_state=SEED)
        space = {'model': 'RF',
                 'n_estimators': 1 + hp.randint('n_estimators', 40),
                 'max_features': 1 + hp.randint('max_features', 15)}
    elif model_type == 'SVC':
        space = {'model': 'SVM',
                'C': hp.choice('C', [0.1, 0.5, 1.0]),
                'kernel': 'linear'}
    elif model_type == 'XGB':
        space = {'model': 'XGB',
                 'n_estimators': hp.choice('n_estimators', list(range(1, 20))),
                 'max_depth': hp.choice('max_depth', [4, 6, 8])}
    elif model_type == 'kNN':
        space = {'model': 'kNN',
                 'k': 1 + hp.randint('k', 15)}
    elif model_type == 'LR':
        space = {'model': 'LR',
                 'C': hp.choice('C', [0.1, 0.5, 1.0, 10, 25, 50])}
    else:
        print('wrong model type {}'.format(model_type))

    optm_params = find_optm_params(ML_objective, space)
    print('found the best hp: {}'.format(optm_params))

    model = define_model(**optm_params)
    model.fit(lang_train_X, lang_train_y)
    cv_acc(model)  # show Acc in training and test

    # thres < 0.3 may cause iRj less than 25% and therefore fail in meeting challenge's requirement
    thres_lst = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    for thres in thres_lst:
        print('---------------------------------------------------------------------------------------------')
        Ds, ICRs, CRs = cross_val_D(model, lang_train_X, train_y, cv=shuffle_inEval, THRES=thres)
        # D on the REAL test set.
        D_test, ICR_test, CR_test = get_D_on_proba(model.predict_proba(lang_test_X),
                                                   test_y, THRES=thres, print=False)
        print('Thres:{}\tCV D:{:2.4f} ICR:{:2.4f} CR:{:2.4f}\tTest D:{:2.4f} ICR:{:2.4f} CR:{:2.4f}'.format(thres,
                                                                                                            Ds.mean(),
                                                                                                            ICRs.mean(),
                                                                                                            CRs.mean(),
                                                                                                            D_test,
                                                                                                            ICR_test,
                                                                                                            CR_test))

    # with open('../data/processed/numpy/year17_models.pkl', 'wb') as pf:
    #     pickle.dump([optm_params_RF_l, optm_params_RF_m,
    #                  optm_params_XGB_l, optm_params_XGB_m,
    #                  optm_params_SVM_l, optm_params_SVM_m], pf)
