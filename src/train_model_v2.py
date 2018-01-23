import pickle

import numpy as np
from hyperopt import fmin, tpe, Trials
from hyperopt import hp
from hyperopt import space_eval
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score

from mlens.ensemble import BlendEnsemble

from utils import *
import argparse

"""
- Support RF, XGBoost, and SVM
- Use **kwargs dict to pass parameters
- Use ML ens to stack models
- Use hyperopt for more convenient space definition

"""
SEED = 1


def define_model(**params):
    if params['model'] == 'RF':
        n_estimators = params['n_estimators']
        max_features = params['max_features']
        model = RandomForestClassifier(n_estimators=n_estimators,
                                       max_features=max_features)
    elif params['model'] == 'XGB':
        n_estimators = params['n_estimators']
        max_depth = params['max_depth']
        model = XGBClassifier(n_estimators=n_estimators,
                              max_depth=max_depth)
    elif params['model'] == 'SVM':
        C = params['C']
        kernel = params['kernel']
        model = SVC(C=C, kernel=kernel, probability=True)
    else:
        pass
    return model

@timeit
def find_optm_params(objective, space, max_evals=20):
    trials = Trials()
    best = fmin(objective,
                space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)
    best_params = space_eval(space, best)
    return best_params


def stack_models(base_estimators, type):
    blender = BlendEnsemble()
    blender.add(base_estimators, proba=True)
    blender.add_meta(LogisticRegression())
    train_eval(blender, type)



if __name__ == '__main__':

    # year17 data
    with open('../data/processed/numpy/year17_withHuy.pkl', 'rb') as pf:
        objs = pickle.load(pf)
        lang_train_X = objs[0]
        lang_train_y = objs[1]
        lang_test_X = objs[2]
        lang_test_y = objs[3]

        meaning_train_X = objs[4]
        meaning_train_y = objs[5]
        meaning_test_X = objs[6]
        meaning_test_y = objs[7]

        train_X = np.concatenate((meaning_train_X, lang_train_X), axis=1)
        train_y = np.column_stack((meaning_train_y, lang_train_y))
        test_X = np.concatenate((meaning_test_X, lang_test_X), axis=1)
        test_y = np.column_stack((meaning_test_y, lang_test_y))

    # skip lots of sklearn depreciation warnings
    import warnings
    warnings.filterwarnings("ignore")

    # logger = get_logger(__name__, simple=True)

    """
    """


    # to use same CV data splitting
    shuffle = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    shuffle_inEval = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED + 1024)
    scoring_metric = 'roc_auc'
    # define inside main to easily access data
    def score_objective(params):
        model = define_model(**params)
        cv_score = cross_val_score(model,
                                   lang_train_X, lang_train_y,
                                   cv=shuffle,
                                   scoring=scoring_metric, n_jobs=-1)
        return 1 - cv_score.mean()


    def train_eval(best_model):
        best_model.fit(lang_train_X, lang_train_y)
        y_true, y_pred = lang_test_y, best_model.predict(lang_test_X)
        print('acc: %1.3f' % accuracy_score(y_true, y_pred))


    def eval_best_model(optm_params):
        best_model = define_model(**optm_params)
        train_eval(best_model)


    def cv_acc(model):
        cv_score = cross_val_score(model,
                                   lang_train_X, lang_train_y,
                                   cv=shuffle_inEval,
                                   scoring='accuracy', n_jobs=-1)
        print(cv_score)
        print(np.median(cv_score))


    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--model_type', type=str, help='model type',
                        required=True)
    args = parser.parse_args()
    model_type = args.model_type
    print('model type: {}'.format(model_type))


    if model_type == 'RF':
        model = RandomForestClassifier(random_state=SEED)
    elif model_type == 'SVC':
        model = SVC(probability=True, random_state=SEED)
    elif model_type == 'XGB':
        model = XGBClassifier()
    else:
        print('wrong model type {}'.format(model_type))

    cv_acc(model)
    # XGB needs very small thres, e.g., 0.185
    for thres in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        Ds = cross_val_D(model, train_X,  train_y, cv=shuffle_inEval, THRES=thres)
        print('------------------------------------------------')
        print('thres: {} mean of D {}'.format(thres, Ds.mean()))
        # D on the REAL test set.
        D_test = get_D_on_proba(model.predict_proba(lang_test_X),
                                test_y, THRES=thres, print=False)
        print('on test set final D: {}'.format(D_test))

    


    # with open('../data/processed/numpy/year17_models.pkl', 'wb') as pf:
    #     pickle.dump([optm_params_RF_l, optm_params_RF_m,
    #                  optm_params_XGB_l, optm_params_XGB_m,
    #                  optm_params_SVM_l, optm_params_SVM_m], pf)
