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

    # skip lots of sklearn depreciation warnings
    import warnings
    warnings.filterwarnings("ignore")

    # logger = get_logger(__name__, simple=True)

    """
    """

    # to use same CV data splitting
    shuffle = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    scoring_metric = 'roc_auc'
    # define inside main to easily access data
    def acc_l_objective(params):
        model = define_model(**params)
        acc_score = cross_val_score(model,
                                    year17_lang_train_X, year17_lang_train_y,
                                    cv=shuffle,
                                    scoring=scoring_metric, n_jobs=-1)
        return 1 - acc_score.mean()


    def acc_m_objective(params):
        model = define_model(**params)
        acc_score = cross_val_score(model,
                                    year17_meaning_train_X, year17_meaning_train_y,
                                    cv=shuffle,
                                    scoring=scoring_metric, n_jobs=-1)
        return 1 - acc_score.mean()


    def eval_best_model(optm_params, type):
        best_model = define_model(**optm_params)
        train_eval(best_model, type)


    def train_eval(best_model, type):
        if type == 'lang':
            best_model.fit(year17_lang_train_X, year17_lang_train_y)
            y_true, y_pred = year17_lang_test_y, best_model.predict(year17_lang_test_X)
        else:
            best_model.fit(year17_meaning_train_X, year17_meaning_train_y)
            y_true, y_pred = year17_meaning_test_y, best_model.predict(year17_meaning_test_X)
        print('acc: %1.3f' % accuracy_score(y_true, y_pred))


    RF = True
    SVM = True
    XGB = True

    if not RF:
        optm_params_RF_l, optm_params_RF_m = None, None
    else:
        space = {'model': 'RF',
                 'n_estimators': 1 + hp.randint('n_estimators', 40),
                 'max_features': 1 + hp.randint('max_features', 15)}

        optm_params_RF_l = find_optm_params(acc_l_objective, space)
        print(optm_params_RF_l)
        eval_best_model(optm_params_RF_l, 'lang')

        optm_params_RF_m = find_optm_params(acc_m_objective, space)
        print(optm_params_RF_m)
        eval_best_model(optm_params_RF_m, 'meaning')


    if not SVM:
        optm_params_SVM_l, optm_params_SVM_m = None, None
    else:
        space = {
            'model': 'SVM',
            'C': hp.choice('C', [0.1, 0.5, 1.0]),
            'kernel': 'linear'
            # 'kernel': hp.choice('svm_kernel', [
            #     {'ktype': 'linear'},
            #     {'ktype': 'rbf', 'width': hp.lognormal('svm_rbf_width', 0, 1)},
             }
        optm_params_SVM_l = find_optm_params(acc_l_objective, space)
        print(optm_params_SVM_l)
        eval_best_model(optm_params_SVM_l, 'lang')
        optm_params_SVM_m = find_optm_params(acc_m_objective, space)
        print(optm_params_SVM_m)
        eval_best_model(optm_params_SVM_m, 'meaning')


    if not XGB:
        optm_params_XGB_l, optm_params_XGB_m = None, None
    else:
        # space = {'model': 'XGB',
        #          'n_estimators': hp.choice('n_estimators', [10, 15, 20]),
        #          'max_depth': hp.choice('max_depth', [4, 6, 8])}
        space = {'model': 'XGB',
                 'n_estimators': hp.choice('n_estimators', list(range(1,20))),
                 'max_depth': hp.choice('max_depth', [4, 6, 8])}

        optm_params_XGB_l = find_optm_params(acc_l_objective, space)
        print(optm_params_XGB_l)
        eval_best_model(optm_params_XGB_l, 'lang')

        optm_params_XGB_m = find_optm_params(acc_m_objective, space)
        print(optm_params_XGB_m)
        eval_best_model(optm_params_XGB_m, 'meaning')

    # ensemble
    base_estimators = [define_model(**optm_params_RF_l),
                       define_model(**optm_params_XGB_l)]
    stack_models(base_estimators, 'lang')

    base_estimators = [define_model(**optm_params_RF_m),
                       define_model(**optm_params_XGB_m)]
    stack_models(base_estimators, 'meaning')

    with open('../data/processed/numpy/year17_models.pkl', 'wb') as pf:
        pickle.dump([optm_params_RF_l, optm_params_RF_m,
                     optm_params_XGB_l, optm_params_XGB_m,
                     optm_params_SVM_l, optm_params_SVM_m], pf)
        # pickle.dump([optm_params_SVM_l, optm_params_SVM_m], pf)
