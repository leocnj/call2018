import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from utils import *
import argparse

"""

"""
SEED = 1


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
        model = RandomForestClassifier(random_state=SEED)
    elif model_type == 'SVC':
        model = SVC(probability=True, random_state=SEED)
    elif model_type == 'XGB':
        model = XGBClassifier()
    elif model_type == 'kNN':
        model = KNeighborsClassifier()
    elif model_type == 'LR':
        model = LogisticRegression(random_state=SEED)
    else:
        print('wrong model type {}'.format(model_type))

    model.fit(lang_train_X, lang_train_y)
    cv_acc(model)  # show Acc in training and test

    # thres < 0.3 may cause iRj less than 25% and therefore fail in meeting challenge's requirement
    if model_type == 'XGB':
        thres_lst = [0.10, 0.15, 0.20, 0.25, 0.30]   # XGB proba concentrate on small values.
    else:
        thres_lst = [0.30, 0.35, 0.40, 0.45, 0.50]
    for thres in thres_lst:
        print('------------------------------------------------')
        Ds = cross_val_D(model, train_X,  train_y, cv=shuffle_inEval, THRES=thres)
        # D on the REAL test set.
        D_test = get_D_on_proba(model.predict_proba(lang_test_X),
                                test_y, THRES=thres, print=True)
        print('thres: {}\tave D: {:2.4f}\t D on test: {:2.4f}'.format(thres, Ds.mean(), D_test))

    


    # with open('../data/processed/numpy/year17_models.pkl', 'wb') as pf:
    #     pickle.dump([optm_params_RF_l, optm_params_RF_m,
    #                  optm_params_XGB_l, optm_params_XGB_m,
    #                  optm_params_SVM_l, optm_params_SVM_m], pf)
