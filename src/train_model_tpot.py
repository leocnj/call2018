import pickle
import multiprocessing

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from tpot import TPOTClassifier

from utils import *
import argparse

"""

"""
SEED = 1


if __name__ == '__main__':

    # multiprocessing.set_start_method('forkserver')

    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_file', type=str, help='model file',
                        required=True)
    parser.add_argument('-d', '--data', type=str, help='pickled data', required=True)
    parser.add_argument('--train', help='train model using TPOT', action='store_true')
    args = parser.parse_args()
    model_file = args.model_file
    pkl_file = args.data
    RUN_TPOT = args.train

    with open('../data/processed/numpy/' + pkl_file, 'rb') as pf:
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

    # logger = get_logger(__name__, simple=True)

    # to use same CV data splitting
    shuffle = StratifiedKFold(n_splits=10, random_state=SEED)
    shuffle_inEval = StratifiedKFold(n_splits=10, random_state=SEED + 1024)

    if RUN_TPOT:
        model = TPOTClassifier(generations=5, population_size=25, cv=shuffle,
                               random_state=SEED, verbosity=2, n_jobs=-1)

        model.fit(lang_train_X, lang_train_y)
        model = model.fitted_pipeline_  # only keep sklearn pipeline.
        print('Acc on test: {}'.format(model.score(lang_test_X, lang_test_y)))
        with open('../result/' + model_file, 'wb') as pf:
            pickle.dump(model, pf)
    else:
        with open('../result/' + model_file, 'rb') as pf:
            model = pickle.load(pf)

    thres_lst = [0.25, 0.30, 0.350, 0.40, 0.45, 0.50]
#    thres_lst = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    for thres in thres_lst:
        print('---------------------------------------------------------------------------------------------')
        Ds, ICRs, CRs = cross_val_D(model, lang_train_X,  train_y, cv=shuffle_inEval, THRES=thres)
        # D on the REAL test set.
        D_test, ICR_test, CR_test = get_D_on_proba(model.predict_proba(lang_test_X), test_y, THRES=thres, print=False, CR_adjust=True)
        print('Thres:{}\tCV D:{:2.4f} ICR:{:2.4f} CR:{:2.4f}\tTest D:{:2.4f} ICR:{:2.4f} CR:{:2.4f}'.format(thres,
                                                                                                            Ds.mean(),
                                                                                                            ICRs.mean(),
                                                                                                            CRs.mean(),
                                                                                                            D_test,
                                                                                                            ICR_test,
                                                                                                            CR_test))

