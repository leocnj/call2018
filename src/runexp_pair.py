from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV, RFE
from sklearn.preprocessing.data import StandardScaler

from tpot import TPOTClassifier
from utils import *
import argparse
import pandas as pd
import pickle
"""

"""
SEED = 1

def get_langauge_X(df, cols):
    X = df.loc[:, cols].values
    return X

def get_langauge_y(df):
    return df['language'].values

def get_meaning_y(df):
    return df['meaning'].values


def prep_data(train_csv, test_csv, USE_RFECV=True, n_feat=0):
    dfs = [] # support multi files
    for csv_ in train_csv:
        dfs.append(pd.read_csv(csv_))
    df_ta = pd.concat(dfs, join='inner') #
    print('df_ta shape {}'.format(df_ta.shape))
    dfs = [] # support multi files
    for csv_ in test_csv:
        dfs.append(pd.read_csv(csv_))
    df_ts = pd.concat(dfs, join='inner') #
    print('df_ts shape {}'.format(df_ts.shape))

    X = df_ta.iloc[:, 3:].values
    y = df_ta['language'].values

    if USE_RFECV:
        # using a fixed cv to make sure REFCV behaves deterministic
        shuffle = StratifiedKFold(n_splits=5, random_state=0)
        selector = RFECV(estimator=RandomForestClassifier(random_state=2018), cv=shuffle, step=1, verbose=0, n_jobs=-1)
    else:
        selector = RFE(estimator=RandomForestClassifier(random_state=2018), n_features_to_select=n_feat)
    selector.fit(X, y)

    feat_names = df_ta.columns[3:]
    col_selected = list(feat_names[selector.get_support()])

    print(col_selected)
    if USE_RFECV:
        print('RFECV found {} features'.format(len(col_selected)))
    else:
        print('RFE found {} features'.format(len(col_selected)))


    X_ta = get_langauge_X(df_ta, col_selected)
    X_ts = get_langauge_X(df_ts, col_selected)

    # pre-processing X
    scaler = StandardScaler()
    scaler.fit(X_ta)
    X_ta = scaler.transform(X_ta)
    X_ts = scaler.transform(X_ts)

    y_l_ta = get_langauge_y(df_ta)
    y_m_ta = get_meaning_y(df_ta)
    y_l_ts = get_langauge_y(df_ts)
    y_m_ts = get_meaning_y(df_ts)

    return [X_ta, y_l_ta, X_ts, y_l_ts, y_m_ta, y_m_ts]


def one_expm(objs, model_type, shuffle, shuffle_inEval, model_file):
    lang_train_X = objs[0]
    lang_train_y = objs[1]
    lang_test_X = objs[2]
    lang_test_y = objs[3]
    meaning_train_y = objs[4]
    meaning_test_y = objs[5]
    train_y = np.column_stack((meaning_train_y, lang_train_y))
    test_y = np.column_stack((meaning_test_y, lang_test_y))

    print('model type: {}'.format(model_type))
    if RUN_FIT:
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
        elif model_type == 'Ensemble':
            model = VotingClassifier(estimators=[
                ('lr', LogisticRegression(random_state=SEED)),
                ('rf', RandomForestClassifier(random_state=SEED)),
                ('svc', SVC(probability=True, random_state=SEED)),
                ('xgb', XGBClassifier())], voting = 'soft')
        elif model_type == 'TPOT':
            model = TPOTClassifier(generations=5, population_size=25, cv=shuffle,
                                   random_state=SEED, verbosity=2, n_jobs=-1)
            model.fit(lang_train_X, lang_train_y)
            model = model.fitted_pipeline_  # only keep sklearn pipeline.
        else:
            print('wrong model type {}'.format(model_type))
            # for test
        model.fit(lang_train_X, lang_train_y)
        # save model
        with open(model_file, 'wb') as pf:
                pickle.dump(model, pf)
    else:
        print('load saved model {}'.format(model_file))
        with open(model_file, 'rb') as pf:
            model = pickle.load(pf)

    cv_score = cross_val_score(model,
                               lang_train_X, lang_train_y,
                               cv=shuffle_inEval,
                               scoring='accuracy')
    acc_test = accuracy_score(lang_test_y, model.predict(lang_test_X))
    print('Acc mean on train: {:2.4f}\tAcc on test: {:2.4f}'.format(cv_score.mean(), acc_test))

    thres_lst = [0.20, 0.25, 0.30, 0.350, 0.40, 0.45, 0.50]
    for thres in thres_lst:
        print('---------------------------------------------------------------------------------------------')
        Ds, ICRs, CRs = cross_val_D(model, lang_train_X, train_y, cv=shuffle_inEval, THRES=thres)
        # D on the REAL test set.
        D_test, ICR_test, CR_test = get_D_on_proba(model.predict_proba(lang_test_X), test_y, THRES=thres, print=False,
                                                   CR_adjust=False)
        print('Thres:{}\tCV D:{:2.4f} ICR:{:2.4f} CR:{:2.4f}\tTest D:{:2.4f} ICR:{:2.4f} CR:{:2.4f}'.format(thres,
                                                                                                            Ds.mean(),
                                                                                                            ICRs.mean(),
                                                                                                            CRs.mean(),
                                                                                                            D_test,
                                                                                                            ICR_test,
                                                                                                            CR_test))
import os
def model_fname(train_csv, model_type):
    csv_ids = [os.path.splitext(os.path.basename(x))[0] for x in train_csv]
    model_file = '../ml_exp/model/' + '-'.join(csv_ids) + '-' + model_type + '.pkl'
    print(model_file)
    return model_file

if __name__ == '__main__':

    # skip lots of sklearn depreciation warnings
    import warnings
    warnings.filterwarnings("ignore")
    # logger = get_logger(__name__, simple=True)

    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--model_type', type=str, help='model type',
                        required=True)
    parser.add_argument('--train', type=str, help='train csv', required=True, nargs='+') # support multi train files.
    parser.add_argument('--test', type=str, help='test csv', required=True, nargs='+')
    parser.add_argument('--fit', help='fit a model', action='store_true')
    parser.add_argument('--nfeat', type=int, help='RFE n_feat', default=0)
    args = parser.parse_args()

    model_type = args.model_type
    train_csv = args.train
    test_csv = args.test
    RUN_FIT = args.fit
    n_feat = args.nfeat

    if n_feat:
        USE_REFCV = False
    else:
        USE_REFCV = True

    objs = prep_data(train_csv, test_csv, USE_RFECV=USE_REFCV, n_feat=n_feat)
    model_file = model_fname(train_csv, model_type)

    # to use same CV data splitting
    shuffle = StratifiedKFold(n_splits=10, random_state=SEED)
    shuffle_inEval = StratifiedKFold(n_splits=10, random_state=SEED + 1024)

    one_expm(objs, model_type, shuffle, shuffle_inEval, model_file)
