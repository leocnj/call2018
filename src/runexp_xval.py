from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.preprocessing.data import StandardScaler

from utils import *
import argparse
import pandas as pd

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


def prep_data(train_csv, ONLY_A=False):

    dfs = [] # support multi files
    csv_idx = 0
    abc_vec  = []
    for csv_ in train_csv:
        df_ = pd.read_csv(csv_)
        abc_vec += [csv_idx for x in range(len(df_.index))]
        csv_idx += 1
        dfs.append(df_)
    df_ta = pd.concat(dfs, join='inner')

    X = df_ta.iloc[:, 3:].values
    y = get_langauge_y(df_ta)

    cv = StratifiedKFold(n_splits=5, random_state=2018)
    selector = RFECV(estimator=RandomForestClassifier(random_state=2018), cv=cv, step=1, verbose=0, n_jobs=-1)
    selector.fit(X, y)

    feat_names = df_ta.columns[3:]
    col_selected = list(feat_names[selector.get_support()])

    print(col_selected)
    print('RFECV found {} features'.format(len(col_selected)))

    X_ta = get_langauge_X(df_ta, col_selected)

    # pre-processing X
    scaler = StandardScaler()
    scaler.fit(X_ta)
    X_ta = scaler.transform(X_ta)

    # xval
    y_m = get_meaning_y(df_ta)
    # abc_vec array can be used to restrict data choice to 2018 A set.
    abc_vec = np.asarray(abc_vec)
    objs_lst = []
    # cv = StratifiedKFold(n_splits=5, random_state=2018)
    for train_index, test_index in cv.split(X_ta, y):
        #
        if ONLY_A:
            train_index = np.extract(abc_vec[train_index] == 0, train_index)
            print('only A size {}'.format(len(train_index)))
        X_ta_, y_l_ta = X_ta[train_index], y[train_index]
        X_ts_, y_l_ts = X_ta[test_index], y[test_index]
        y_m_ta, y_m_ts = y_m[train_index], y_m[test_index]
        objs_lst.append([X_ta_, y_l_ta, X_ts_, y_l_ts, y_m_ta, y_m_ts])
    return objs_lst


def one_expm(objs, model_type, shuffle, shuffle_inEval):
    lang_train_X = objs[0]
    lang_train_y = objs[1]
    lang_test_X = objs[2]
    lang_test_y = objs[3]
    meaning_train_y = objs[4]
    meaning_test_y = objs[5]
    train_y = np.column_stack((meaning_train_y, lang_train_y))
    test_y = np.column_stack((meaning_test_y, lang_test_y))

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
    cv_score = cross_val_score(model,
                               lang_train_X, lang_train_y,
                               cv=shuffle_inEval,
                               scoring='accuracy')
    # when w/o onlyA, cv_score freezes after first fold. had to disable njobs=-1

    # for test
    model.fit(lang_train_X, lang_train_y)
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

if __name__ == '__main__':

    # skip lots of sklearn depreciation warnings
    import warnings
    warnings.filterwarnings("ignore")
    # logger = get_logger(__name__, simple=True)

    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--model_type', type=str, help='model type',
                        required=True)
    parser.add_argument('--train', type=str, help='train csv', required=True, nargs='+')
    parser.add_argument('--onlyA', help='only using A to train', action='store_true')
    args = parser.parse_args()

    model_type = args.model_type
    train_csv = args.train

    objs_lst = prep_data(train_csv, ONLY_A=args.onlyA)

    # to use same CV data splitting
    shuffle = StratifiedKFold(n_splits=10, random_state=SEED)
    shuffle_inEval = StratifiedKFold(n_splits=10, random_state=SEED + 1024)

    for objs in objs_lst:
        one_expm(objs, model_type, shuffle, shuffle_inEval)
