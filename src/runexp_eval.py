from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFECV, RFE
from sklearn.preprocessing.data import StandardScaler
from sklearn.pipeline import make_pipeline

from tpot import TPOTClassifier
from utils import *
import argparse
import pandas as pd
import pickle

"""
runexp_train.py

- only do training
- pickle transformer and model for future use

"""
SEED = 1

def get_meaning_y(df):
    return df['meaning'].values

def prep_test(test_csv, pipe):
    df_ts = pd.read_csv(test_csv)
    print('df_ts shape {}'.format(df_ts.shape))

    X = df_ts.iloc[:, 3:].values
    X_ts = pipe.transform(X)
    return X_ts


import os
def pred_fname(test_csv):
    csv_id = os.path.splitext(os.path.basename(test_csv))[0]
    model_file = '../ml_exp/preds/' + csv_id + '.csv'
    print(model_file)
    return model_file


if __name__ == '__main__':

    # skip lots of sklearn depreciation warnings
    import warnings

    warnings.filterwarnings("ignore")
    # logger = get_logger(__name__, simple=True)

    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str, help='model file',
                        required=True)
    parser.add_argument('--test', type=str, help='test csv', required=True)
    args = parser.parse_args()

    model_file = args.model_file
    test_csv = args.test

    with open(model_file, 'rb') as pf:
         pipe, ml_model = pickle.load(pf)

    pred_file = pred_fname(test_csv)
    X_ts = prep_test(test_csv, pipe)
    probs = ml_model.predict_proba(X_ts)
    print(probs)

    pd.DataFrame(probs[:,1], columns=['prob_1']).to_csv(pred_file, index=False)
