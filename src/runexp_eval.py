import argparse
import pandas as pd
import pickle
from utils import get_D_on_df

"""
runexp_eval.py

- load saved pipeline
- apply the pre-processing and model on the test file
- generate result csv file at preds dir

"""
SEED = 1

def get_meaning_y(df):
    return df['meaning'].values

def prep_test(test_csv, pipe):
    df_ts = pd.read_csv(test_csv)
    print('df_ts shape {}'.format(df_ts.shape))
    if {'language', 'meaning'}.issubset(df_ts.columns):
        X = df_ts.iloc[:, 3:].values
    else:
        X = df_ts.iloc[:, 1:].values
    X_ts = pipe.transform(X)
    return X_ts


import os
def pred_fname(model_file, test_csv, thres):
    model_id = os.path.splitext(os.path.basename(model_file))[0]
    csv_id = os.path.splitext(os.path.basename(test_csv))[0]
    model_file = '../ml_exp/preds/' + model_id + '_' + csv_id + '_t' + str(thres) + '.csv'
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
    parser.add_argument('--year', type=str, help='challenge year', required=True)
    parser.add_argument('--thres', type=float, help='proba_ threshold', required=True)
    args = parser.parse_args()

    model_file = args.model_file
    test_csv = args.test
    test_year = args.year
    test_thres = args.thres

    with open(model_file, 'rb') as pf:
         pipe, ml_model = pickle.load(pf)

    pred_file = pred_fname(model_file, test_csv, test_thres)
    X_ts = prep_test(test_csv, pipe)
    probs = ml_model.predict_proba(X_ts)

    # based on year_thres, to make judgement
    make_judge = lambda x: 'accept' if x >= test_thres else 'reject'
    judgements = [make_judge(prob) for prob in probs[:, 1]]

    # load test meta
    if test_year == '2017':
        meta_csv = '../data/scst1/scst1_testData_annotated.csv'
    elif test_year == '2018_text':
        meta_csv = '../data/texttask_trainData/scst2_testDataText.csv'
    elif test_year == '2018_asr':
        meta_csv = '../data/texttask_trainData/scst2_testDataSpeech.csv'
    result_df = pd.read_csv(meta_csv, sep='\t', encoding="utf-8", na_filter=False)
    result_df['Judgement'] = judgements
    # result_df['proba'] = probs[:, 1]
    result_df.to_csv(pred_file, index=False, sep='\t')

    if test_year == '2017':
        get_D_on_df(result_df)
