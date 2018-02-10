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
def pred_fname(model_file, test_csv):
    model_id = os.path.splitext(os.path.basename(model_file))[0]
    csv_id = os.path.splitext(os.path.basename(test_csv))[0]
    model_file = '../ml_exp/preds/' + model_id + '_' + csv_id + '.csv'
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

    pred_file = pred_fname(model_file, test_csv)
    X_ts = prep_test(test_csv, pipe)
    probs = ml_model.predict_proba(X_ts)
    print(probs)

    pd.DataFrame(probs[:,1], columns=['prob_1']).to_csv(pred_file, index=False)
