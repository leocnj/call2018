import pandas as pd
import pickle


def recode_df(df):
    recode = lambda x: 1 if x == 'correct' else 0
    df['language'] = df['language'].apply(recode)
    df['meaning'] = df['meaning'].apply(recode)
    return df


def get_17df():
    with open('../data/processed/data.pkl', 'rb') as pf:
        objs = pickle.load(pf)

    df_train = objs[1]  # using RecResult
    df_test = objs[2]  # using RecResult

    df_train = recode_df(df_train)
    df_test = recode_df(df_test)

    print('2017 DF')
    print('train: {} test: {}'.format(df_train.shape, df_test.shape))
    return (df_train, df_test)


def get_18df():
    with open('../data/processed/data.pkl', 'rb') as pf:
        objs = pickle.load(pf)

    df_A = objs[3]  # using RecResult
    df_B = objs[4]  # using RecResult
    df_C = objs[5]

    df_A = recode_df(df_A)
    df_B = recode_df(df_B)
    df_C = recode_df(df_C)

    print('2018 DF')
    for df_ in [df_A, df_B, df_C]:
        print('shape: {}'.format(df_.shape))
    return (df_A, df_B, df_C)

def get_18test():
    # 2018 test
    spreadsheet_in = '../data/texttask_trainData/scst2_testDataText.csv'
    df_18_test = pd.read_csv(spreadsheet_in, sep='\t', encoding="utf-8", na_filter=False)
    print('2018 test\nshape: {}'.format(df_18_test.shape))
    return df_18_test


def load_huy(csv_file):
    df = pd.read_csv(csv_file, sep='\t')
    df.rename(columns={'ID': 'Id'}, inplace=True)
    df.drop(columns=['CLASS'], inplace=True)
    print('Huy features: {}'.format(df.shape))
    # Id plus 43 columns
    col_huy = ["Id", "ppl-ref", "ppl-ref_pos", "ppl-ref_prod", "ppl-ref_dep", "ppl-prompt", "ppl-prompt_pos", "ppl-correct", "ppl-correct_pos", "ppl-correct_prod", "ppl-correct_dep", "ppl-ge", "ppl-ge_pos", "ppl-incorrect", "ppl-incorrect_pos", "ppl-incorrect_prod", "ppl-incorrect_dep", "maxsim_15_skip", "maxsim_30_skip", "maxsim_50_skip", "maxsim_15_cbw", "maxsim_30_cbw", "maxsim_50_cbw", "lda_sim-min", "ngram_match", "ngram_match-lem", "ngram_unseen-1", "ngram_unseen-2", "ngram_unseen-3", "error_count", "parse_score-ratio", "length_ratio", "length_01", "length_unknown", "length_unknown-ratio", "length_sounds", "length_sounds-ratio", "prompt_missing", "prompt_missing-pct", "prompt_DT", "prompt_IN", "prompt_MD", "prompt_NN", "prompt_VB"]
    df = df[col_huy]
    print('reduced to {}'.format(df.shape))
    return df


def gen_ml_csv(df_main, grmerr_csv, huy_csv, ml_csv):
    """

    :param df_main:
    :param grmerr_csv:
    :param huy_csv:
    :param ml_csv:
    :return:
    """
    print('ml_csv: {}'.format(ml_csv))
    df_grmerr = pd.read_csv(grmerr_csv)
    df_huy = load_huy(huy_csv)

    # 2018 test has no L and M columns
    if {'language', 'meaning'}.issubset(df_main.columns):
        df_ml = df_main[['Id', 'language', 'meaning']]
    else:
        df_ml = df_main[['Id']]

    df_ml = pd.merge(df_ml, df_grmerr, on='Id',
                     how='left')  # grmerr may miss some Ids due to ASR null outputs. use left to keep all Ids.
    df_ml = pd.merge(df_ml, df_huy, on='Id', how='left')
    df_ml.fillna(0, inplace=True)
    df_ml.to_csv(ml_csv, index=False)


if __name__ == '__main__':
    y17_train, y17_test = get_17df()
    y18_train_A, y18_train_B, y18_train_C = get_18df()
    y18_test = get_18test()

    # ------------------------- TEXT task -------------------------------------------------
    # 2017 train text
    gen_ml_csv(y17_train,
               '../data/processed/df17_train_grmerror.csv',
               '../data/processed/Huy/v6_17ABC/textProcessing_trainingKaldi_features.csv',
               '../ml_exp/inputs/y17_train_text.csv')
    # 2017 test text
    gen_ml_csv(y17_test,
               '../data/processed/df17_test_grmerror.csv',
               '../data/processed/Huy/v6_17ABC/textProcessing_testKaldi_annotated_features.csv',
               '../ml_exp/inputs/y17_test_text.csv')  # 2017 train asr


    # text
    # 2018 train A
    gen_ml_csv(y18_train_A,
               '../data/processed/df18_A_train_grmerror.csv',
               '../data/processed/Huy/v6_17ABC/scst2_training_data_A_text_features.csv',
               '../ml_exp/inputs/y18_train_A_text.csv')
    # 2018 train B
    gen_ml_csv(y18_train_B,
               '../data/processed/df18_B_train_grmerror.csv',
               '../data/processed/Huy/v6_17ABC/scst2_training_data_B_text_features.csv',
               '../ml_exp/inputs/y18_train_B_text.csv')
    # 2018 train C
    gen_ml_csv(y18_train_C,
               '../data/processed/df18_C_train_grmerror.csv',
               '../data/processed/Huy/v6_17ABC/scst2_training_data_C_text_features.csv',
               '../ml_exp/inputs/y18_train_C_text.csv')
    # 2018 test
    gen_ml_csv(y18_test,
               '../data/processed/df18_test_text_grmerror.csv',
               '../data/processed/Huy/v6_17ABC/scst2_testDataText_features.csv',
               '../ml_exp/inputs/y18_test_text.csv')


    # ------------------------- ASR task -------------------------------------------------
    # ASR using 1-best
    #
    # 2017 train asr
    gen_ml_csv(y17_train,
               '../data/processed/df17_train_asr_grmerror.csv',
               '../data/processed/Huy/v7_ASR/data/17ABC-asr-1best/textProcessing_trainingKaldi_features.csv',
               '../ml_exp/inputs/y17_train_asr.csv')
    # 2017 test asr
    gen_ml_csv(y17_test,
               '../data/processed/df17_test_asr_grmerror.csv',
               '../data/processed/Huy/v7_ASR/data/17ABC-asr-1best/textProcessing_testKaldi_annotated_features.csv',
               '../ml_exp/inputs/y17_test_asr.csv')

    # 2018 train A
    gen_ml_csv(y18_train_A,
               '../data/processed/df18_train_asr_grmerror.csv',
               '../data/processed/Huy/v7_ASR/data/17ABC-asr-1best/scst2_training_data_A_text_features.csv',
               '../ml_exp/inputs/y18_train_A_asr.csv')
    # 2018 train B
    gen_ml_csv(y18_train_B,
               '../data/processed/df18_train_asr_grmerror.csv',
               '../data/processed/Huy/v7_ASR/data/17ABC-asr-1best/scst2_training_data_B_text_features.csv',
               '../ml_exp/inputs/y18_train_B_asr.csv')
    # 2018 train C
    gen_ml_csv(y18_train_C,
               '../data/processed/df18_train_asr_grmerror.csv',
               '../data/processed/Huy/v7_ASR/data/17ABC-asr-1best/scst2_training_data_C_text_features.csv',
               '../ml_exp/inputs/y18_train_C_asr.csv')

    # 2018 test
    gen_ml_csv(y18_test,
               '../data/processed/df18_test_asr_grmerror.csv',
               '../data/processed/Huy/v7_ASR/data/18test-asr-1best-new/scst2_testDataText_features.csv',
               '../ml_exp/inputs/y18_test_asr.csv')


    # ASR using 2-best (when testing)
    # 2017 test asr
    gen_ml_csv(y17_test,
               '../data/processed/df17_test_asr_grmerror.csv',
               '../data/processed/Huy/v7_ASR/data/17ABC-asr-2best/textProcessing_testKaldi_annotated_features.csv',
               '../ml_exp/inputs/y17_test_2best.csv')

    # 2018 test
    gen_ml_csv(y18_test,
               '../data/processed/df18_test_asr_grmerror.csv',
               '../data/processed/Huy/v7_ASR/data/18test-asr-2best-new/scst2_testDataText_features.csv',
               '../ml_exp/inputs/y18_test_2best.csv')

