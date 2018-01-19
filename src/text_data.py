import pandas as pd
import pickle
from minimal_text_task_script import read_grammar
from utils import get_logger

call_root = '../data'

logger = get_logger(__name__, simple=True)

# grammar is the XML prompt/response grammar
grammar = call_root + '/reference_grammar/referenceGrammar_v2.8.6.xml'
(grammar_dic, known_prompts) = read_grammar()

# 2017 files used:
# spreadsheet_in is the input spreadsheet with the prompts, recognition results, transcriptions and judgements
# train
spreadsheet_in = call_root + '/scst1/scst1_trainingData_textTask.csv'
df_17_train = pd.read_csv(spreadsheet_in, sep='\t', encoding="utf-8", na_filter=False)
logger.info('df_17_train shape; {}'.format(df_17_train.shape))
# test
spreadsheet_in = call_root + '/scst1/scst1_testData_annotated.csv'
df_17_test = pd.read_csv(spreadsheet_in, sep='\t', encoding="utf-8", na_filter=False)
# NOTE scst1/textProcessing_testKaldi.csv provided Kaldi ASR outputs.
spreadsheet_in = call_root + '/scst1/textProcessing_testKaldi.csv'
df_17_test_asr = pd.read_csv(spreadsheet_in, sep='\t', encoding='utf-8', na_filter=False)
df_17_test = pd.merge(df_17_test, df_17_test_asr[['Id', 'RecResult']], on='Id')
# remove \*+ to ''
df_17_test['RecResult'] = df_17_test['RecResult'].str.replace(r'\*+', '')
logger.info('df_17_test shape; {}'.format(df_17_test.shape))

# 2018 files reading
spreadsheet_in = call_root + '/texttask_trainData/scst2_training_data_A_text.csv'
df_18_A_train = pd.read_csv(spreadsheet_in, sep='\t', encoding="utf-8", na_filter=False)
logger.info('df_18_A_train shape: {}'.format(df_18_A_train.shape))
spreadsheet_in = call_root + '/texttask_trainData/scst2_training_data_B_text.csv'
df_18_B_train = pd.read_csv(spreadsheet_in, sep='\t', encoding="utf-8", na_filter=False)
logger.info('df_18_B_train shape: {}'.format(df_18_B_train.shape))
spreadsheet_in = call_root + '/texttask_trainData/scst2_training_data_C_text.csv'
df_18_C_train = pd.read_csv(spreadsheet_in, sep='\t', encoding="utf-8", na_filter=False)
logger.info('df_18_train shape: {}'.format(df_18_C_train.shape))

# pickle all loaded data
with open('../data/processed/data.pkl', 'wb') as pf:
    pickle.dump([grammar_dic,
                 df_17_train,
                 df_17_test,
                 df_18_A_train,
                 df_18_B_train,
                 df_18_C_train], pf)

logger.info('saving grammar and df to data.pkl')

# For training word2vec model using grammar's prompts and df_17_train
# reference answers in the grammar file
with open("../data/fastText/ref_and_df17train.txt", "w") as output:
    for p in known_prompts:
        output.writelines("%s\n" % ans for ans in grammar_dic[p])
    for _, row in df_17_train.iterrows():
        output.writelines("%s\n" % row['RecResult'])

# save updated df_17_test
df_17_test.to_csv('../data/scst1/2017_test_withASR.csv', index=False)