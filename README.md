# Intro

# File Structure

- data
  - fastText: containing fastText related training and trained files
  - processed: containing feature files computed
    - *numpy*: containing pickled stuffs
  - refernce_grammar:  
  - scst1: contains training and test data sets for 2017 SCST1 challenge (only for the text task)
  - texttask_trainData: contains the training data sets for 2018 SCST2 challenge's text task. Please check internal readme to know three files' purposes.
- result: 
- src: codes

# Codes

 - text_data.py: loading 2017 and 2018 data sets to generate data/processed/data.pkl
 - vecdist_feature_fasttext.py: using FastText word embedding to compute Cos-Sim and Word Moving Distance features
 - parse_grm_error.py: parse grammar error count sent by Chuan
 - prep_ml_data.ipynb  IPython notebook to prepare data for running ML tasks. The output will be in data/processed/numpy
 
 - train_model.py: using hyperopt to tune model parameters. Now support RF, SVC, and XGBoost.
 - eval_model.py: tuning a prob-cutoff for predicting "accept/reject" and run D score evaluation
 - utils.py: utility functions

## Codes for debugging purpose

- try_default_SVC.py: trying default SVC
- try_tuned_SVC.py: tuned SVC didn't show help on higher D score
- ml_model.py
- model_sandbox.ipynb
- end_to_end.py
