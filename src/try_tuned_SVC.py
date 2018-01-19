import pickle
import numpy as np
from utils import *
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import optunity
import optunity.metrics

label_back = lambda x: 'correct' if x==1 else 'incorrect'

def get_D_based_thresholds(LANG_T, lang_prob, y, print=False):
    scores = init_scores()
    for i, p_lang in enumerate(lang_prob):
        if p_lang[1] >= LANG_T:
            decision = 'accept'
        else:
            decision = 'reject'
        score_decision(decision, label_back(get_langauge_y(y)[i]), label_back(get_meaning_y(y)[i]), scores)
    if print:
        print_scores(scores)
    return get_D(scores)

MEANING_DIM = 24
def get_meaning_X(X):
    return X[:,0:MEANING_DIM]

def get_meaning_y(y):
    return y[:,0]

def get_langauge_X(X):
    return X[:, MEANING_DIM:]  # using -1 will lose the last column.

def get_langauge_y(y):
    return y[:,1]


if __name__ == '__main__':

 # year17 data
    with open('../data/processed/numpy/year17_withHuy.pkl', 'rb') as pf:
        objs = pickle.load(pf)
        year17_lang_train_X = objs[0]
        year17_lang_train_y = objs[1]
        year17_lang_test_X = objs[2]
        year17_lang_test_y = objs[3]

        year17_meaning_train_X = objs[4]
        year17_meaning_train_y = objs[5]
        year17_meaning_test_X = objs[6]
        year17_meaning_test_y = objs[7]

        year17_train_X = np.concatenate((year17_meaning_train_X, year17_lang_train_X), axis=1)
        year17_train_y = np.column_stack((year17_meaning_train_y, year17_lang_train_y))
        year17_test_X = np.concatenate((year17_meaning_test_X, year17_lang_test_X), axis=1)
        year17_test_y = np.column_stack((year17_meaning_test_y, year17_lang_test_y))


    logger = get_logger(__name__, simple=True)

    @optunity.cross_validated(x=year17_train_X, y=year17_train_y, num_folds=5)
    def tune_Dscore(x_train, y_train, x_test, y_test, L_thres=0.5):
        # ------------ Found these params after tuning -----------------------
        # {'C': 1000, 'gamma': 0.001, 'kernel': 'rbf'}
        model_lang = SVC(C=1000, gamma=0.001, kernel='rbf', probability=True)
        model_lang.fit(get_langauge_X(x_train), get_langauge_y(y_train))
        D_score = get_D_based_thresholds(L_thres,
                                         model_lang.predict_proba(get_langauge_X(x_test)),
                                         y_test)
        return D_score

    @timeit
    # this can use optunity's multi-thread and smarter solver
    def tune_D_smart():
        thres_grid = {'L_thres': [0.25, 0.35]}
        optm_thres, info, _ = optunity.maximize_structured(tune_Dscore,
                                                           num_evals=20,
                                                           pmap=optunity.pmap,
                                                           search_space=thres_grid)
        print("Optimal parameters" + str(optm_thres))
        print("best D after CV running: %2.3f" % info.optimum)
        return (optm_thres['L_thres'])

    best_L = tune_D_smart()  # can run this after first round tuning.
    # best_L = 0.3

    TO_TUNE = False
    if not TO_TUNE:
        model_lang = SVC(C=1000, gamma=0.001, kernel='rbf', probability=True)
        model_lang.fit(year17_lang_train_X, year17_lang_train_y)
    else:
        # Using GridSearch model tuning
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 10]}]
        model_lang = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5, scoring='accuracy', verbose=5)
        model_lang.fit(year17_lang_train_X, year17_lang_train_y)
        print("Best parameters set found on development set:")
        print()
        print(model_lang.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = model_lang.cv_results_['mean_test_score']
        stds = model_lang.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, model_lang.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

    probs_l = model_lang.predict_proba(year17_lang_test_X)
    y_true, y_pred = year17_lang_test_y, model_lang.predict(year17_lang_test_X)

    print(accuracy_score(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))

    D_score = get_D_based_thresholds(best_L,
                                     probs_l,
                                     year17_test_y, True)
