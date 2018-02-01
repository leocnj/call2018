import pickle
import numpy as np
import sklearn.svm
import optunity
import optunity.metrics
from utils import *
import time

# https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
#
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1)
        else:
            print('%r  %2.2f Seconds' % \
                  (method.__name__, (te - ts) * 1))
        return result
    return timed


def train_model(x_train, y_train, kernel, C, logGamma, degree, coef0):
    """A generic SVM training function, with arguments based on the chosen kernel."""
    if kernel == 'linear':
        model = sklearn.svm.SVC(kernel=kernel, C=C, probability=True)
    elif kernel == 'poly':
        model = sklearn.svm.SVC(kernel=kernel, C=C, degree=degree, coef0=coef0, probability=True)
    elif kernel == 'rbf':
        model = sklearn.svm.SVC(kernel=kernel, C=C, gamma=10 ** logGamma, probability=True)
    else:
        raise ArgumentError("Unknown kernel function: %s" % kernel)
    model.fit(x_train, y_train)
    return model


def get_svm_bypars(x_train, y_train, pars):
    model = train_model(x_train, y_train, pars['kernel'], pars['C'], pars['logGamma'], pars['degree'], pars['coef0'])
    return model


def svm_tuned_acc(x_train, y_train, x_test, y_test, kernel='linear', C=0, logGamma=0, degree=0, coef0=0):
    model = train_model(x_train, y_train, kernel, C, logGamma, degree, coef0)
    y_true, y_pred = y_test, model.predict(x_test)
    return optunity.metrics.accuracy(y_true, y_pred)


@timeit
def get_optm_svm_pars(x, y, f, space, num_folds=10, num_iter=1):
    cv_decorator = optunity.cross_validated(x=x, y=y, num_folds=num_folds, num_iter=num_iter)
    f_decorated = cv_decorator(f)
    optimal_svm_pars, info, _ = optunity.maximize_structured(f_decorated, space, num_evals=50, pmap=optunity.pmap)
    print("Optimal parameters" + str(optimal_svm_pars))
    print("ACC of tuned SVM: %1.3f" % info.optimum)
    return optimal_svm_pars


def get_D_based_thresholds(LANG_T, MEAN_T, lang_prob, meaning_prob, y, print=False):
    scores = init_scores()
    for i, (p_lang, p_meaning) in enumerate(zip(lang_prob, meaning_prob)):
        if p_lang[0] >= LANG_T and p_meaning[0] >= MEAN_T:
            decision = 'accept'
        else:
            decision = 'reject'
        score_decision(decision, get_langauge_y(y)[i], get_meaning_y(y)[i], scores)
    if print:
        print_scores(scores)
    return get_D(scores)




if __name__ == '__main__':

 # year17 data
    with open('../data/processed/numpy/year17.pkl', 'rb') as pf:
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


    """
    Step 1:
    
    Call get_optm_svm_pars to find optimal SVM setups for both models
    """
    space = {'kernel': {'linear': {'C': [0, 2]},
                        'rbf': {'logGamma': [-5, 0], 'C': [0, 10]},
                        'poly': {'degree': [2, 5], 'C': [0, 5], 'coef0': [0, 2]}
                        }
             }
    MODEL_TUNE = True
    if not MODEL_TUNE: # for debug purpose, skip long model-param tweaking.
        optm_lang = {'kernel': 'rbf', 'C': 8.9892578125, 'coef0': None, 'degree': None, 'logGamma': -0.37353515625}
        optm_meaning = {'kernel': 'rbf', 'C': 5.686328124999999, 'coef0': None, 'degree': None, 'logGamma': -2.1400390624999996}
    else:
        optm_lang = get_optm_svm_pars(year17_lang_train_X, year17_lang_train_y, svm_tuned_acc, space)
        optm_meaning = get_optm_svm_pars(year17_meaning_train_X, year17_meaning_train_y, svm_tuned_acc, space)


    """
    Step 2:
    
    The following codes are for tuning the thresholds to map probs output by both language and meaning 
    SVM models to decisions (accept vs. reject).
    
    Note that due to D computation doesn't use numpy, its running speed is in fact slow. For example,
    compared with the time used for model training in the Step 1, tuning D uses 3 to 4 times long time.
    """
    # put svm_tuned_Dscore here so that can access both optm_lang and optm_meaning
    @optunity.cross_validated(x=year17_train_X, y=year17_train_y, num_folds=5)
    def svm_tuned_Dscore(x_train, y_train, x_test, y_test, lang_pars=optm_lang, meaning_pars=optm_meaning, L_thres=0.5,
                         M_thres=0.5):
        model_lang = get_svm_bypars(get_langauge_X(x_train), get_langauge_y(y_train), lang_pars)
        model_meaning = get_svm_bypars(get_meaning_X(x_train), get_meaning_y(y_train), meaning_pars)

        D_score = get_D_based_thresholds(L_thres, M_thres,
                                         model_lang.predict_proba(get_langauge_X(x_test)),
                                         model_meaning.predict_proba(get_meaning_X(x_test)),
                                         y_test)
        return D_score


    @timeit
    # this can use optunity's multi-thread and smarter solver
    def tune_D_smart():
        thres_grid = {'L_thres' :  [0.35, 0.65],
                       'M_thres' : [0.25, 0.55]}
        optm_thres, info, _ = optunity.maximize_structured(svm_tuned_Dscore,
                                                            num_evals=100,
                                                            pmap=optunity.pmap,
                                                            search_space=thres_grid)
        print("Optimal parameters" + str(optm_thres))
        print("best D after CV running: %2.3f" % info.optimum)
        return(optm_thres['L_thres'], optm_thres['M_thres'])


    # my own naive threshold tuning to obtain the largest D
    def tune_D_naive():
        import pylab as pl
        lang_thres = pl.frange(0.25, 0.50, 0.050)
        mean_thres = pl.frange(0.35, 0.75, 0.050)
        # find L and M thresholds to get max D
        max_D = 0
        max_L = 0
        max_M = 0
        for l in lang_thres:
            for m in mean_thres:
                d_now = svm_tuned_Dscore(L_thres=l, M_thres=m)
                print('L {}\t M {}:\t D {}'.format(l, m, d_now))
                if max_D < d_now:
                    max_D = d_now
                    max_L = l
                    max_M = m
            print('\n')
        print('Max D is {}'.format(max_D))
        print('based on max_L:{} max_M:{}'.format(max_L, max_M))
        return (max_L, max_M)


    # max_L, max_M = tune_D_naive()
    max_L, max_M = tune_D_smart()

    """
    Step 3:
    
    Use the optimized models and D-related thresholds to generate predictions on the 2017 test data set.
    """
    # apply to the year17 test set.
    model_lang = get_svm_bypars(get_langauge_X(year17_test_X), get_langauge_y(year17_test_y), optm_lang)
    model_meaning = get_svm_bypars(get_meaning_X(year17_test_X), get_meaning_y(year17_test_y), optm_meaning)

    D_score = get_D_based_thresholds(max_L, max_M,
                                     model_lang.predict_proba(get_langauge_X(year17_test_X)),
                                     model_meaning.predict_proba(get_meaning_X(year17_test_X)),
                                     year17_test_y, True)
