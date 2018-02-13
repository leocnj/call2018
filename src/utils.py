import logging, coloredlogs
import numpy as np


def get_logger(name, simple=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # using color in log
    coloredlogs.install(level='INFO')

    handler = logging.StreamHandler()
    handler.setLevel(logging.WARNING)
    if simple:
        formatter = logging.Formatter("[%(filename)s:%(lineno)s - %(levelname)s ] %(message)s")
    else:
        formatter = logging.Formatter("%(asctime)s [%(filename)s:%(lineno)s - "
                                      "%(funcName)s - %(levelname)s ] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

#
#
def get_langauge_y(y):
    return y[:,1]


def get_meaning_y(y):
    return y[:,0]


label_back = lambda x: 'correct' if x==1 else 'incorrect'

def get_D_on_df(df):
    scores = init_scores()
    for _, row in df.iterrows():
        decision = row['Judgement']
        lang_label = row['language']
        mean_label = row['meaning']
        score_decision(decision, lang_label, mean_label, scores)
    print_scores(scores)


def get_D_on_class(lang_pred, y, print=False, CR_adjust=False):
    scores = init_scores()
    for i, lang in enumerate(lang_pred):
        if lang:
            decision = 'accept'
        else:
            decision = 'reject'
        score_decision(decision, label_back(get_langauge_y(y)[i]), label_back(get_meaning_y(y)[i]), scores)
    if print:
        print_scores(scores)
    return get_D(scores, CR_adjust), get_ICRrate(scores), get_CRrate(scores)

def get_D_on_proba(lang_pred, y, THRES=0.5, print=False, CR_adjust=False):
    scores = init_scores()
    for i, lang in enumerate(lang_pred):
        if lang[1] >= THRES:
            decision = 'accept'
        else:
            decision = 'reject'
        score_decision(decision, label_back(get_langauge_y(y)[i]), label_back(get_meaning_y(y)[i]), scores)
    if print:
        print_scores(scores)
    return get_D(scores, CR_adjust), get_ICRrate(scores), get_CRrate(scores)


def cross_val_D(model, X, y, cv, THRES=0.5):
    """
    For the cv split plan, compute D values among all cv-folds
    :param model:
    :param X:
    :param y:
    :param cv:
    :param THRES: default 0.5
    :return: numpy array D values
    """
    Ds = []
    CRs = []
    ICRs = []
    for train_index, test_index in cv.split(X, get_langauge_y(y)):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        model.fit(X_train, get_langauge_y(y_train))
        y_pred = model.predict_proba(X_test)
        D, ICR, CR = get_D_on_proba(y_pred, y_test, THRES=THRES, CR_adjust=True)
        Ds.append(D)
        CRs.append(CR)
        ICRs.append(ICR)
    return np.asarray(Ds), np.asarray(ICRs), np.asarray(CRs)

# from CALL

def init_scores():
    return {'CorrectAccept': 0, 'GrossFalseAccept': 0, 'PlainFalseAccept': 0, 'CorrectReject': 0, 'FalseReject': 0}

# Compare decision with gold standard judgements for language and meaning
def score_decision(decision, language_correct_gs, meaning_correct_gs, scores):
    if ( decision == 'accept' and language_correct_gs == 'correct' ):
        result = 'CorrectAccept'
    elif ( decision == 'accept' and meaning_correct_gs == 'incorrect' ):
        result = 'GrossFalseAccept'
    elif ( decision == 'accept' ):
        result = 'PlainFalseAccept'
    elif ( decision == 'reject' and language_correct_gs == 'incorrect' ):
        result = 'CorrectReject'
    else:
        result = 'FalseReject'
    scores[result] = scores[result] + 1
    return result


def get_D(scores, CR_adjust):
    CA = scores['CorrectAccept']
    GFA = scores['GrossFalseAccept']
    PFA = scores['PlainFalseAccept']
    CR = scores['CorrectReject']
    FR = scores['FalseReject']
    k = 3

    FA = PFA + k * GFA
    Correct = CA + FR
    Incorrect = CR + GFA + PFA

    if (CR + FA) > 0:
        IncorrectRejectionRate = CR / (CR + FA)
    else:
        IncorrectRejectionRate = 'undefined'

    if (FR + CA) > 0:
        # CorrectRejectionRate = FR / (FR + CA)
        if CR_adjust:
            CorrectRejectionRate = FR / (FR + CA) if FR/(FR+CA) > 0.04 else 0.04  # penalize using low cRj to boost D
        else:
            CorrectRejectionRate = FR / (FR + CA)
    else:
        CorrectRejectionRate = 'undefined'

    if (CorrectRejectionRate != 'undefined' and IncorrectRejectionRate != 'undefined'):
        D = IncorrectRejectionRate / CorrectRejectionRate
    else:
        D = 'undefined'
    return D


def get_CRrate(scores):
    CA = scores['CorrectAccept']
    GFA = scores['GrossFalseAccept']
    PFA = scores['PlainFalseAccept']
    CR = scores['CorrectReject']
    FR = scores['FalseReject']
    k = 3

    FA = PFA + k * GFA
    Correct = CA + FR
    Incorrect = CR + GFA + PFA

    if (CR + FA) > 0:
        IncorrectRejectionRate = CR / (CR + FA)
    else:
        IncorrectRejectionRate = 'undefined'

    if (FR + CA) > 0:
        CorrectRejectionRate = FR / (FR + CA)
        # CorrectRejectionRate = FR / (FR + CA) if FR/(FR+CA) > 0.04 else 0.04  # penalize using low cRj to boost D
    else:
        CorrectRejectionRate = 'undefined'

    return CorrectRejectionRate


def get_ICRrate(scores):
    CA = scores['CorrectAccept']
    GFA = scores['GrossFalseAccept']
    PFA = scores['PlainFalseAccept']
    CR = scores['CorrectReject']
    FR = scores['FalseReject']
    k = 3

    FA = PFA + k * GFA
    Correct = CA + FR
    Incorrect = CR + GFA + PFA

    if (CR + FA) > 0:
        IncorrectRejectionRate = CR / (CR + FA)
    else:
        IncorrectRejectionRate = 'undefined'

    if (FR + CA) > 0:
        CorrectRejectionRate = FR / (FR + CA)
        # CorrectRejectionRate = FR / (FR + CA) if FR/(FR+CA) > 0.04 else 0.04  # penalize using low cRj to boost D
    else:
        CorrectRejectionRate = 'undefined'

    return IncorrectRejectionRate

def print_scores(scores):
    CA = scores['CorrectAccept']
    GFA = scores['GrossFalseAccept']
    PFA = scores['PlainFalseAccept']
    CR = scores['CorrectReject']
    FR = scores['FalseReject']
    k = 3

    FA = PFA + k * GFA
    Correct = CA + FR
    Incorrect = CR + GFA + PFA

    if (CR + FA) > 0:
        IncorrectRejectionRate = CR / (CR + FA)
    else:
        IncorrectRejectionRate = 'undefined'

    if (FR + CA) > 0:
        CorrectRejectionRate = FR / (FR + CA)
        # CorrectRejectionRate = FR / (FR + CA) if FR / (FR + CA) > 0.04 else 0.04  # penalize using low cRj to boost D
    else:
        CorrectRejectionRate = 'undefined'

    if (CorrectRejectionRate != 'undefined' and IncorrectRejectionRate != 'undefined'):
        D = IncorrectRejectionRate / CorrectRejectionRate
    else:
        D = 'undefined'

    print('\nINCORRECT UTTERANCES (' + str(Incorrect) + ')')
    print('CorrectReject    ' + str(CR))
    print('GrossFalseAccept ' + str(GFA) + '*' + str(k) + ' = ' + str(GFA * k))
    print('PlainFalseAccept ' + str(PFA))
    print('RejectionRate    ' + two_digits(IncorrectRejectionRate))

    print('\nCORRECT UTTERANCES (' + str(Correct) + ')')
    print('CorrectAccept    ' + str(CA))
    print('FalseReject      ' + str(FR))
    print('RejectionRate    ' + two_digits(CorrectRejectionRate))

    print('\nD                ' + two_digits(D))


def two_digits(x):
    if x == 'undefined':
        return 'undefined'
    else:
        return ("%.2f" % x)


# https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
#
import time
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
