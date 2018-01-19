import logging, coloredlogs


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


def get_D(scores):
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
    else:
        CorrectRejectionRate = 'undefined'

    if (CorrectRejectionRate != 'undefined' and IncorrectRejectionRate != 'undefined'):
        D = IncorrectRejectionRate / CorrectRejectionRate
    else:
        D = 'undefined'
    return D


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
