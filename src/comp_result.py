import argparse

import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix


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
        return ("%.4f" % x)

def get_D_on_df(df):
    scores = init_scores()
    for _, row in df.iterrows():
        decision = row['Judgement']
        lang_label = row['language']
        mean_label = row['meaning']
        score_decision(decision, lang_label, mean_label, scores)
    print_scores(scores)


if __name__ == '__main__':

    # df = pd.read_csv(sys.argv[1], delimiter='\t')
    make_judge = lambda x: 1 if x == 'accept' or x == 'correct' else 0
    parser = argparse.ArgumentParser()
    parser.add_argument('df_file', type=str)
    parser.add_argument('--year', type=str, help='which test file', default='2018')
    args = parser.parse_args()
    df = pd.read_csv(args.df_file, delimiter='\t')
    if (args.year == '2018'):
        # load 2018 test judgement
        scores = pd.read_csv('../data/texttask_trainData/testDataWithJudgements.csv', delimiter='\t')
        # always use language and meaning
        scores.rename(columns={'FullyAcceptable': 'language',
                               'MeaningAcceptable': 'meaning'}, inplace=True)
        df = pd.merge(df, scores, on='Id')
    df['pred_lang'] = [make_judge(x) for x in df['Judgement']]
    df['true_lang'] = [make_judge(x) for x in df['language']]

    # F1
    f1 = f1_score(df['true_lang'], df['pred_lang'], average='macro')
    print('f1 using f1_score average=macro: {}'.format(f1))

    cfm = confusion_matrix(df['true_lang'], df['pred_lang'])
    print('confusion matrix:\n')
    print(cfm)

    report = classification_report(df['true_lang'], df['pred_lang'])
    print(report)

    D = get_D_on_df(df)

