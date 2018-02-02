import pandas as pd
import argparse
import pickle

if __name__ == '__main__':

    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('--ctm', type=str, help='CTM file', required=True)
    parser.add_argument('--header', dest='header', action='store_true')
    parser.add_argument('--no-header', dest='header', action='store_false')
    parser.set_defaults(header=False)
    parser.add_argument('--getId', dest='getId', action='store_true')
    parser.add_argument('--no-getId', dest='getId', action='store_false')
    parser.set_defaults(getId=True)
    parser.add_argument('--conf', help='include confidence scores', action='store_true')
    args = parser.parse_args()

    df = pd.read_csv(args.ctm, sep=' ', header=None)

    def ctm_test(df):
        df.columns = ['Id', 'chan', 'start', 'end', 'word', 'conf', 'foo']
        # group by Id
        if args.header:
            print('Id,RecResult,Confidence')
        for name, group in df.groupby('Id'):
            sent = ' '.join([w.lower() for w in group['word']])
            cfs = ' '.join([repr(cf) for cf in group['conf']])
            if args.conf:
                print('{},{},{}'.format(name, sent, cfs))
            else:
                print('{},{}'.format(name, sent))


    def ctm_train(df):
        """
        need fetch Id based on wav
        """
        df.columns = ['Wavfile', 'chan', 'start', 'end', 'word', 'conf', 'foo']
        df['Wavfile'] = df['Wavfile'].apply(lambda x: x+'.wav')
        #
        with open('../processed/data.pkl', 'rb') as pf:
            objs = pickle.load(pf)

        df_18_A_train = objs[3]
        df_18_B_train = objs[4]
        df_18_C_train = objs[5]
        cols = ['Id', 'Wavfile']
        df_18_train = pd.concat([df_18_A_train[cols],
                                 df_18_B_train[cols],
                                 df_18_C_train[cols]])
        df = pd.merge(df_18_train, df, on='Wavfile')
        # group by Id
        if args.header:
            print('Id,RecResult,Confidence')
        for name, group in df.groupby('Id'):
            sent = ' '.join([w.lower() for w in group['word']])
            cfs = ' '.join([repr(cf) for cf in group['conf']])
            if args.conf:
                print('{},{},{}'.format(name, sent, cfs))
            else:
                print('{},{}'.format(name, sent))


    if args.getId:
        ctm_train(df)
    else:
        ctm_test(df)