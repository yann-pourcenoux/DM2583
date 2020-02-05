import numpy as np
import pandas as pd
import os
import sys

def main(args):
    df_eval = pd.read_csv('../data/evaluation_cleaning.csv', engine='python')

    if len(args)==2:
        df_eval = df_eval[df_eval.prediction==args[-1]]

    for index, row in df_eval.iterrows():
        os.system('cls' if os.name == 'nt' else 'clear')
        print(row['review'])
        print()
        print(row['prediction'])
        print()
        print()
        input("Press Enter to continue...")


if __name__ == '__main__':
    args = sys.argv
    main(args)
