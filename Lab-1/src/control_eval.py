import numpy as np
import pandas as pd
import os

def main():
    df_eval = pd.read_csv('../data/evaluation.csv', engine='python')

    for index, row in df_eval.iterrows():
        os.system('cls' if os.name == 'nt' else 'clear')
        print(row['review'])
        print()
        print(row['prediction'])
        print()
        print()
        input("Press Enter to continue...")


if __name__ == '__main__':
    main()
