import pandas as pd
import numpy as np
import sys

if len(sys.argv) < 2 or ((sys.argv[-1] != 'yann') and (sys.argv[-1] != 'benjam')):
    raise NameError("enter 'yann' or 'benjam'")

if sys.argv[-1] == 'yann':
    N=[i for i in range(100)]
else:
    N=[i for i in range(100,200)]

df_test = pd.read_csv("../data/test.csv")

print("Type 1 or 0 then enter, press enter without pressing any number to skip if crash occured previously or something")

for k in N:
    print("review number: " + str(k) + "\n")
    print(df_test.review.iloc[k])
    score = input("\nEnter appreciation: 1 for positive and 0 for negative:\n")
    if score:
        score = np.float(int(score))
        df_test.score.iloc[k] = score
    df_test.to_csv(path_or_buf ='../data/test.csv', index=False)
