import pandas as pd
import sklearn as sc

df = pd.read_csv("scotch.csv")

out_df = df.tail(3).copy()
print(df.tail(3))

out_df = out_df.reset_index(drop=True, inplace=True)
df = df.iloc[:-3]

print(out_df)
