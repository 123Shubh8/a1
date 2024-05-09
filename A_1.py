import pandas as pd
import numpy as np

df = pd.read_csv("titanic.csv")
df.head()

df.isnull()

df.isnull().sum()

df.notnull().sum()

df.describe()

df.size

df.ndim

df.shape

df.info()

df = df.dropna()
df

df["fare"] = df["fare"].astype(int)
df

df['sex'].replace({"female":0,"male":1}, inplace=True)
df

