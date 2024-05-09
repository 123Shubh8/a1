import pandas as pd
import numpy as np
import seaborn as sns

df = sns.load_dataset("iris")
df

df.describe()

df["sepal_length"].describe()

df.groupby("species").describe().sum()

