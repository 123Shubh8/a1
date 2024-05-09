import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#loading dataset
data = sns.load_dataset('titanic')
data.head()

data.describe()

data.info()

data.isnull().sum()

data['age'] = data['age'].fillna(np.mean(data['age']))
data['embarked'] = data['embarked'].fillna(data['embarked'].mode()[0])
data['embark_town'] = data['embark_town'].fillna(data['embark_town'].mode()[0])

data.isnull().sum()

sns.boxplot(x='sex', y='age', hue='survived', data=data)

