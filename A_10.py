import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = sns.load_dataset('iris')
data

data.head()

Iris.describe()

data.describe(include = 'object')

data.isnull().sum()

print("\n\nThe features in the dataset are as follows : ")
print("1. Sepal length : ", data['sepal_length'].dtype)
print("2. Sepal width : ", data['sepal_width'].dtype)
print("3. Petal length : ", data['petal_length'].dtype)
print("4. Petal width : ", data['petal_width'].dtype)
print("5. Species : ", data['species'].dtype)

sns.histplot(x = data['sepal_length'], kde=True)

sns.histplot(x = data['sepal_width'], kde=True)

sns.histplot(x = data['petal_length'], kde=True)

sns.histplot(x = data['petal_width'], kde=True)

sns.boxplot(data['sepal_length'])

sns.boxplot(data['sepal_width'])

sns.boxplot(data['petal_length'])

sns.boxplot(data['petal_width'])

sns.boxplot(x='sepal_length',y='species',data=data)

sns.boxplot(x='petal_length',y='species',data=data)
