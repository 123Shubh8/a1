import pandas as pd
import numpy as np
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, mean_squared_error
from sklearn.linear_model import LogisticRegression

df1 = pd.read_csv("Social_Network_Ads.csv")
df1.columns
df1['Gender'].replace({'Female':0, 'Male':1}, inplace=True)
x = df1[['User ID', 'Gender', 'Age', 'EstimatedSalary']]
y = df1['Purchased']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

model1 = LogisticRegression()

model1.fit(x_train, y_train)

y_predict = model1.predict(x_test)

a = accuracy_score(y_test, y_predict)
a

e = 1-a
e

tn, fn, fp, tp = confusion_matrix(y_test, y_predict).ravel()

p = precision_score(y_test, y_predict)
p