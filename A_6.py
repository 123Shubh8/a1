import pandas as pd
import numpy as np
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, mean_squared_error
from sklearn.naive_bayes import GaussianNB

df2 = sns.load_dataset('iris')
df2.columns

x = df2[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df2['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

model2 = GaussianNB()

model2.fit(x_train, y_train)

y_predict = model2.predict(x_test)

a = accuracy_score(y_test, y_predict)
a

cm = confusion_matrix(y_test, y_predict)
cm

tp = cm[0][0]
fn = cm[0][1] + cm[0][2]
fp = cm[1][0] + cm[2][0]
tn = cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2]

print(tp, fn, fp, tn)