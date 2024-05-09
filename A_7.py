import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from nltk.tokenize import word_tokenize
text='Real madrid is set to win the UCL for the season . Benzema might win Balon dor . Salah might be the runner up'
word_token = word_tokenize(text)
print(word_token)

from nltk.tokenize import sent_tokenize
sent_token = sent_tokenize(text)
print(sent_token)

from nltk.corpus import stopwords
stop = stopwords.words('english')
arr =[]
for w in word_token:
    if w not in stop:
        arr.append(w)
print(arr)

# nltk.download('averaged_perceptron_tagger')
nltk.pos_tag(word_token)

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
lemmatize = WordNetLemmatizer()
ps = PorterStemmer()
arr_stem = []
for w in word_token:
    stem = ps.stem(w)
    arr_stem.append(stem)
arr_stem

arr_lamet = []
for w in word_token:
    lam = lemmatize.lemmatize(w)
    arr_lamet.append(lam) 
arr_lamet