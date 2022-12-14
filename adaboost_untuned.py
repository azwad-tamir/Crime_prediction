# -*- coding: utf-8 -*-
"""ChicagoAdaBoost.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17F2Elg_TNtsepZaWrZ7NKfeYsaZcLjNX
"""

#Import Libraries
import numpy as np
import pandas as pd

#Import Graph Libraries
from matplotlib import style
from collections import Counter

#Import Machine Learning Methods
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

#Load Data
df = pd.read_csv('drive/My Drive/CAP5610/Group Project/Data/total_df.csv', error_bad_lines=False)

print(df.columns.values)

df = df.drop(['Unnamed: 0'], axis=1)

print(df.columns.values)

X = df.drop(['Arrest'], axis=1)
Y = df['Arrest']

print(X.value_counts)
print(Y.value_counts)

#Cross-validation 5-fold
folds = StratifiedKFold(n_splits = 5, shuffle = True)

#Getting all the scores from the cross validation
accuracy_scores_AdaBoost = []
precision_scores_AdaBoost= []
recall_scores_AdaBoost = []
f1_scores_AdaBoost = []

for train_index, test_index in folds.split(X, Y):
  X_train, X_test, Y_train, Y_test = X.iloc[train_index], X.iloc[test_index], Y.iloc[train_index], Y.iloc[test_index]

  #Use AdaBoost
  boost = AdaBoostClassifier()
  boost.fit(X_train, Y_train)

  predBoost = boost.predict(X_test)

  accuracy_scores_AdaBoost.append(accuracy_score(Y_test, predBoost))
  precision_scores_AdaBoost.append(precision_score(Y_test, predBoost))
  recall_scores_AdaBoost.append(recall_score(Y_test, predBoost))
  f1_scores_AdaBoost.append(f1_score(Y_test, predBoost))

#Print Average Scores for AdaBoost
print('---- Average Scores ----')
print('Accuracy Score: ', np.average(accuracy_scores_AdaBoost))
print('Precision Score: ', np.average(precision_scores_AdaBoost))
print('Recall Score: ', np.average(recall_scores_AdaBoost))
print('F1 Score: ', np.average(f1_scores_AdaBoost))
print()