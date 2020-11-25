# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 01:59:48 2020

@author: morri
"""
import pandas as pd
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('classfication/csv/train.csv').values
test_data = pd.read_csv('classfication/csv/test.csv').values

#%%
data = preprocessing.scale(train_data[:,:-1])
X_train, X_test, y_train, y_test = train_test_split(
    data, train_data[:,-1], test_size=0.1, random_state=0)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
validation = clf.score(X_test, y_test)

#%%
from print_answer import printOutAnswer
data = preprocessing.scale(test_data)
answer = clf.predict(data)
printOutAnswer("classification",answer)