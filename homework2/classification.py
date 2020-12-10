# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 01:59:48 2020

@author: morri
"""
import pandas as pd
from sklearn import preprocessing
from sklearn import tree#DecisionTreeClassifier
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('classfication/csv/train.csv').values
test_data = pd.read_csv('classfication/csv/test.csv').values

#%%
data = preprocessing.scale(train_data[:,:-1])
X_train, X_test, y_train, y_test = train_test_split(
    data, train_data[:,-1], test_size=0.25,random_state=0)
clf = tree.DecisionTreeClassifier(
    criterion='entropy',
    class_weight='balanced',
    random_state=0,
    min_samples_split=5,
    max_depth=15)
clf = clf.fit(X_train, y_train)
validation=clf.score(X_test, y_test)

#%%
from print_answer import printOutAnswer
data = preprocessing.scale(test_data)
answer = clf.predict(data)
printOutAnswer("classification",answer)

#%%
count = [0,0,0,0]
for i in range(len(train_data[:,-1])):
    count[train_data[:,-1][i]]+=1