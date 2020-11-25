# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 01:56:53 2020

@author: morri
"""
import pandas as pd
from sklearn import cluster
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

data_origin = preprocessing.scale(pd.read_csv('Clustering/csv/origin/cluster_train.csv').values)

data_5f = preprocessing.scale(pd.read_csv('Clustering/csv/lsa/LSA_5Feature.csv').values)
data_10f = preprocessing.scale(pd.read_csv('Clustering/csv/lsa/LSA_10Feature.csv').values)
data_20f = preprocessing.scale(pd.read_csv('Clustering/csv/lsa/LSA_20Feature.csv').values)
data_40f = preprocessing.scale(pd.read_csv('Clustering/csv/lsa/LSA_40Feature.csv').values)

#%%
kmeans_origin = cluster.KMeans(n_clusters=6, random_state=0).fit(data_origin).labels_
kmeans_5f = cluster.KMeans(n_clusters=6, random_state=0).fit(data_5f).labels_
kmeans_10f = cluster.KMeans(n_clusters=6, random_state=0).fit(data_10f).labels_
kmeans_20f = cluster.KMeans(n_clusters=6, random_state=0).fit(data_20f).labels_
kmeans_40f = cluster.KMeans(n_clusters=6, random_state=0).fit(data_40f).labels_

#%%
from print_answer import printOutAnswer
printOutAnswer("clustering",kmeans_origin)