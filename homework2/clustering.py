# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 01:56:53 2020

@author: morri
"""
import pandas as pd
from sklearn import metrics
from sklearn import cluster
from sklearn import preprocessing
#
data_origin = preprocessing.scale(pd.read_csv('Clustering/csv/origin/cluster_train.csv').values)

data_5f = (pd.read_csv('Clustering/csv/lsa/LSA_5Feature.csv').values)
data_10f = (pd.read_csv('Clustering/csv/lsa/LSA_10Feature.csv').values)
data_20f = (pd.read_csv('Clustering/csv/lsa/LSA_20Feature.csv').values)
data_40f = (pd.read_csv('Clustering/csv/lsa/LSA_40Feature.csv').values)

#%%
clustering_origin = cluster.KMeans(n_clusters=6).fit(data_origin)
clustering_5f = cluster.KMeans(n_clusters=6).fit(data_5f)
clustering_10f = cluster.KMeans(n_clusters=6).fit(data_10f)
clustering_20f = cluster.KMeans(n_clusters=6).fit(data_20f)
clustering_40f = cluster.KMeans(n_clusters=6).fit(data_40f)

#%
score_origin = metrics.silhouette_score(data_origin, clustering_origin.labels_, metric='euclidean')
score_5f = metrics.silhouette_score(data_5f, clustering_5f.labels_, metric='euclidean')
score_10f = metrics.silhouette_score(data_10f, clustering_10f.labels_, metric='euclidean')
score_20f = metrics.silhouette_score(data_20f, clustering_20f.labels_, metric='euclidean')
score_40f = metrics.silhouette_score(data_40f, clustering_40f.labels_, metric='euclidean')

#%%
from print_answer import printOutAnswer
printOutAnswer("clustering",clustering_5f.labels_)

#%%
count = [0,0,0,0,0,0]
labels = clustering_5f.labels_
for i in range(len(labels)):
    count[labels[i]]+=1