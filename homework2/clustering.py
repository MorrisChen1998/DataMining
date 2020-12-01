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
data_origin = (pd.read_csv('Clustering/csv/origin/cluster_train.csv').values)

data_5f = (pd.read_csv('Clustering/csv/lsa/LSA_5Feature.csv').values)
data_10f = (pd.read_csv('Clustering/csv/lsa/LSA_10Feature.csv').values)
data_20f = (pd.read_csv('Clustering/csv/lsa/LSA_20Feature.csv').values)
data_40f = (pd.read_csv('Clustering/csv/lsa/LSA_40Feature.csv').values)

#%%
def clustering(data):
    return cluster.KMeans(n_clusters=6).fit(data)
def silhouetteScore(data,label):
    return metrics.silhouette_score(data, label, metric='euclidean')
    
clustering_origin = clustering(data_origin)
clustering_5f = clustering(data_5f)
clustering_10f = clustering(data_10f)
clustering_20f = clustering(data_20f)
clustering_40f = clustering(data_40f)

score_origin = silhouetteScore(data_origin,clustering_origin.labels_)
score_5f = silhouetteScore(data_5f, clustering_5f.labels_)
score_10f = silhouetteScore(data_10f, clustering_10f.labels_)
score_20f = silhouetteScore(data_20f, clustering_20f.labels_)
score_40f = silhouetteScore(data_40f, clustering_40f.labels_)
print('5f=%.3f'%score_5f)
print('10f=%.3f'%score_10f)
print('20f=%.3f'%score_20f)
print('40f=%.3f'%score_40f)
print('origin(60f)=%.3f'%score_origin)

#%%
from print_answer import printOutAnswer
printOutAnswer("clustering",clustering_5f.labels_)

#%%
count = [0,0,0,0,0,0]
labels = clustering_5f.labels_
for i in range(len(labels)):
    count[labels[i]]+=1