# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 18:40:05 2020

@author: morris
"""
import pandas as pd

records = pd.read_csv('records.csv')

#%%
count = records.sum() #count each item fq

support = 0.02

asscoMetric = 'confidence'
threshold = 0.02

#%%
from mlxtend.frequent_patterns import apriori

fqItems = apriori(records, min_support = support, use_colnames=True)

#%%
from mlxtend.frequent_patterns import association_rules

rules = association_rules(fqItems, metric = asscoMetric, min_threshold = threshold)

