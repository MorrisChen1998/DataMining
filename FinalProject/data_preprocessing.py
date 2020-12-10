# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:16:33 2020

@author: morri
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.sparse import coo_matrix, save_npz, load_npz

#%%
training_data = pd.read_json('yelp_training_review.json',lines=True).drop(columns=['votes','date','type','review_id'])
user_list = list(pd.read_json('yelp_training_user.json',lines=True)['user_id'])
business_list = list(pd.read_json('yelp_training_business.json',lines=True)['business_id'])

# training_data = pd.pivot_table(training_data, values='stars', index=['user_id'], columns=['business_id'], aggfunc='mean')
# training_data.fillna(0,inplace=True)

#%%

for i in tqdm(range(len(user_list))):
    training_data = training_data.replace(user_list[i], i)
for i in tqdm(range(len(business_list))):
    training_data = training_data.replace(business_list[i], i)
    
                           #data
rating_matrix = coo_matrix(np.array(list(training_data['stars'])),
                 #row,col
                (np.array(list(training_data['user_id'])), np.array(list(training_data['business_id']))))

save_npz('rating_matrix.npz', rating_matrix)

#%%
rating_matrix = load_npz('rating_matrix.npz')