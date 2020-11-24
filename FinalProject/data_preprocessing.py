# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 22:16:33 2020

@author: morri
"""
import pandas as pd

training_data = pd.read_json('yelp_training_review.json',lines=True)
training_data = training_data.drop(columns=['votes','date','type'])
#user_list = list(pd.read_json('yelp_training_user.json',lines=True)['user_id'])
#business_list = list(pd.read_json('yelp_training_business.json',lines=True)['business_id'])
training_data = pd.pivot_table(training_data, values='stars', index=['user_id'], columns=['business_id'], aggfunc='mean')
training_data.fillna(0,inplace=True)