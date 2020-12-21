# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 16:48:09 2020

@author: Amith Lakkakula
"""

import json
import itertools

with open('./TestFiles/datasetforclustertesting/english-12-150.json', encoding="utf8") as f:
    data = json.load(f)
    
english_files = ['english-12-150.json', 'english-13-227.json', 'english-14-261.json', 'english-15-178.json']

english_titles = []

for eng_file in english_files:
    with open(f'./TestFiles/datasetforclustertesting/{eng_file}', encoding="utf8") as f:
        data = json.load(f)
        english_titles += data['Titles']
        
ta_files = ['tamil-12-485', 'tamil-13-667', 'tamil-14-767', 'tamil-15-728']

ta_titles = []

for ta_file in ta_files:
    with open(f'./TestFiles/datasetforclustertesting/{ta_file}.json', encoding="utf8") as f:
        data = json.load(f)
        ta_titles += data['Titles']
#%%        
# Simulating realtime batches

en_batches=[]
for i in range(0,816, 7):
    en_batches.append(english_titles[i:i+7])

en_batches_daywise= []
for i in range(0,117,39):
    temp = []
    for j in range(i,i+39):
        temp.append(list(itertools.chain.from_iterable(en_batches[i:j+1])))
    en_batches_daywise.append(temp)
    
ta_batches=[]
for i in range(0,2657, 23):
    ta_batches.append(ta_titles[i:i+23])

ta_batches_daywise= []
for i in range(0,116,39):
    temp = []
    for j in range(i,i+39):
        temp.append(list(itertools.chain.from_iterable(ta_batches[i:j+1])))
    ta_batches_daywise.append(temp)
    
#%%

keys = ['day1', 'day2', 'day3']
result = {}
with open('en_summary_class_results.json', 'w', encoding="utf8") as f:
    for idx, key in enumerate(keys):
        result[key] = json.loads(daywise_result[idx])
    json.dump(result, f, ensure_ascii=False, indent=4)
    
#%%