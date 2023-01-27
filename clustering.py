#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 11:37:46 2022

@author: sopsla
"""
import os
import numpy as np
import pandas as pd
import sklearn
import scipy.stats as stats
import nltk

import random
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# %% open logfile
PATH = '/'
mtype='scrambled' # 'structured' #
log = open(f'{PATH}/clustering-{mtype}-tmp.txt', 'w')
  
# %%             
def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

# %% load the results from both languages

if mtype == 'scrambled':
    log.write('loading scrambled surprisal values\n')
    spanish = pd.read_csv('/project/3027007.06/simulations/results/spanish/116044-spanish-random-model-1layer-results-10000sents.csv')
    english = pd.read_csv('/project/3027007.06/simulations/results/english/118311-english-random-model-1layer-results-10000sents.csv')
else:
    log.write('loading structured surprisal values\n')
    spanish = pd.read_csv('/project/3027007.06/simulations/results/spanish/116044-spanish-model-1layer-results-10000sents.csv')
    english = pd.read_csv('/project/3027007.06/simulations/results/english/118311-english-model-1layer-results-10000sents.csv')
    
# %%
if len(english)>len(spanish):
    log.write(f'cropping the english results to {len(spanish)} words\n')
    english = english[:spanish.shape[0]]
elif len(spanish)>len(english):
    log.write(f'cropping the spanish results to {len(spanish)} words\n')
    spanish = spanish[:english.shape[0]]

log.write('normalizing to avoid clustering on basis of one being more surprising than the other\n')
spanish['zscored'] = stats.zscore(spanish['target_surprisal'])
english['zscored'] = stats.zscore(english['target_surprisal'])

log.write('rounding to 1 decimal to avoid clustering on the basis of highly specific values for one and not the other language\n')
spanish['zscored_round'] = spanish['zscored'].round(decimals=1)
english['zscored_round'] = english['zscored'].round(decimals=1)

# %% correlation
from scipy.stats import pearsonr

df = pd.concat([spanish['zscored'], english['zscored']], axis=1)

rho = df.corr()
pval = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x<=t]))
print(rho.round(2).astype(str) + p)


# %% plot the surprisal values
fig,ax=plt.subplots()
ax.hist(english['zscored'], bins=200, alpha=0.5)
ax.hist(spanish['zscored'], bins=200, alpha=0.5)
plt.legend(['english', 'spanish'])
fig.savefig(f'/project/3027007.06/simulations/results/clustering-{mtype}-zscored.svg')

fig,ax=plt.subplots()
ax.hist(english['zscored_round'], bins=200, alpha=0.5)
ax.hist(spanish['zscored_round'], bins=200, alpha=0.5)
plt.legend(['english', 'spanish'])
fig.savefig(f'/project/3027007.06/simulations/results/clustering-{mtype}-zscored_round.svg')

# %% are the dists different?

# yes
#print(stats.ttest_ind(spanish['target_surprisal'], english['target_surprisal']))

# get most frequent values
#stats.mode(english['zscored'])
#Out[56]: ModeResult(mode=array([-0.79960329]), count=array([934]))

#stats.mode(spanish['zscored'])
#Out[57]: ModeResult(mode=array([-0.98609135]), count=array([852]))

#clf.predict(np.array([-0.79960329,-0.98609135]).reshape(-1,1))
#Out[59]: array(['english', 'spanish'], dtype='<U7')

# see where the most frequent values come from
modes = {}
for lang, lname in zip([spanish,english],['spanish', 'english']):
    
    val = stats.mode(lang['target_surprisal'])[0][0]
    wrds = lang.loc[lang['target_surprisal'] == val, 'target']
    cont = lang.loc[lang['target_surprisal'] == val, 'context']

    modes[lname] = val
    
    log.write(f'Mode of {lname}: {val}, occurs {len(wrds)} times\n')
    log.write(f'{lname} target is: {list(set(wrds))[0]}\n')
    log.write(f'{lname} context is: {list(set(cont))[0]}\n')

# %% sets
for typ in ['zscored', 'zscored_round']:
    es_sort = set(spanish['zscored_round'])
    en_sort = set(english['zscored_round'])
    inters = es_sort.intersection(en_sort)
    
    log.write(f'{len(inters)} overlapping values in {typ}. Values: {inters}\n')
    log.write(f'unique values: {len(en_sort-es_sort)}\n')

#print(sum(np.array(es_sort) == np.array(en_sort)) / len(es_sort))

# %% in a looopppp
log.write(f'\n\nClassifying the {mtype} results for ngram sizes of 1 to 10...\n')

ngrams = {}
classified = {}

for typ in ['zscored', 'zscored_round']:
    log.write(f'type: {typ}\n')

    for ngram_len in range(1, 11):
        log.write(f'Context length: {ngram_len}\n')
        if ngram_len == 1:
            
            ngram_es = np.asarray(spanish[typ], dtype=float)
            ngram_en = np.asarray(english[typ], dtype=float) #ngram_es.copy() #
            
            ngram_en = ngram_en[:ngram_es.shape[0]]
            data = np.hstack([ngram_en, ngram_es]).reshape(-1,1) # surprisal values per ngrams
            
        else:
            ngram_es = np.asarray(list(nltk.ngrams(spanish[typ], ngram_len, pad_left=False, pad_right=False)), dtype=float)
            ngram_en = np.asarray(list(nltk.ngrams(english[typ], ngram_len, pad_left=False, pad_right=False)), dtype=float)
            
            ngram_en = ngram_en[:ngram_es.shape[0],:]
        
            # structure the data
            data = np.vstack([ngram_en, ngram_es]) # surprisal values per ngrams
            
        labels = np.asarray(['english']*len(ngram_en) + ['spanish']*len(ngram_es))
        idx = np.random.rand(data.shape[0]).argsort(axis=0)
        
        data = np.take_along_axis(data, np.expand_dims(idx, 1), axis=0)
        labels = np.take_along_axis(labels, idx, axis=0)
        
        train, test, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
        
        # random forest classifier
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(train, train_labels)
        
        # predict and calculate accuracy
        prediction = clf.predict(test)
        acc = clf.score(test, test_labels)
            
        correctly_classified = []
        for test_val,pred,target in zip(test, prediction, test_labels):
            if pred == target:
                correctly_classified.append((test_val,target))
                
        classified[ngram_len] = correctly_classified
        ngrams[ngram_len] = data, labels
    
                
        log.write(f'Classifier was correct {acc*100} percent of the time.\n')
    # on 500 sentences: interestingly, the classifier was correct 68% of the time

# %% # %% removing the mode
log.write('test: remove the mode. Does it still work?\n')
spanish = spanish[~np.array(spanish['target_surprisal'] == modes['spanish'])]
english = english[~np.array(english['target_surprisal'] == modes['english'])]   

log.write(f'\n\nClassifying the {mtype} results for ngram sizes of 1 to 10...\n')

ngrams = {}
classified = {}

for typ in ['zscored', 'zscored_round']:
    log.write(f'type: {typ}\n')

    for ngram_len in range(1, 11):
        log.write(f'Context length: {ngram_len}\n')
        if ngram_len == 1:
            
            ngram_es = np.asarray(spanish[typ], dtype=float)
            ngram_en = np.asarray(english[typ], dtype=float) #ngram_es.copy() #
            
            ngram_en = ngram_en[:ngram_es.shape[0]]
            data = np.hstack([ngram_en, ngram_es]).reshape(-1,1) # surprisal values per ngrams
            
        else:
            ngram_es = np.asarray(list(nltk.ngrams(spanish[typ], ngram_len, pad_left=False, pad_right=False)), dtype=float)
            ngram_en = np.asarray(list(nltk.ngrams(english[typ], ngram_len, pad_left=False, pad_right=False)), dtype=float)
            
            ngram_en = ngram_en[:ngram_es.shape[0],:]
        
            # structure the data
            data = np.vstack([ngram_en, ngram_es]) # surprisal values per ngrams
            
        labels = np.asarray(['english']*len(ngram_en) + ['spanish']*len(ngram_es))
        idx = np.random.rand(data.shape[0]).argsort(axis=0)
        
        data = np.take_along_axis(data, np.expand_dims(idx, 1), axis=0)
        labels = np.take_along_axis(labels, idx, axis=0)
        
        train, test, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
        
        # random forest classifier
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(train, train_labels)
        
        # predict and calculate accuracy
        prediction = clf.predict(test)
        acc = clf.score(test, test_labels)
            
        correctly_classified = []
        for test_val,pred,target in zip(test, prediction, test_labels):
            if pred == target:
                correctly_classified.append((test_val,target))
                
        classified[ngram_len] = correctly_classified
        ngrams[ngram_len] = data, labels
    
                
        log.write(f'Classifier was correct {acc*100} percent of the time.')
        # on 500 sentences: interestingly, the classifier was correct 68% of the time

# %% close the logfile
log.close()