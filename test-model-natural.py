#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 12:55:11 2022

@author: sopsla
"""
import os
import pickle 
import gc
import sys

import numpy as np
import pandas as pd
import time

import torch
from predictor import prepare_sequence

# %% device - to use GPU
if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu" 
  
device = torch.device(dev) 

# %% path
PATH = '/project/3027007.06/simulations'

# spanish: 116044
# english: 118311

languages = {'spanish': 116044,
             'english': 118311}

language = 'english'
nr = languages[language]

# select one
mname = f'{nr}-{language}-model-1layer.pkl'

# %% open vocabulary
with open(f'{PATH}/OpenSubtitles/{language}-vocab.pkl', 'rb') as f:
    vocabulary = pickle.load(f)

#%% load the test set & model
with open(os.path.join(PATH, f'OpenSubtitles/{language}-test.pkl'), 'rb') as f:
    test = pickle.load(f) 

model = torch.load(f=f'{PATH}/results/{language}/{mname}')

# %% indices
# if we have calculated them already
if os.path.isfile(f'{PATH}/OpenSubtitles/{language}-test-idx.pkl'):
    print('Loading indices')
    with open(f'{PATH}/OpenSubtitles/{language}-test-idx.pkl', 'rb') as f:
        randomidx = pickle.load(f)

else:
    # take random part of the test set of 1000 sentences for intermediate testing 
    randomidx = np.random.randint(0, len(test), 1000)
    
    # save the indices
    with open(f'{PATH}/OpenSubtitles/{language}-test-idx.pkl', 'wb') as f:
        pickle.dump(randomidx, f)
    
# %% get the test set & calc results
test_ngrammed = np.delete(np.asarray(test, dtype=object), randomidx)[:10000]
del test
gc.collect()

# %% run test set through the model 
start = time.time()
results = pd.DataFrame(index=range(sum(len(t) for t in test_ngrammed)-len(test_ngrammed)), \
                       columns=['sent_nr', 'context', 'target', 'prediction', 'target_surprisal'])

i = 0
for st_no,sentence in enumerate(test_ngrammed):
    for idx,word in enumerate(sentence):
        if idx < len(sentence)-1:
            with torch.no_grad():
                inputs = prepare_sequence(word, vocabulary).to(device)                   
                scores = model(inputs)
                
                tgt = scores[-1] # last one of scores is distribution of predicted word
                tgt_wrd = sentence[idx+1][-1] # this is the word we were looking fore
                tgt_idx = vocabulary.word_to_index[tgt_wrd] # get index of actual target
    
                #prediction = vocabulary.index_to_word[list(tgt).index(max(tgt))] # max score = prediction - takes a minute, dont run
                surprisal = -tgt[tgt_idx]# surprisal of target
                
                results.iloc[i, 0] = st_no
                results.iloc[i, 1] = list(word)
                results.iloc[i, 2] = tgt_wrd
                #results.iloc[i, 3] = list(tgt).index(max(tgt))
                results.iloc[i, 4] = float(surprisal.cpu())
                
                i += 1
                
            if i % 100 == 0:
                print(f'Test word {i}. Movin on')

end = time.time()
#%%
results.to_csv(f'/project/3027007.06/simulations/results/{language}/{mname[:-4]}-results-10000sents.csv')

#%% example for paper
from utils import prepare_ngrams

base = ['on', 'the', 'street', 'lies', 'a', 'man']
plural = ['on', 'the', 'street', 'lies', 'a', 'men']
sem = ['on', 'the', 'street', 'lies', 'a', 'kite']
sem_plural = ['on', 'the', 'street', 'lies', 'a', 'kites']

sents = [base, plural, sem, sem_plural]
ngrams = prepare_ngrams(sents)

for sent in ngrams:
    word = sent[-2]
    context = sent[-3]
    #print(word, context)
    
    inputs = prepare_sequence(context, vocabulary).to(device)
    scores = model(inputs)
    tgt = scores[-1]
    tgt_wrd = sent[-2][-1]
    tgt_idx = vocabulary.word_to_index[tgt_wrd] # get index of actual target
    
    #prediction = vocabulary.index_to_word[list(tgt).index(max(tgt))] # max score = prediction - takes a minute, dont run
    surprisal = -tgt[tgt_idx]# surprisal of target
    print(tgt_wrd, surprisal)
    
#np.indices(np.shape(tgt))[np.where(tgt == max(tgt))]