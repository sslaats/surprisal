#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 15:02:08 2022

@author: sopsla
"""
import os
import torch

import pandas as pd
import numpy as np

from vocabulary import Vocabulary
from predictor import prepare_sequence

import pickle

# %% settings
if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu" 
  
device = torch.device(dev) 

PATH = '/home/lacnsg/sopsla/Documents/code-repositories/surprisal-simulations'
mtype = 'structured' # other option: 'random'
language = pd.read_csv('/home/lacnsg/sopsla/Documents/code-repositories/surprisal-simulations/language.csv')

# %% prepare the vocabulary
vocab = language['word']

# vocab to map indices
vocabulary = Vocabulary(default_indexes={i: word for i,word in enumerate(vocab)})
vocabulary.index_words(['<pad>']) 

#%% load the model and the test sentences
model = torch.load(f=f'{PATH}/{mtype}_1pass.pkl')

with open(f'{PATH}/test_sentences.pkl', 'rb') as f:
    test = pickle.load(f)

# %% test the model, save the results
results = pd.DataFrame(index=range(sum(len(t) for t in test)-len(test)), columns=['context', 'target', 'POS_target', 'prediction', 'POS_prediction', 'prob_dist', 'target_surprisal'])

i = 0
for sentence in test:
    for idx,word in enumerate(sentence):
        if idx < len(sentence)-1:
            with torch.no_grad():
                inputs = prepare_sequence(word, vocabulary).to(device)
                target = prepare_sequence(sentence[idx+1], vocabulary).to(device)
                
                scores = model(inputs)
                
                tgt = scores[-1]
                tgt_wrd = sentence[idx+1][-1]
                tgt_idx = vocabulary.word_to_index[tgt_wrd]
                
        
                prediction = vocabulary.index_to_word[list(tgt).index(max(tgt))]
                surprisal = -tgt[tgt_idx]
                
                results.iloc[i, 0] = list(word)
                results.iloc[i, 1] = tgt_wrd
                results.iloc[i, 2] = language.loc[language['word'] == tgt_wrd, 'pos'].item()
                results.iloc[i, 3] = prediction
                try:
                    results.iloc[i, 4] = language.loc[language['word'] == prediction, 'pos'].item()
                except ValueError:
                    results.iloc[i, 4] = 'pad'
                results.iloc[i, 5] = np.asarray(tgt.cpu())
                results.iloc[i, 6] = float(surprisal.cpu())
                
                i += 1
                
            if i % 100 == 0:
                print(f'Test word {i}. Movin on')

results.to_pickle(os.path.join(PATH, f'{mtype}_1pass_results.pkl'))