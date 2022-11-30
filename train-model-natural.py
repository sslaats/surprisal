#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:27:20 2022

@author: sopsla
"""
import os
import pickle 
import gc

import torch
from torch import nn

from vocabulary import Vocabulary
from predictor import LSTM_predictor, prepare_sequence
from utils import get_surprisal_for_test, prepare_ngrams

import numpy as np

mtype = 'structured'
language = 'english'

# %% device - to use GPU
if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu" 
  
device = torch.device(dev) 

# %% load the pre-processed input (lists of words/sentence, decapitalized, punctuation removed)
# spanish
with open(f'/project/3027007.06/simulations/OpenSubtitles/{language}-words.pkl', 'rb') as f:
    words = pickle.load(f)

nsent = len(words)
vocab = list(set([w for sent in words for w in sent]))

# prepare the vocabulary
vocabulary = Vocabulary(default_indexes={i: word for i,word in enumerate(vocab)})
vocabulary.index_words(['<pad>', '<EOS>'])

# we have 61 million sentences. In this script we test & save the model 
# occasionally to find what works best.

# %% ngrams, train & test set
  
# the test set will be 5% of the total sentences
test_idx = int(len(words)) - int(len(words)/20)

if mtype != 'random':
    scrambled = False
else:
    scrambled = True
    
train = prepare_ngrams(words[:test_idx], scrambled)
test = list(prepare_ngrams(words[test_idx:], scrambled))
                        
# remove the full corpus to save some space
del words
gc.collect()

# take random part of the test set of 1000 sentences for intermediate testing 
randomidx = np.random.randint(0, len(test), 1000)
ngrammed_save = np.asarray(test, dtype=object)[randomidx]
test = np.asarray(test, dtype=object)[~randomidx] # the actual test set does not include any of these

# model saves - to check how many sentences will lead to lowest surprisal values
saves = [int(test_idx/10000), int(test_idx/5000), int(test_idx/1000),
         int(test_idx/500), int(test_idx/100), int(test_idx/50),
         int(test_idx/10), int(test_idx/5), test_idx]

if mtype != 'random': # we do this only once
    print('Saving the test set')
    with open(f'/project/3027007.06/simulations/OpenSubtitles/{language}-test.pkl', 'wb') as f:
        pickle.dump(test, f)
        
del test

# %% initialize model
# for later: test with different layer sizes?
model = LSTM_predictor(embedding_dim = 10, hidden_dim = 500, vocab_size = vocabulary.num_words)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# %% train the model
# where to save results
SAVE_PATH = f'/project/3027007.06/simulations/results/{language}/'

for st_no, sentence in enumerate(train):

    for i,word in enumerate(sentence):
        if i < len(sentence)-1:
            model.zero_grad()
                
            sentence_in = prepare_sequence(word, vocabulary).to(device)
            target = prepare_sequence(sentence[i+1], vocabulary).to(device)
        
            prediction_scores = model(sentence_in)
            
            # compute the loss, gradients, update parameters
            loss = loss_function(prediction_scores, target)
            loss.backward()
            optimizer.step()
            
    if st_no in saves:
        print('Getting (intermediate) results after {st_no} sentences')
        
        # obtain results
        results = get_surprisal_for_test(ngrammed_save, model, vocabulary, device)
        
        print('Saving (intermediate) results and model')
        results.to_pickle(os.path.join(SAVE_PATH, f'{str(st_no)}-{language}-results.pkl'))
    
        # save the model too
        torch.save(model, os.path.join(SAVE_PATH, f'{str(st_no)}-{language}-model.pkl'))
        
print('Done! Exiting...')
