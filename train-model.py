#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 10:57:13 2022

@author: sopsla
"""
import os
import pickle 

import torch
from torch import nn
import random

from vocabulary import Vocabulary
from predictor import LSTM_predictor, prepare_sequence

import pandas as pd
import nltk
import numpy as np
import matplotlib.pyplot as plt


# %% device - to use GPU
if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu" 
  
device = torch.device(dev) 

# %% load the data & settings
with open('/home/lacnsg/sopsla/Documents/code-repositories/surprisal-simulations/corpus-syn.txt', 'r') as f:
    corpus = f.readlines()
    
language = pd.read_csv('/home/lacnsg/sopsla/Documents/code-repositories/surprisal-simulations/language.csv')

PATH = '/home/lacnsg/sopsla/Documents/code-repositories/surprisal-simulations'
mtype = 'structured'

# %% prepare the data
wordlist = []
for s in corpus:
    tree = nltk.Tree.fromstring(s)
    words = tree.leaves()
    words = [w for w in words if w != 't']
    wordlist.append(words)
    
# store lengths
lens = [len(s) for s in wordlist]
max_length = max(lens)
total_words = sum([len(w) for w in wordlist])

# %% prepare the training data
# maximum context is 10 words
from utils import prepare_ngrams

if mtype == 'random':
    scrambled = True
else:
    scrambled = False
    
train = list(prepare_ngrams(wordlist[:8000], scrambled=scrambled))
#test = list(prepare_ngrams(wordlist[8000:], scrambled=False)) # always false
# %%
"""
ngrammed = []
for i,sentence in enumerate(wordlist):
    if mtype == 'random':
        random.shuffle(sentence)
        
    sentence = list(nltk.ngrams(sentence, 10, pad_left=True, pad_right=False, left_pad_symbol='<pad>'))
    sentence = [('<pad>','<pad>','<pad>','<pad>','<pad>','<pad>','<pad>','<pad>','<pad>','<pad>')] + sentence
    if i > 7999:
        if sentence in ngrammed:
            continue
        
    ngrammed.append(sentence)
    
train = ngrammed[:8000]
#train = [list(NGRAM) for sentence in train for NGRAM in sentence]
"""
# %% the test sentences (do not run, saved already)
#test = ngrammed[8000:]

#with open(os.path.join(PATH, 'test_sentences.pkl'), 'wb') as f:
 #   pickle.dump(test, f)

with open(os.path.join(PATH, 'test_sentences.pkl'), 'rb') as f:
    test = pickle.load(f)

# %% prepare the vocabulary
vocab = language['word']

# vocab to map indices
vocabulary = Vocabulary(default_indexes={i: word for i,word in enumerate(vocab)})
vocabulary.index_words(['<pad>', '<EOS>']) # , '</pad>'
print(vocabulary.index_to_word)

# %% initialize the model
model = LSTM_predictor(embedding_dim = 10, hidden_dim = 64, vocab_size = vocabulary.num_words)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
# %% train it once
# TRAIN MODEL
# moving window
for sentence in train:
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

# %% save the model
mname = f'{mtype}_1pass-eos-syn.pkl'
torch.save(model, os.path.join(PATH, mname))
