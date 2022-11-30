#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 10:57:13 2022

@author: sopsla
"""
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import math

from vocabulary import Vocabulary
from predictor import LSTM_predictor, prepare_sequence

import pandas as pd
import nltk
import numpy as np
import matplotlib.pyplot as plt

import random

# %% device - to use GPU
if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu" 
  
device = torch.device(dev) 

# %% load the data
with open('/home/lacnsg/sopsla/Documents/code-repositories/surprisal-simulations/corpus.txt', 'r') as f:
    corpus = f.readlines()
    
language = pd.read_csv('/home/lacnsg/sopsla/Documents/code-repositories/surprisal-simulations/language.csv')

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
ngrammed = []
for i,sentence in enumerate(wordlist):
    if i < 8000:
        random.shuffle(sentence)
        
    sentence = list(nltk.ngrams(sentence, 10, pad_left=True, pad_right=False, left_pad_symbol='<pad>'))
    ngrammed.append(sentence)
    
train = ngrammed[:8000]
test = ngrammed[8000:]

# last item is the target
train = [list(NGRAM) for sentence in train for NGRAM in sentence]
test = [list(NGRAM) for sentence in test for NGRAM in sentence]

#all_data = [list(NGRAM) for sentence in ngrammed for NGRAM in sentence]

# %% prepare the vocabulary
vocab = language['word']

# vocab to map indices
vocabulary = Vocabulary(default_indexes={i: word for i,word in enumerate(vocab)})
vocabulary.index_words(['<pad>']) # , '</pad>'
print(vocabulary.index_to_word)

# %% initialize the model
model = LSTM_predictor(embedding_dim = 10, hidden_dim = 64, vocab_size = vocabulary.num_words)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

with torch.no_grad():
    inputs = prepare_sequence(train[0][:-1], vocabulary).to(device)
    print(inputs)
    scores = model(inputs)
    print(scores)
    
# %% train it once on the random data
# TRAIN MODEL
for sentence in train:
    model.zero_grad()
        
    sentence_in = prepare_sequence(sentence[:-1], vocabulary).to(device)
    target = prepare_sequence(sentence[1:], vocabulary).to(device)

    prediction_scores = model(sentence_in)
    
    # compute the loss, gradients, update parameters
    loss = loss_function(prediction_scores, target)
    loss.backward()
    optimizer.step()
    
# %% save the model
PATH = '/home/lacnsg/sopsla/Documents/code-repositories/surprisal-simulations'
mname = 'random_1pass.pkl'
torch.save(model, os.path.join(PATH, mname))
