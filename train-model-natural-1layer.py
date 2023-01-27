#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 14:37:23 2022

@author: sopsla
"""
# model with a single layer may work better
# also adjust the embedding size
import os
import pickle 
import gc
import sys

import torch
from torch import nn

from vocabulary import Vocabulary
from predictor import LSTM_predictor, prepare_sequence
from utils import prepare_ngrams

# %% settings
PATH = '/project/3027007.06/simulations'
mtype = 'random'
languages = ['spanish', 'english']
language = languages[int(sys.argv[1])]

print(f'Preparing to train LSTM on {language}')

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

# %% load the highest model
#models = [m for m in os.listdir(os.path.join(PATH, f'results/{language}')) if m.endswith('1layer.pkl')]
#sizes = [int(m.split('-')[0]) for m in models]
#nsent_done = max(sizes)
#model = torch.load(f=f'{PATH}/results/{language}/{nsent_done}-{language}-model-1layer.pkl')
nsent_done = 0

# %% vocabulary computation
if os.path.isfile(f'{PATH}/OpenSubtitles/{language}-vocab.pkl'):
    # load it
    with open(f'{PATH}/OpenSubtitles/{language}-vocab.pkl', 'rb') as f:
        vocabulary = pickle.load(f)
else:
    # prepare the vocabulary
    vocab = list(set([w for sent in words for w in sent]))
    vocabulary = Vocabulary(default_indexes={i: word for i,word in enumerate(vocab)})
    vocabulary.index_words(['<pad>', '<EOS>'])

print('Vocabulary done')

# we have 61 million sentences. In this script we test & save the model 
# occasionally to find what works best.

# %% ngrams, train & test set
  
# the test set will be 5% of the total sentences
test_idx = int(len(words)) - int(len(words)/20)

if mtype != 'random':
    scrambled = False
else:
    scrambled = True

#train = prepare_ngrams(words[:test_idx], scrambled)
train = prepare_ngrams(words[nsent_done:test_idx], scrambled)
test = list(prepare_ngrams(words[test_idx:], scrambled))
print('Ngrams done')                    
    
# remove the full corpus to save some space
del words
gc.collect()

# model saves - to check how many sentences will lead to lowest surprisal values
saves = [int(test_idx/10000), int(test_idx/5000), int(test_idx/1000),
         int(test_idx/500), int(test_idx/100), int(test_idx/50),
         int(test_idx/10), int(test_idx/5), test_idx]

if mtype != 'random': # we do this only once
    if ~os.path.isfile(f'/project/3027007.06/simulations/OpenSubtitles/{language}-test.pkl'):
        print('Saving the test set')
        with open(f'/project/3027007.06/simulations/OpenSubtitles/{language}-test.pkl', 'wb') as f:
            pickle.dump(test, f)
        
del test

# %% initialize model
# for later: test with different layer sizes?
model = LSTM_predictor(embedding_dim = 300, hidden_dim = 600, vocab_size = vocabulary.num_words, nlayers=1)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
print('Model initialized')

# %% train the model
# where to save results
SAVE_PATH = f'/project/3027007.06/simulations/results/{language}/'

for st_no, sentence in enumerate(train):
    st_no_true = nsent_done + 1 + st_no
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
            
    if st_no_true in saves:      
        print(f'Saving (intermediate) model after {st_no_true} sentences')
    
        # save the model too
        torch.save(model, os.path.join(SAVE_PATH, f'{str(st_no_true)}-{language}-{mtype}-model-1layer.pkl'))

print('Saving final model')
torch.save(model, os.path.join(SAVE_PATH, f'full-{language}-{mtype}-model-1layer.pkl'))

print('Done! Exiting...')
