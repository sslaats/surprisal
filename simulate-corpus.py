#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 10:40:25 2022

@author: sopsla
"""
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from grammar import Grammar

# %% the language
language = pd.DataFrame(columns=['word', 'pos', 'freq'])

language['word'] = ['that',
                    'a', 'the', 
                    'woman', 'dog', 'goat', 'president', 'bird', 'colleague', 'mother', 'toddler', 'scientist', 'child', 'farmer', 'painter', 'cat',
                    'loves', 'discovers', 'reveals', 'notices', 'assumes', 'indicates', 'finds', 'senses', 'guarantees', 'teaches', 'hears', 'understands']
language['pos'] = ['comp',
                   'det', 'det',
                   'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n', 'n','n', 'n', 'n', 'n',
                   'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v', 'v']
language['freq'] = np.ones((len(language['word'])))
language['p'] = language['freq'] / sum(language['freq'])

# language = pd.read_csv('/home/lacnsg/sopsla/Documents/code-repositories/surprisal-simulations/language.csv')

# %% parameters
update = False
memory_trace = 0
sentence_no = 10000
sentence_lengths = np.round(np.random.lognormal(mean=2.85,sigma=0.4,size=sentence_no)) # set up probability distribution for sentence length

# %%
gramm = Grammar(language, update= update, memory_trace = memory_trace, max_subordinate = 5)

corpus = []

while len(corpus) < sentence_no:
    
    try:
        sent = gramm.IP(noun_fun=gramm.NP(0), verb_fun=gramm.VP(0))
        corpus.append(sent)
        
    except(KeyError, RecursionError):
        continue

with open('/home/lacnsg/sopsla/Documents/code-repositories/surprisal-simulations/corpus.txt', 'w') as f:
    f.write('\n'.join(corpus))