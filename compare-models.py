#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 15:27:58 2022

@author: sopsla
"""
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import scipy

import nltk 
import os

# %% load results
PATH = '/home/lacnsg/sopsla/Documents/code-repositories/surprisal-simulations'

#random_results = pd.read_pickle(f'{PATH}/random_1pass_eos_results.pkl')
structured_results = pd.read_pickle(f'{PATH}/structured_1pass_eos_results.pkl')
random_results = pd.read_pickle(f'{PATH}/structured_1pass_eos-syn-results.pkl')

# %% plot histogram of surprisal values
fig,ax=plt.subplots(figsize=(5,3))

sns.histplot(random_results['target_surprisal'],ax=ax, color=sns.color_palette("YlOrBr",n_colors=1)) #'flare' for random, "YlOrBr" for syntactic/wf adaptations
sns.histplot(structured_results['target_surprisal'],ax=ax, color=sns.color_palette('crest',n_colors=1))

plt.legend(['Syntax altered model', 'Original structured model'], frameon=False) # ''Word-frequency altered', Original 
ax.set_xlabel('Target surprisal (bits)')
plt.tight_layout()
sns.despine()

fig.savefig(f'{PATH}/hist-syn-LTEXT.svg')

# %% lexical accuracy
random_results['lex_acc'] = np.where(random_results['prediction'] == random_results['target'], 1, 0)
structured_results['lex_acc'] = np.where(structured_results['prediction'] == structured_results['target'], 1, 0)

random_lex = np.sum(random_results['lex_acc']) / len(random_results) * 100
structured_lex = np.sum(structured_results['lex_acc']) / len(structured_results) * 100

print(random_lex, structured_lex)
# Random: 16.65%
# Structured: 27.95%

# What is the chance percentage? Absolute is  1/28 of course, but weighted by word frequency?
# random is virtually always predicting a determiner or a complementizer; these make up 0.16 + 0.15 + 0.15 = 0.46 of the corpus
# of these 45% occurrences, the network is right 1/3rd of the time, hence 15%

# %% POS accuracy
random_results['pos_acc'] = np.where(random_results['POS_prediction'] == random_results['POS_target'], 1, 0)
structured_results['pos_acc'] = np.where(structured_results['POS_prediction'] == structured_results['POS_target'], 1, 0)

random_pos = np.sum(random_results['pos_acc']) / len(random_results) * 100
structured_pos = np.sum(structured_results['pos_acc']) / len(structured_results) * 100

print(random_pos, structured_pos)
# Random: 26.95% is chance, there are 4 types
# Structured: 78.21% 

# %% how do the random outputs compare to frequency values?

# load the corpus
with open('/home/lacnsg/sopsla/Documents/code-repositories/surprisal-simulations/corpus.txt', 'r') as f:
    corpus = f.readlines()
    
# load the language
language = pd.read_csv('/home/lacnsg/sopsla/Documents/code-repositories/surprisal-simulations/language.csv')
    
# prepare the data
wordlist = []
for s in corpus:
    tree = nltk.Tree.fromstring(s)
    words = tree.leaves()
    words = [w for w in words if w != 't']
    wordlist.append(words)
    
# look at the training data only
trainwords = wordlist[:8000]
bagofwords = [w for stc in trainwords for w in stc]

counts = []
for wrd in list(language['word']):
    counts.append(sum([bg == wrd for bg in bagofwords]))
    
language['corpus_freq'] = counts
language['corpus_p'] = np.asarray(counts) / len(bagofwords)
language['corpus_surprisal'] = np.log10(np.asarray(counts) / len(bagofwords))

# %%
probdist = np.asarray(random_results['prob_dist'])

fig,ax = plt.subplots(figsize=(9,9))
for i,wrd in enumerate(language['word']):
    ax.hist([dist[i] for dist in probdist], alpha = 0.6)

ax.legend(list(language['word']))

# %% correlate
df = pd.DataFrame(data=np.asarray([random_results['target_surprisal'], structured_results['target_surprisal']], dtype=float).T, columns=['scrambled', 'structured'])

rho = df.corr()
pval = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x<=t]))
print(rho.round(2).astype(str) + p)

# %% test difference
print(scipy.stats.ttest_ind(random_results['target_surprisal'], structured_results['target_surprisal']))