#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:40:42 2022

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

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

from vocabulary import Vocabulary

# %% load values
PATH = '/project/3027007.06/simulations'
language = 'english'
languages = {'spanish': 116044,
             'english': 118311}
nr = languages[language]

random_results = pd.read_csv(f'{PATH}/results/{language}/{nr}-{language}-random-model-1layer-results-10000sents.csv')
structured_results = pd.read_csv(f'{PATH}/results/{language}/{nr}-{language}-model-1layer-results-10000sents.csv')

# %% plot histogram of surprisal values
fig,ax=plt.subplots(figsize=(5,3))

sns.histplot(random_results['target_surprisal'],ax=ax, color=sns.color_palette('flare',n_colors=1))
sns.histplot(structured_results['target_surprisal'],ax=ax, color=sns.color_palette('crest',n_colors=1))

plt.legend(['Scrambled model', 'Structured model'], frameon=False)
ax.set_xlabel('Target surprisal (bits)')
plt.tight_layout()
sns.despine()

#fig.savefig(f'/home/lacnsg/sopsla/Documents/code-repositories/surprisal-simulations/hist-scrambled-{language}-LTEXT.svg')

# %% add them to same df
results = structured_results.copy()
results['random_surprisal'] = random_results['target_surprisal']

# %% correlate
from scipy.stats import pearsonr

df = results[['random_surprisal', 'target_surprisal']]

rho = df.corr()
pval = df.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x<=t]))
print(rho.round(2).astype(str) + p)

#print(results[['random_surprisal', 'target_surprisal']].corr())

# %% difference

results['diff'] = results['random_surprisal'] - results['target_surprisal']
print(np.mean(results['diff']))
print(np.std(results['diff']))

# %% the means & std
print(stats.ttest_ind(results['random_surprisal'], results['target_surprisal']))

print(f"mean random: {np.mean(results['random_surprisal'])}, std: {np.std(results['random_surprisal'])}\n \
      mean structured: {np.mean(results['target_surprisal'])}, std: {np.std(results['target_surprisal'])}")
      
print(f"Mean difference: {np.mean(results['random_surprisal'])-np.mean(results['target_surprisal'])}")

# %% pinguoin for effect sizes and confidence intervals
import pingouin

res = pingouin.ttest(np.asarray(results['random_surprisal'].values, dtype=float), np.asarray(results['target_surprisal'].values, dtype=float), paired=False)

print(res)

# %% plotting the scrambled and structured languages
spanish_scrambled = pd.read_csv('/project/3027007.06/simulations/results/spanish/116044-spanish-random-model-1layer-results-10000sents.csv')
english_scrambled = pd.read_csv('/project/3027007.06/simulations/results/english/118311-english-random-model-1layer-results-10000sents.csv')

spanish = pd.read_csv('/project/3027007.06/simulations/results/spanish/116044-spanish-model-1layer-results-10000sents.csv')
english = pd.read_csv('/project/3027007.06/simulations/results/english/118311-english-model-1layer-results-10000sents.csv')

spanish['zscored'] = stats.zscore(spanish['target_surprisal'])
english['zscored'] = stats.zscore(english['target_surprisal'])

spanish_scrambled['zscored'] = stats.zscore(spanish_scrambled['target_surprisal'])
english_scrambled['zscored'] = stats.zscore(english_scrambled['target_surprisal'])

fig,ax=plt.subplots(figsize=(8,3), ncols = 2, sharey=True)

sns.histplot(english['zscored'],color=sns.color_palette('Blues',n_colors=1), bins=200, ax = ax[0])
sns.histplot(spanish['zscored'],color=sns.color_palette('Reds',n_colors=1), bins=200, ax = ax[0])
ax[0].set_title('Structured models')

sns.histplot(english_scrambled['zscored'],color=sns.color_palette('Blues',n_colors=1), bins=200, ax = ax[1])
sns.histplot(spanish_scrambled['zscored'],color=sns.color_palette('Reds',n_colors=1), bins=200, ax = ax[1])
ax[1].set_title('Scrambled models')

plt.legend(['English', 'Spanish'], frameon=False)
ax[0].set_xlabel('Target surprisal (bits)')
ax[1].set_xlabel('Target surprisal (bits)')
plt.tight_layout()
sns.despine()

fig.savefig('/home/lacnsg/sopsla/Documents/code-repositories/surprisal-simulations/clustering-bothlangs-zscored-LTEXT.svg')

# %%
