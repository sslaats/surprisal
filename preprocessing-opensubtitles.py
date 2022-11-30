#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:27:20 2022

@author: sopsla
"""
import gc
import pickle 
import nltk
import string


#%% spanish
with open('/project/3027007.06/simulations/OpenSubtitles/spanish-sentences.pkl', 'rb') as f:
    sentences = pickle.load(f)

# remove punctuation & split into words
translator = str.maketrans('', '', string.punctuation + '¡¿')
words = [nltk.tokenize.word_tokenize(sent.translate(translator).lower(), language='spanish') for sent in sentences]
del sentences

with open('/project/3027007.06/simulations/OpenSubtitles/spanish-words.pkl', 'wb') as f:
    pickle.dump(words, f)

del words
gc.collect()


"""
# %% english
with open('/project/3027007.06/simulations/OpenSubtitles/en-es.txt/OpenSubtitles.en-es.en') as f:
    english = f.read()
    
# split into sentences - 61 million of them
sentences = nltk.tokenize.sent_tokenize(english, language='english')
del english
gc.collect()

with open('/project/3027007.06/simulations/OpenSubtitles/english-sentences.pkl', 'wb') as f:
    pickle.dump(sentences, f)

# remove punctuation & split into words
translator = str.maketrans('', '', string.punctuation + '¡¿')
words = [nltk.tokenize.word_tokenize(sent.translate(translator).lower(), language='english') for sent in sentences]
del sentences

with open('/project/3027007.06/simulations/OpenSubtitles/english-words.pkl', 'wb') as f:
    pickle.dump(words, f)

del words
gc.collect()"""