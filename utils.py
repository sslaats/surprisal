#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 11:09:31 2022

@author: sopsla
"""
import pandas as pd
import numpy as np

import torch
from predictor import prepare_sequence
import random
import nltk

def prepare_ngrams(words, scrambled=False):
    """
    Function prepares a generator of lists of tuples; tuples are ngrams,
    inner lists are sentences
    
    INPUT
    -----
    words : list of list of str
    scrambled: boolean - when True, words in sentence are scrambled
    
    OUTPUT
    -----
    Generator object containing list of tuples - tuples are ngrams
    
    """
    for i,sentence in enumerate(words):
        # this is for the random 
        if scrambled:
            random.shuffle(sentence)
    
        sentence = list(nltk.ngrams(sentence, 10, pad_left=True, pad_right=False, left_pad_symbol='<pad>'))
        sentence = [('<pad>','<pad>','<pad>','<pad>','<pad>','<pad>','<pad>','<pad>','<pad>','<pad>')] + sentence
        sentence = sentence + [sentence[-1][1:] + tuple(['<EOS>'])]
        
        yield sentence

def get_surprisal_for_test(test_ngrammed, model, vocabulary, device):
    """
    little function to shorten the loop
    takes ngrams and calculates surprisal for all the words.
    
    INPUT
    ------
    test_ngrammed : list or array of tuples (ngrams)
    model : instance of LSTM_predictor (torch.Module)
    vocabulary : instance of Vocabulary
    deivice : CPU/GPU
    
    OUTPUT
    -------
    results : pd.DataFrame
    
    """
    results = pd.DataFrame(index=range(sum(len(t) for t in test_ngrammed)-len(test_ngrammed)), \
                           columns=['context', 'target', 'prediction', 'prob_dist', 'target_surprisal'])
    
    i = 0
    for sentence in test_ngrammed:
        for idx,word in enumerate(sentence):
            if idx < len(sentence)-1:
                with torch.no_grad():
                    inputs = prepare_sequence(word, vocabulary).to(device)                   
                    scores = model(inputs)
                    
                    tgt = scores[-1] # last one of scores is distribution of predicted word
                    tgt_wrd = sentence[idx+1][-1] # this is the word we were looking fore
                    tgt_idx = vocabulary.word_to_index[tgt_wrd] # get index of actual target
        
                    prediction = vocabulary.index_to_word[list(tgt).index(max(tgt))] # max score = prediction
                    surprisal = -tgt[tgt_idx]# surprisal of target
                    
                    results.iloc[i, 0] = list(word)
                    results.iloc[i, 1] = tgt_wrd
                    results.iloc[i, 2] = prediction
                    results.iloc[i, 3] = np.asarray(tgt.cpu())
                    results.iloc[i, 4] = float(surprisal.cpu())
                    
                    i += 1
                    
                if i % 100 == 0:
                    print(f'Test word {i}. Movin on')
    
    return results