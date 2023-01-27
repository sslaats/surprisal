#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 11:03:35 2022

@author: sopsla
"""
import torch

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
    
device = torch.device(dev)

class LSTM_predictor(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, nlayers=1):
        super(LSTM_predictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim).to(device)
        
        # LSTM takes word embeddings as inputs and outputs hidden states with dimensionality hidden_dim
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=nlayers).to(device)
        
        # linear layer maps from hidden state back into vocab space
        self.hidden2word = torch.nn.Linear(hidden_dim, vocab_size).to(device)
        
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence).to(device)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        word_space = self.hidden2word(lstm_out.view(len(sentence), -1)).to(device)
        word_scores = torch.nn.functional.log_softmax(word_space, dim=1).to(device)
        return word_scores
    
def prepare_sequence(wordlist, vocab):
    idx = [vocab.word_to_index[word] for word in wordlist]
    return torch.tensor(idx, dtype=torch.long)
