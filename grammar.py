#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 10:23:14 2022

@author: sopsla
"""
from numpy import random


class Grammar():
    def __init__(self, language, update, memory_trace = 0.1, max_subordinate = 5):
        self.language = language
        self.update = update
        self.max_subordinate = max_subordinate
        self.memory_trace = memory_trace
        
    def frequency_update(self, word):
        self.language.loc[self.language['word'] == word, 'freq'] += self.memory_trace
        self.language['p'] = self.language['freq'] / sum(self.language['freq'])
    
    def IP(self, noun_fun, verb_fun):
        return f'(IP {noun_fun} {verb_fun})'
    
    def CP(self, noun_fun, verb_fun):
        return f'(CP (C that) {self.IP(noun_fun, verb_fun)})'
    
    def NP(self, subordinate_clauses):
        
        det = random.choice(self.language.loc[self.language['pos'] == 'det', 'word'], p = self.language.loc[self.language['pos'] == 'det', 'freq']/sum(self.language.loc[self.language['pos'] == 'det', 'freq']))
        n = random.choice(self.language.loc[self.language['pos'] == 'n', 'word'], p =self.language.loc[self.language['pos'] == 'n', 'freq']/sum(self.language.loc[self.language['pos'] == 'n', 'freq']))
    
        if subordinate_clauses < self.max_subordinate:
            if random.choice([True, True, False]):
                return f'(NP (Det {det}) (nbar (N {n})))'
            else:
                subordinate_clauses += 1
                return f'(NP (Det {det}) (nbar (N {n}) {self.CP(noun_fun="(NP t)", verb_fun=self.VP(subordinate_clauses))}))'
        else:
            return f'(NP (Det {det}) (nbar (N {n})))'
        
    def VP(self, subordinate_clauses):
        v = random.choice(self.language.loc[self.language['pos'] == 'v', 'word'], p =self.language.loc[self.language['pos'] == 'v', 'freq']/sum(self.language.loc[self.language['pos'] == 'v', 'freq']))
        
        # choose a complement
        if subordinate_clauses < self.max_subordinate:
            if random.choice([True, True, False]):
                return f'(VP (V {v}) {self.NP(subordinate_clauses)})'
            else:
                subordinate_clauses += 1
                return f'(VP (V {v}) {self.CP(self.NP(subordinate_clauses), self.VP(subordinate_clauses))})'
        else:
            return f'(VP (V {v}) {self.NP(subordinate_clauses)})'