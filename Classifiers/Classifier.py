#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 21:44:41 2019

@author: gtallec
"""
class Classifier:
    
    def __init__(self):
        self.inputs = None
        self.labels = None
        self.predictions = None
        
    def fit(self,inputs, labels):
        self.inputs = inputs
        self.labels = labels
        self.classify()
        

        