#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 21:44:41 2019

@author: gtallec
"""
import numpy as np
class Classifier:
    
    def __init__(self):
        self.inputs = None
        self.labels = None
        self.predictions = None
        
    def fit(self,inputs, labels):
        self.inputs = inputs
        self.labels = labels
        self.classify()
        
    def compute_test_accuracy(self,test_inputs, test_labels):
        print('predictions', self.predictions)
        self.predict(test_inputs)
        return 1 - (np.count_nonzero(self.predictions - test_labels)/self.predictions.shape[0])
        

        