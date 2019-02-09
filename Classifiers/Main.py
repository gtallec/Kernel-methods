#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 22:22:40 2019

@author: gtallec
"""

import SVMClassifier

SVM_classifier = SVMClassifier.SVMClassifier(lam=5)
print(SVM_classifier.kernel(5,7))