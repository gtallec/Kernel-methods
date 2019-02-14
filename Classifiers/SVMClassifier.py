#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Fri Feb  8 21:52:27 2019

@author: gtallec
"""
from . import Classifier               
from qpsolvers import solve_qp        

import numpy as np        
import matplotlib.pyplot as plt


class SVMClassifier(Classifier.Classifier):
    
    def __init__(self,kernel):
        Classifier.Classifier.__init__(self)
        self.lam = None
        self.kernel = kernel
        self.predictor=None
        
    def setParams(self):
        if len(self.hyperParameters) != 1:
            print('WRONG PARAMETRIZATION OF SVM MODEL')
        self.lam = self.hyperParameters[0]
            
        
    def classify(self):
        print('STEP 1 - Kernel Matrix computation')
        kernel_mat = self.kernel.matrix_from_data(self.inputs)
        print('END STEP 1')
        labels = self.labels
        n = labels.shape[0]
        P = kernel_mat
        q = -labels
        G = np.vstack([np.diag(labels), np.diag(-labels)])
        h = np.hstack([(1/(2*self.lam*n))*np.ones((n,)), np.zeros((n,))])
        
        print('STEP 2 - QP Solving')
        self.predictor = solve_qp(P=P,
                                  q=q,
                                  G=G,
                                  h=h
                                  )
        print('END STEP 2')
        
    def predict(self,input_to_predict):
        print('STEP x - PREDICTIONS')
        inputs = self.inputs
        n = inputs.shape[0]
        m = input_to_predict.shape[0]
        pred_matrix = self.kernel.compute_similarity_matrix(inputs,input_to_predict)
        return np.sign((self.predictor.reshape(n,1)).T@pred_matrix)[0]

    
    def plot_boundaries(self):
        inputs = self.inputs
        labels = self.labels
        n = inputs.shape[0]
        
        inputl_1 = inputs[labels==1]
        inputl_2 = inputs[labels==-1]
        
        coefs = (self.predictor.reshape(n,1)).T@inputs
        coef_dir = coefs[0][0]/coefs[0][1]
        
        X = np.linspace(-5,5,10)
        Y_1 = 1 - coef_dir*X
        Y_2 = -1 - coef_dir*X
        plt.plot(X,Y_1)
        plt.plot(X,Y_2)
        plt.scatter(inputl_1[:,0], inputl_1[:,1],color='blue')
        plt.scatter(inputl_2[:,0], inputl_2[:,1],color='red')
        plt.grid(True)
        plt.axis()
        plt.show()
        

        
        
                
                
            
        
        
        
    
    