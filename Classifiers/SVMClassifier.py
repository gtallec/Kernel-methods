#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Fri Feb  8 21:52:27 2019

@author: gtallec
"""
######################
import Classifier

######################
import numpy as np
import matplotlib.pyplot as plt

######################
from qpsolvers import solve_qp

class SVMClassifier(Classifier.Classifier):
    
    def __init__(self,lam,kernel):
        Classifier.Classifier.__init__(self)
        self.lam = lam
        self.kernel = kernel
        self.vector_machine = None
        self.predictor=None
        self.pred_matrix = None
        
    def classify(self):
        kernel_mat = self.kernel.matrix_from_data(self.inputs)
        labels = self.labels
        n = labels.shape[0]
        P = kernel_mat
        q = -labels
        G = np.diag(labels)
        h = (1/(2*self.lam*n))*np.ones((n,))
        
        self.predictor = solve_qp(P=P,
                                  q=q,
                                  G=G,
                                  h=h
                                  )
    def predict(self,input_to_predict):
        
        inputs = self.inputs
        n = inputs.shape[0]
        m = input_to_predict.shape[0]
        pred_matrix = self.kernel.compute_prediction_matrix(inputs,input_to_predict)
        self.pred_matrix = pred_matrix
        self.predictions = np.sign((self.predictor.reshape(n,1)).T@pred_matrix)[0]
    
    def plot_boundaries(self):
        inputs = self.inputs
        labels = self.labels
        n = inputs.shape[0]
        
        inputl_1 = inputs[labels==1]
        inputl_2 = inputs[labels==-1]
        
        coefs = (self.predictor.reshape(n,1)).T@inputs
        coef_dir = coefs[0][0]/coefs[0][1]
        print(coef_dir)
        
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
        

        
        
                
                
            
        
        
        
    
    