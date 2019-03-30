import itertools
import re
import numpy as np
from . import Kernel
from numba import jit
import copy

class KSpectrumKernel(Kernel.Kernel):
    
    def __init__(self,k,alphabet,lmbda):
        Kernel.Kernel.__init__(self, None, 'kstring?k=' + str(k) + '&lmbda=' + str(lbmda))
        self.lmbda = lmbda
        self.k = k
        self.alphabet_length = len(alphabet)
        
    def compute_occurences(self,x):
        k_words = copy.deepcopy(self.k_words)
        res = np.fromiter(map(lambda word : len(re.findall('(?='+word+')',x)), k_words), dtype = int)
        del k_words
        return res
    
    def compute_similarity_matrix(self, x, y):
        vfun = np.vectorize(self.compute_occurences, signature='()->(n)')
        X = np.apply_along_axis(func1d=vfun, arr=x, axis = 0)
        Y = np.apply_along_axis(func1d=vfun, arr=y, axis = 0)
            
        return X@(Y.T)
    
    @jit
    def K_p(self, p, x, y):
        if p == 0:
            return 0
        else:
            s,a = x[:-1], x[-1]
            t,b = y[:-1], y[-1]
            if a == b:
                return (self.lmbda**2)*(1 + K_p(p-1,s, t))
            else:
                return 0
    @jit       
    def compute_k(self,x,y):
        Kern = 0
        X = len(x)
        Y = len(y)
        k = self.k
        
        for i in range(X-k):
            for j in range(Y-k):
                Kern += K_p(p, x[i:i+k], y[j:j+k])
                
        return Kern
    @jit
    def compute_prediction_embedding(self,inputs, tests):
        n = inputs.shape[0]
        m = tests.shape[0]
        embedding = np.zeros((n,m))
        
        for i in range(n):
            for j in range(m):
                embedding[i,j] = self.compute_k(inputs[i], tests[j])
                
        return embedding
    
    @jit
    def matrix_from_data(self, inputs):
        return compute_prediction_embedding(inputs, inputs)
                
        
        
        
          
        