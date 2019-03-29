import numpy as np
from tqdm import tqdm_notebook
from . import Kernel

class MismatchKernel(Kernel.Kernel):
    
    def __init__(self,k,m,alphabet):
        Kernel.Kernel.__init__(self,None, 'mismatch?k=' + str(k) + '&m=' +str(m))
        self.k = k
        self.m = m
        self.alphabet = alphabet
        
    def find_prefixed_words(self,X, number_of_mistakes, prefix):
    
        prefices, suffices = X[:,0], X[:,1:]
        errors = np.zeros((number_of_mistakes.shape))
        errors[np.where(prefices != prefix)[0]] += 1
        return number_of_mistakes + errors, suffices
    
    def matrix_from_data(self,inputs):
        def string_to_array(string):
            return np.array(list(string))
        X = string_to_array(inputs[0])
        for i in range(1, inputs.shape[0]):
            X = np.vstack([X, string_to_array(inputs[i])])
        
        
        n = X.shape[0]
        k = self.k
        D = dict()
        
        def depth_first_search(X, number_of_mistakes, depth_accu, prefix_accu, ID, n, verbose = False):
            if verbose :
                print('prefix accu', prefix_accu)
                print('number of mistakes', number_of_mistakes)
                print('X', X)
                print('shape', X.shape)
                print('depth_accu', depth_accu)
                print('\n')

            if depth_accu == self.k:
                valid_indices = np.where(number_of_mistakes <= self.m)[0]
                if valid_indices.shape[0] != 0: 
                    number_of_mistakes = number_of_mistakes[valid_indices]
                    X = X[valid_indices]
                    if not(prefix_accu in D):
                        D[prefix_accu] = np.zeros((n,))
                        D[prefix_accu][ID] = X.shape[0]
                        if verbose:
                            print('dictionary update :', prefix_accu,',',ID,',',X.shape[0]) 
                    else:
                        D[prefix_accu][ID] += X.shape[0]
            else:
                #drop the k-mer with already to much mismatches
                valid_indices = np.where(number_of_mistakes <= self.m)[0]
                if verbose:
                    print('valid_indices', valid_indices)
                if valid_indices.shape[0] != 0: 
                    number_of_mistakes = number_of_mistakes[valid_indices]
                    X = X[valid_indices]
                    for letter in self.alphabet:
                        mistakes, suffices = self.find_prefixed_words(X, number_of_mistakes, letter)
                        depth_first_search(suffices, mistakes, depth_accu + 1, prefix_accu + letter, ID, n, verbose)

        for i in tqdm_notebook(range(n)):
            x = X[i]
            l = x.shape[0]
            select = (np.tile(np.arange(k)[np.newaxis,:], reps = (l-k + 1, 1)) +
                      np.tile(np.arange(l-k + 1)[:, np.newaxis], reps = (1,k)))
            X_init = np.apply_along_axis(lambda select : x[select], arr = select, axis = 1)
            number_of_mistakes = np.zeros((X_init.shape[0],))
            depth_first_search(X_init, number_of_mistakes, 0,'', i, n)
            
        print('D', D)
        feature_matrix = np.array(list(D.values()))
        print('feature_matrix', feature_matrix)
            
                                      
        return feature_matrix.T@feature_matrix

                
                    