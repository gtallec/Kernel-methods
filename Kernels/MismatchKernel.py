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
    
    def features_from_data(self,inputs):
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
            
        feature_matrix = np.array(list(D.values()))
            
                                      
        return feature_matrix
    
    def matrix_from_data(self,inputs):
        features = self.features_from_data(inputs)
        return features.T@features
    
    
    def compute_prediction_embedding(self, inputs, tests):
                
        n = inputs.shape[0]
        m = tests.shape[0]
        
        tot_data = np.concatenate((inputs, tests))
        features = self.features_from_data(tot_data)
        
        training_features = features[:,:n]
        testing_features = features[:,n:]
        
        return training_features.T@testing_features
        
        
        
    
    def compute_prediction_embedding2(self, inputs, tests):
        n = inputs.shape[0]
        m = tests.shape[0]
        embedding = np.zeros((n,m))
        for i in tqdm_notebook(range(n)):
            for j in range(m):
                embedding[i,j] = self.compute_k(inputs[i], tests[j])
                
        return embedding
    
    def compute_k(self,x1,x2):
        
        k = self.k
        l = len(x1)
        Kern_value = 0
        
        select = (np.tile(np.arange(k)[np.newaxis,:], reps = (l-k + 1, 1)) +
                  np.tile(np.arange(l-k + 1)[:, np.newaxis], reps = (1,k)))
        #initialise with all k substring of x1 and x2
        x1_list = np.array(list(x1))
        x2_list = np.array(list(x2))
        L_x1 = dict()
        L_x1[''] = np.apply_along_axis(lambda select: (''.join(x1_list[select]),0,0), arr = select, axis = 1)
        L_x2 = dict()
        L_x2[''] = np.apply_along_axis(lambda select: (''.join(x2_list[select]),0,0), arr = select, axis = 1)
        L_k = dict()
        L_k['K'] = 0
        
        def add_to_dict(dictionary, key, value):
            if not key in dictionary:
                dictionary[key] = []
            dictionary[key].append(value)
            
        def retrieve_from_dict(dictionary, key):
            if not key in dictionary:
                return []
            else:
                return dictionary[key]
       
        def processnode(v, depth):

            L1 = retrieve_from_dict(L_x1, v)
            L2 = retrieve_from_dict(L_x2, v)
            
            l1 = len(L1)
            l2 = len(L2)
            
            if depth == self.k :
                L_k['K']+= l1*l2
            
            elif l1*l2 != 0:
                for el in L1:
                    u,i,j = el[0],int(el[1]),int(el[2])
                    u_c = u[i]
                    add_to_dict(L_x1, v + u_c, (u, i+1, j))
                    if j < self.m:
                        for a in self.alphabet:
                            if a != u_c:
                                add_to_dict(L_x1, v + a, (u, i+1, j+1))
                for el in L2:
                    u,i,j = el[0],int(el[1]),int(el[2])
                    u_c = u[i]
                    add_to_dict(L_x2, v + u_c, (u, i+1, j))
                    if j < self.m:
                        for a in self.alphabet:
                            if a != u_c:
                                add_to_dict(L_x2, v + a, (u, i+1, j+1))
                                
                for a in self.alphabet:
                    processnode(v + a, depth + 1)
                    
        processnode('',0)
        return L_k['K']
                            
                                
                
        
        

                
                    