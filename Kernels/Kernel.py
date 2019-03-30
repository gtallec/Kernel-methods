import numpy as np


class Kernel:
    
    def __init__(self, kernel_fun, kernel_name):
        self.kernel_fun = kernel_fun
        self.kernel_name = kernel_name
        
    def matrix_from_data(self, inputs):
        eps = 10e-7
        n = inputs.shape[0]
        print('GETTING INTO IT')
        return self.compute_similarity_matrix(inputs, inputs) + eps*np.identity(n)
          
    def compute_similarity_matrix(self, x, y):
        n = x.shape[0]
        m = y.shape[0]
        
        similarity_matrix = np.fromfunction(np.vectorize(lambda i,j : self.kernel_fun(x[int(i)], y[int(j)])),
                                      shape = (n,m),
                                      dtype = x.dtype)
        return similarity_matrix
    
    def save_embedding(self, embedding, embed_type, dataset):
        filename = 'embeddings/' + embed_type + dataset + self.kernel_name
        np.savetxt(filename + '.csv', embedding, delimiter=",")
        
    def load_embedding(self, embed_type, dataset):
        filename = 'embeddings/' + embed_type + dataset + self.kernel_name
        return np.genfromtxt(filename + '.csv', delimiter=',')
        
        
    
    
                