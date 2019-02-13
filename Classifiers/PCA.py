import numpy as np

class PCA:
    
    def __init__(self, kernel):
        self.kernel = kernel
        self.inputs = None
        self.centered_graham_matrix = None
        
    def perform_PCA(self, inputs, dim_num):
        self.inputs = inputs
        
        #compute centered graham matrix
        graham_matrix = self.kernel.matrix_from_data(self.inputs)
        print('graham_matrix',graham_matrix)
        n = graham_matrix.shape[0]
        I_U = np.identity(n) - np.ones((n,n))/n
        self.centered_graham_matrix = (I_U)@graham_matrix@(I_U)
        
        #compute the dim_num first eigen vectors of centered_graham _matrix
        eigen_values, eigen_vectors = np.linalg.eigh(self.centered_graham_matrix)

        idx = np.argsort(eigen_values)[::-1][:dim_num]
        s_eigen_values = eigen_values[idx]
        s_eigen_vectors = eigen_vectors[idx]

        projections = self.centered_graham_matrix@(np.divide(s_eigen_vectors,np.sqrt(s_eigen_values[:,None]))).T
        
        
        return projections
        
       
        
        
        