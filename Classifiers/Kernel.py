import numpy as np

class Kernel:
    
    def __init__(self, kernel_fun = lambda x,y : x*y):
        self.kernel_fun = kernel_fun
        
    def matrix_from_data(self, inputs):
        eps = 10e-7
        n = inputs.shape[0]
        
        kernel_matrix = np.fromfunction(np.vectorize(lambda i,j : self.kernel_fun(inputs[int(i)], inputs[int(j)])), shape = (n,n), dtype = inputs.dtype)
        print('Kernel Matrix computed', kernel_matrix.shape)
        return kernel_matrix + eps*np.identity(n)
    
    def compute_prediction_matrix(self, inputs, input_to_predict):
        n = inputs.shape[0]
        m = input_to_predict.shape[0]
        
        pred_matrix = np.zeros((n,m))
        for i in range(n):
            for j in range(m):
                pred_matrix[i,j] = self.kernel_fun(inputs[i], input_to_predict[j])
        return pred_matrix
                