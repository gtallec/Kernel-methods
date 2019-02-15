import numpy as np
from tqdm import tqdm

class ModelOptimizer:
    
    def __init__(self, model):
        self.model = model
    
    def compute_test_accuracy(self, test_inputs, test_labels):
        test_predictions = self.model.predict(test_inputs)
        return 1 - (np.count_nonzero(test_predictions - test_labels)/test_predictions.shape[0])
    
    def test_routine(self, train_inputs, train_labels, test_inputs, test_labels, hyperParameters):
        self.model.setHyperParameters(hyperParameters)
        self.model.fit(train_inputs, train_labels.astype(np.double))
        return self.compute_test_accuracy(test_inputs, test_labels)
    
    def k_fold_cross_validation(self, k, inputs, labels, hyperParameters):
        
        n = inputs.shape[0]
        p = n//k
        testing_slices = np.arange(k*p).reshape((k,p))
        training_slices = np.zeros((k,p*(k-1)))
        for i in range(k):
            training_slices[i] = np.delete(np.arange(k*p), testing_slices[i])
            
        training_slices = training_slices.astype(int)
        testing_slices = testing_slices.astype(int)
        
        
        accuracy_list = [self.test_routine(inputs[training_slices[i]],
                                          labels[training_slices[i]],
                                          inputs[testing_slices[i]],
                                          labels[testing_slices[i]],
                                          hyperParameters
                                         )
                        for i in tqdm(range(k))
                       ]
        print('accuracy list', accuracy_list)
        return np.mean(accuracy_list)
    
    def find_optimal_parameters(self, k, inputs, labels, hyperParametersList):
        return hyperParametersList[np.argmax([self.k_fold_cross_validation(k, inputs, labels, hyperParametersList[i])
                                          for i in range(hyperParametersList.shape[0])]),:]
    
        
        