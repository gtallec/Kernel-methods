import numpy as np
from tqdm import tqdm

class KernelMethodOptimizer:
    
    def __init__(self, model, kernel):
        self.model = model
        self.kernel = kernel
    
    def compute_test_accuracy(self, test_embedding, test_labels):
        test_predictions = self.model.predict(test_embedding)
        return 1 - (np.count_nonzero(test_predictions - test_labels)/test_predictions.shape[0])
    
    def test_routine(self, train_embedding, train_labels, test_embedding, test_labels, hyperParameters,i):
        self.model.setHyperParameters(hyperParameters)
        self.model.fit(train_embedding, train_labels)
        res = self.compute_test_accuracy(test_embedding, test_labels)
        
        print('test routine ', i, ' : ', res,'%')
        return self.compute_test_accuracy(test_embedding, test_labels)
    
    def k_fold_cross_validation(self, k, input_embedding, labels, hyperParameters):
        
        #The idea is to slice the input embedding into test and training for the k different fold
        n = input_embedding.shape[0]
        p = n//k
        testing_slices = np.arange(k*p).reshape((k,p))
        training_slices = np.zeros((k,p*(k-1)))
        for i in range(k):
            training_slices[i] = np.delete(np.arange(k*p), testing_slices[i])
            
        training_slices = training_slices.astype(int)
        testing_slices = testing_slices.astype(int)
              
        accuracy_list = [self.test_routine(input_embedding[np.ix_(training_slices[i], training_slices[i])],
                                          labels[training_slices[i]],
                                          input_embedding[np.ix_(training_slices[i], testing_slices[i])],
                                          labels[testing_slices[i]],
                                          hyperParameters,
                                           i
                                         )
                        for i in range(k)
                        ]
        print('accuracy list', accuracy_list)
        return np.mean(accuracy_list)
    
    def find_optimal_parameters(self, k, inputs, labels, hyperParametersList, dataset, load = False):
        input_embedding = None
        if load:
            input_embedding = self.kernel.load_embedding('train', dataset)
        else:
            input_embedding = self.kernel.matrix_from_data(inputs)
        self.kernel.save_embedding(input_embedding, 'train', dataset)
        return hyperParametersList[np.argmax([self.k_fold_cross_validation(k, input_embedding, labels, hyperParametersList[i])
                                          for i in range(hyperParametersList.shape[0])]),:]
    
    def make_optimal_prediction(self, k, inputs, tests, labels, hyperParametersList, dataset, load_train = False, load_test = False):
        
        if load_test:
            test_embedding = self.kernel.load_embedding('test', dataset)
        else:
            print('je suis en train de calculer')
            test_embedding = self.kernel.compute_prediction_embedding(inputs, tests)
            self.kernel.save_embedding(test_embedding, 'test', dataset)
        
        hyperParameters = self.find_optimal_parameters(k , inputs, labels, hyperParametersList, dataset, load_train)
        self.model.setHyperParameters(hyperParameters)
 
        return self.model.predict(test_embedding)

    def compute_embeddings(self, inputs, tests, dataset):
        n = inputs.shape[0]
        m = tests.shape[0]
        all_features = self.kernel.features_from_data(np.concatenate((inputs, tests)))
        train_features = all_features[:,:n]
        test_features = all_features[:,n:]
        
        test_embedding = train_features.T@test_features
        self.kernel.save_embedding(test_embedding, 'test', dataset)
        
        train_embedding = train_features.T@train_features
        self.kernel.save_embedding(train_embedding, 'train', dataset)
    
        
        