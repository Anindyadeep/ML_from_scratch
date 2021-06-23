import math 
import itertools as itr
import numpy as np

def universal_reshape(X, y):
        if len(X.shape) == 1:
            X = X.reshape(1, len(X))

        if X.shape[0] > X.shape[1]: 
            X = X.T

        if len(y.shape) == 1:
            y = y.reshape(1, len(y))
            
        if y.shape[0] > y.shape[1]:
            y = y.T

        return X, y



def polynomial_features(X, degree):
    '''
    This function below is been temporarily taken from
    the github: https://github.com/eriklindernoren/ML-From-Scratch/blob/a2806c6732eee8d27762edd6d864e0c179d8e9e8/mlfromscratch/utils/data_manipulation.py#L61
    As I will understand the concept, I will replace with my own logic
    '''
    n_samples, n_features = np.shape(X)

    def index_combinations():
        combs = [itr.combinations_with_replacement(range(n_features), i) for i in range(0, degree + 1)]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs
    
    combinations = index_combinations()
    n_output_features = len(combinations)
    X_new = np.empty((n_samples, n_output_features))
    
    for i, index_combs in enumerate(combinations):  
        X_new[:, i] = np.prod(X[:, index_combs], axis=1)

    return X_new



if __name__ == '__main__':
    '''
    testing the 'universal_reshape' function
    '''
    # test case 1
    X1 = np.random.randn(200,3)
    y1 = np.random.randn(100, 1)

    X1_ans, y1_ans = universal_reshape(X1, y1)
    print(X1_ans.shape, y1_ans.shape, "\n")

    # test case 2
    X2 = np.random.randn(3, 200)
    y2 = np.random.randn(1, 100)

    X2_ans, y2_ans = universal_reshape(X2, y2)
    print(X2_ans.shape, y2_ans.shape, "\n")

    # test case 3
    X3 = np.random.randn(200)
    y3 = np.random.randn(100)

    X3_ans, y3_ans = universal_reshape(X3, y3)
    print(X3_ans.shape, y3_ans.shape, "\n")

    # test case 4
    '''
    under developement
    X4 = np.random.randn(200)
    y4 = np.random.randn(100)

    X4_ans, _ = universal_reshape(X4)
    print(X4_ans.shape)
    print(_)
    '''
