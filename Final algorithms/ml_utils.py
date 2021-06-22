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
