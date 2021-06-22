import itertools as itr
import numpy as np

def polynomial_features(X, degree):
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


X = np.array([[1,2]*1]*2)
print(X.shape)
n_features, n_samples = X.shape
degree = 2

combinations = [itr.combinations_with_replacement(range(n_features), 1)]
print(combinations)

print(polynomial_features(X.T, 2))

