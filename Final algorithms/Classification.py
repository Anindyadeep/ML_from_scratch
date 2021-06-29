import numpy as np 
from ml_regression_utils import universal_reshape


                            #######################################
                            #          K-Nearest Neighbors        #
                            #######################################

class KNearestNeighborsClassifier:
    """
    K-Nearest Neighbors Classifier:

    parameters:
    ----------

    n_neighbors : int
                  The number of closest neighbors that will determine the class of the 
                  sample.
    """

    def __init__(self, n_neighbors):
        self.K = n_neighbors
    
    def fit(self, X, y):
        self.X_train = X 
        self.Y_train = y 
    
    
    def _ecludian_distances(self, x1, x2):
        if x1.shape[0] != 1:
            x1 = x1.reshape(1, len(x1))
        if x2.shape[0] != 1:
            x2 = x2.reshape(1, len(x2))
        
        dist = np.sum((x1 - x2)**2, axis = 1, keepdims=True)
        dist = np.sqrt(dist)
        return dist
    
    def predict(self, X_test):
        predictions = []
        for x_test in X_test:
            distances = {}
            for (x_train, y_train) in zip(self.X_train, self.Y_train):
                dist = self._ecludian_distances(x_test, x_train)
                distances[float(dist)] = y_train 

            k = 0
            k_dist = []

            for i in sorted(distances):
                k += 1
                k_dist.append(distances[i])
                if k == self.K:
                    break 
            
            k_nearest = np.array(k_dist)
            predictions.append(np.bincount(k_nearest).argmax())
        
        return predictions
    

    def compute_accuracy(self, predictions, ground_truth):
        return (np.sum(predictions == y_test))/len(y_test)
