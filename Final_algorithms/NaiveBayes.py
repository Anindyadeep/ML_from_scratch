import math 
import numpy as np 
from ml_regression_utils import universal_reshape

class GaussianNaiveBayes:
    def __init__(self, X, y):
        """
        Gaussian Naive Bayes:
        --------------------

        Parameters:
        X: nd array of shape: (m, n) where m = number of examples, n = number of featuers

        """
        self.X, self.y = X, y 
        self._classes, self._class_counts = np.unique(self.y, return_counts=True)
        self._num_samples, self._num_features = self.X.shape
        self._mean = np.zeros(shape=(len(self._classes), self._num_features))
        self._var =  np.zeros(shape=(len(self._classes), self._num_features))
        self._prior = np.zeros(shape=(len(self._classes)))

        for _class in self._classes:
            X_class = self.X[self.y == _class]
            self._mean[_class, :] = np.mean(X_class, axis=0)
            self._var[_class, :] = np.var(X_class, axis=0)
            self._prior[_class] = self._class_counts[_class] / float(np.sum(self._class_counts))
        
    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self._predict(x))
        return np.array(predictions)


    def _predict(self, x):
        prosteriors = []
        for idx in self._classes:
            likelihood = np.sum(np.log(self.getLikelihoodFromGaussianFunction(x, idx)))
            prior = self._prior[idx]
            prosterior = likelihood + prior
            prosteriors.append(prosterior)
        return self._classes[np.argmax(prosteriors)]    
    

    def getLikelihoodFromGaussianFunction(self,x,idx):
        """
        Vectorised impleamentation of the Gaussian Distriution Function such that, it will return the 
        value of the P(Xi|y = y) in an numpy array at once, making the computation a bit optimized 
        """
        constant = np.sqrt(2 * math.pi * self._var[idx])
        exp_numerator = -1 * ((x - self._mean[idx])**2)
        exp_denominator = 2 * self._var[idx]
        fraction = np.divide(exp_numerator, exp_denominator)
        exp = np.exp(fraction)
        return np.divide(exp, constant)
    
    def accuracy(self, prediction, ground_truth):
        return np.sum(prediction == ground_truth)/len(ground_truth)
    
