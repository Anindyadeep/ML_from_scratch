import numpy as np
import matplotlib.pyplot as plt 
import itertools as  itr
import math

'''
This function below is been temporarily taken from
the github: https://github.com/eriklindernoren/ML-From-Scratch/blob/a2806c6732eee8d27762edd6d864e0c179d8e9e8/mlfromscratch/utils/data_manipulation.py#L61
As I will understand the concept, I will replace with my own logic
'''

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


class PolynomialRegression:
    def __init__(self, X, y, degree):
        if X.shape[0] > X.shape[1]: 
            self.X = X.T
            self.y = y.T
        else:
            self.X = X
            self.y = y
        self.degree = degree
        self.X_new = polynomial_features(self.X.T, self.degree).T
        n_features = int(self.X_new.shape[0])
        limit = 1/math.sqrt(n_features)

        self.W = np.random.uniform(-limit, limit, (1, n_features))
        self.b = np.zeros(shape=(1,1))
        self.history = {'loss': []}
        self.count = 0
        self.m = int(self.X_new.shape[1])
    
    def predict(self, X):
        if X.shape[0] > X.shape[1]:
            X = X.T
        X_new = polynomial_features(X.T, self.degree).T
        predictions = np.dot(self.W, X_new) + self.b
        return predictions
    
    def _MSE(self):
        J = (self.y - (np.dot(self.W, self.X_new)+self.b)) ** 2
        mse = 1/(2*self.m) * (np.sum(J,axis=1, keepdims=True))
        return mse 
    
    def loss(self, predictions, ground_truth):
        if predictions.shape[0] > predictions.shape[1]:
            predictions = predictions.T
        if ground_truth.shape[0] > ground_truth.shape[1]:
            ground_truth = ground_truth.T

        m = int(predictions.shape[1])
        loss = (1/(2*m) * (np.sum((ground_truth - predictions), axis=1, keepdims=True)**2)) + 1e-8
        return loss
    
    def _compute_grads(self):
        w_grads = -2/self.m * np.dot((self.y - (np.dot(self.W, self.X_new)+self.b)), self.X_new.T) 
        b_grads = -2/self.m * np.sum((self.y - (np.dot(self.W, self.X_new)+self.b)), axis=1, keepdims=True)
        return (w_grads, b_grads)

    def train(self, epochs=100, learning_rate=0.001, show_history=False):
        for epoch in range(epochs+1):
            '''
            TODO: find an algorithm such that the user do not need to tune the epochs, it would be done
            automatically by the algorithm such that if the loss at nth epoch (last) > loss at (n+1)th epoch
            the it would run for more steps, tuning the leraning rate as well
            '''
            '''TODO: IMP DO THE LOSS AND ALL w.r.t THE VALIDATION LOSS'''
            
            w_grad, b_grad = self._compute_grads()
            self.W -= learning_rate * w_grad 
            self.b -= learning_rate * b_grad

            if epoch % 50 == 0 and show_history:
                if epoch % 50 == 0: print(f"After epoch {epoch} loss : {self._MSE()}")
            self.history['loss'].append(int(self._MSE()))


if __name__ == '__main__':

    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression

    n_features = 10
    X, y = datasets.make_regression(n_samples=600, n_features=n_features, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    y_train = y_train.reshape(len(y_train), 1)
    X_test = X_test.reshape(len(X_test), n_features)
    y_test = y_test.reshape(len(y_test), 1)

    mlr = PolynomialRegression(X_train, y_train, 4)
    mlr.train(100, 0.03, show_history=True)

    pred_mlr = mlr.predict(X_test)
    l_mlr = mlr.loss(pred_mlr, y_test)
    print(l_mlr)