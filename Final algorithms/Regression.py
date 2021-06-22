import math 
import numpy as np
from ml_utils import *

class LinearRegression:
    def __init__(self, X, y, weight_initializer = 'uniform', ):
        '''
        LINEAR REGRESSION CLASS

        Two parameters are to be inserted here one 
        is the Features (X) and other is the lables (y)\n
        NOTE:\n
        you can intialize your X, y in any ways and this class will figure 
        that out by itself, but remmember that the dimension of the X, y 
        accepted here will be:

        X ∈ (n, m) | where (n) --> no of features and (m) --> no of examples\n
        y ∈ (1, m) | where (m) --> no of the examples

        Other parameters: 
        weight_initializer :\n 
                   'uniform' : (default) is that is the weight will
                                be will be intialised as uniform gaussian distribution\n
                    'random' :  means that the weights will be initialised randomly
        '''

        self.X, self.y = universal_reshape(X, y)
        
        self.m = self.X.shape[1]
        n_features = self.X.shape[0]

        if weight_initializer == 'uniform':
            limit = 1/(math.sqrt(n_features))
        
            self.W = np.random.uniform(-limit, limit, (1, n_features))

        elif weight_initializer == 'random':
            self.W = np.random.randn(1, n_features)
        
        else:
            print('error')

        self.b = np.zeros(shape=(1,1))
        self.history = {'loss': []}
        self.count = 0

    def predict(self, X):
        X, _ = universal_reshape(X, X)
        predictions = np.dot(self.W, X) + self.b
        return predictions
    
    def _MSE(self):
        J = (self.y - (np.dot(self.W, self.X)+self.b)) ** 2
        mse = 1/(2*self.m) * (np.sum(J,axis=1, keepdims=True))
        return mse 
    
    def loss(self, predictions, ground_truth):
        predictions, _ = universal_reshape(predictions, predictions)
        ground_truth, _ = universal_reshape(ground_truth, ground_truth)

        if predictions.shape[0] > predictions.shape[1]:
            predictions = predictions.T
        if ground_truth.shape[0] > ground_truth.shape[1]:
            ground_truth = ground_truth.T

        loss = 1/(2*self.m) * (np.sum((ground_truth - predictions), axis=1, keepdims=True)**2)
        return loss
    
    def _compute_grads(self):
        w_grads = -2/self.m * np.dot((self.y - (np.dot(self.W, self.X)+self.b)), self.X.T) 
        b_grads = -2/self.m * np.sum((self.y - (np.dot(self.W, self.X)+self.b)), axis=1, keepdims=True)
        return (w_grads, b_grads)

    def train(self, epochs=100, learning_rate=0.001, show_history=False):
        for epoch in range(1, epochs+1):
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
    #from sklearn.linear_model import LinearRegression

    n_features = 10
    X, y = datasets.make_regression(n_samples=600, n_features=n_features, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    regressor = LinearRegression(X_train, y_train)
    regressor.train(epochs=200, learning_rate=0.03, show_history=True)

    predictions = regressor.predict(X_test)
    print(regressor.loss(predictions, y_test))