import math 
import numpy as np 

class LassoRegression:
    def __init__(self, X, y, regularization_factor):
        if X.shape[0] > X.shape[1]: 
            self.X = X.T
            self.y = y.T
        else:
            self.X = X
            self.y = y
        
        self.m = int(self.X.shape[1])
        n_features = int(self.X.shape[0])
        limit = 1/math.sqrt(n_features)

        self.W = np.random.uniform(-limit, limit, (1, n_features))
        self.b = np.zeros(shape=(1,1))
        self.regularization_factor = regularization_factor

        self.history = {'loss': []}
        self.count = 0

    def predict(self, X):
        if X.shape[0] > X.shape[1]:
            X = X.T
        predictions = np.dot(self.W, X) + self.b
        return predictions
    
    def _L1_regularization(self):
        return self.regularization_factor * np.linalg.norm(self.W)

    def _lasso_MSE(self):
        J = (self.y - (np.dot(self.W, self.X)+self.b)) ** 2
        mse = 1/(2*self.m) * (np.sum(J,axis=1, keepdims=True)) + self._L1_regularization()
        return mse 
    
    def loss(self, predictions, ground_truth):
        if predictions.shape[0] > predictions.shape[1]:
            predictions = predictions.T
        if ground_truth.shape[0] > ground_truth.shape[1]:
            ground_truth = ground_truth.T

        m = int(predictions.shape[1])
        loss = 1/(2*m) * (np.sum((ground_truth - predictions), axis=1, keepdims=True)**2)
        return loss
    
    def _compute_grads(self):
        w_grads = -2/self.m * np.dot((self.y - (np.dot(self.W, self.X)+self.b)), self.X.T) + (self.regularization_factor * self.W)
        b_grads = -2/self.m * np.sum((self.y - (np.dot(self.W, self.X)+self.b)), axis=1, keepdims=True)
        return (w_grads, b_grads)
    
    def train(self, epochs=100, learning_rate=0.001, show_history=False):
        for epoch in range(epochs):
            
            w_grad, b_grad = self._compute_grads()
            self.W -= learning_rate * w_grad 
            self.b -= learning_rate * b_grad

            if epoch % 50 == 0 and show_history:
                if epoch % 50 == 0: print(f"After epoch {epoch} loss : {self._lasso_MSE()}, weights: {self.W}, bias: {self.b}")
            self.history['loss'].append(int(self._lasso_MSE()))