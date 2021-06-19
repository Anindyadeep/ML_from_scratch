import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
import matplotlib.pyplot as plt

'''
TODO: add batch learning
TODO: add the global minima target reach point
'''
'''
Things to change
--> add the universal predict function
--> add the universal loss function
--> solve the dimension problem
'''

class LinearRegression:
    def __init__(self, X, y):
        if X.shape != (1,len(X)): 
            self.X = X.T
            self.y = y.T
        else:
            self.X = X
            self.y = y

        self.W = np.random.randn(1,1)
        self.b = np.zeros(shape=(1,1))
        self.history = {'loss': []}
    
    def predict(self, X):
        predictions = np.dot(self.W, X) + self.b
        return predictions
    
    def MSE(self):
        m = len(self.X)
        mse = 1/(2*m) * (np.sum((self.y-self.predict(self.X)), axis=1, keepdims=True)**2)
        return mse
    
    def _gradient_descent(self):
        m = len(self.X)
        w_grad = -1/m * (np.sum(((self.X*self.y) - np.dot(self.W, self.X**2) + self.b * self.X), axis=1, keepdims=True))
        b_grad = -1/m * (np.sum((self.y - (np.dot(self.W, self.X) + self.b)), axis=1, keepdims=True))
        return (w_grad, b_grad)
    
    def _optimal_loss_point(self):
        m = len(self.X)
        X_bar = 1/m * (np.sum(self.X, axis=1, keepdims=True))
        y_bar = 1/m * (np.sum(self.y, axis=1, keepdims=True))

        numerator = (X_bar * y_bar) - (1/m * np.sum(self.y * self.X, axis = 1, keepdims=True))
        denominator = (X_bar ** 2) - (1/m * np.sum(self.X ** 2, axis=1, keepdims=True))

        W_opt = np.divide(numerator, denominator)
        b_opt = y_bar - (W_opt * X_bar)

        loss_opt = 1/(2*m) * (np.sum(self.y - (np.dot(W_opt, self.X) + b_opt), axis=1, keepdims=True)**2)
        return loss_opt

    def train(self, epochs = 100, learning_rate = 0.001):
        for epoch in range(epochs):
            self.MSE()
            w_grad, b_grad = self._gradient_descent()
            self.W -= learning_rate * w_grad
            self.b -= learning_rate * b_grad

            if epoch % 50 == 0: print(f"After epoch {epoch} loss : {self.MSE()}, weights: {self.W}")
            self.history['loss'].append(self.MSE())
        if int(self.MSE()) < int(self._optimal_loss_point()): print("reached less than optimal loss")
    

if __name__ == '__main__':
    X = np.random.randn(100,1)
    y = np.random.randint(2, size=(100,1))
    slr = LinearRegression(X, y)
    #print(int(slr.MSE()))
    #print(slr._optimal_loss_point())
    slr.train()