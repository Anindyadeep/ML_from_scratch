import math 
import numpy as np 
from ml_regression_utils import universal_reshape
from ml_classification_utils import OneHotEncode

class LogisticRegression:

    """
    Logistic Regression

    parameters:
    ----------

    X : ndarray
                The features to be fed to the model
    y : ndarray
                The labels to train this model (should not be one hot encoded)

    weight_initializer : string
                                1. uniform (will initialize uniformly)
                                2. random (will initialize randomly)

    epochs : int
                The number iterations the model should be trained to get the optimal results
    
    learning_rate : float
                        This will determine by how much the gradient descent would be 
                        maintained by the model for better training 

    """
    def __init__(self, X, y, weight_initializer='random'):
        self.X_train, self.y_train = universal_reshape(X, y)
        self.y_train = OneHotEncode(self.y_train)

        if weight_initializer == 'random':
            self.W = np.random.randn(self.y_train.shape[1], self.X_train.shape[0]) * 1/math.sqrt(self.X_train.shape[0])
        
        elif weight_initializer == 'uniform':
            limit = 1/math.sqrt(self.X_train.shape[0])
            self.W = np.random.uniform(-limit, limit, (self.y_train.shape[1], self.X_train.shape[0]))
        else:
            raise Exception("No such initialization availabel")
        
        self.b = np.zeros(shape=(1,1))
    
    def _softmax(self, Z):
        if Z.shape[0] < Z.shape[1]:
            Z = Z.T 
        
        diff = np.max(Z, axis=1)
        diff = diff.reshape(len(diff), 1)
        exps = np.exp(Z-diff)
        sums = np.sum(exps, axis=1, keepdims=True)
        softmax = exps/sums 
        return softmax
    
    def _feed_forward(self):
        Z = np.dot(self.W, self.X_train) + self.b 
        A = self._softmax(Z)
        return (Z, A)
    
    def predict(self, X):
        X, _ = universal_reshape(X, X)

        Z = np.dot(self.W, X) + self.b 
        A = self._softmax(Z)
        probabilities = np.argmax(A, axis=1)
        return probabilities
    
    def log_loss(self, prediction, ground_truth):
        m = len(prediction)
        loss = -1/m * np.sum(ground_truth * np.log(prediction + 1e-8), keepdims=True)
        return loss

    def accuracy(self, prediction, ground_truth):
        prediction = np.argmax(prediction, axis=1)
        if ground_truth.shape[0] != 1:
            ground_truth = np.argmax(ground_truth, axis=1)
        m = len(prediction)
        return 1/m * np.sum(prediction == ground_truth)
    
    def _compute_grads(self, A):
        m = len(A)
        delta = A - self.y_train 

        W_grad = np.dot(delta.T, self.X_train.T)
        b_grad = 1/m * np.sum(delta.T, keepdims=True)

        return (W_grad, b_grad)
    
    def fit(self, epochs, learning_rate=0.001):
        for epoch in range(1, epochs+1):
            if epoch % 50  == 0:
                print(f"After epoch {epoch} loss: {loss} acc: {self.accuracy(A_train, self.y_train)}")

            Z_train, A_train = self._feed_forward()
            loss = self.log_loss(A_train, self.y_train)

            W_grad, b_grad = self._compute_grads(A_train)
            self.W -= learning_rate * W_grad 
            self.b -= learning_rate * b_grad 

if __name__ == '__main__':
    pass 