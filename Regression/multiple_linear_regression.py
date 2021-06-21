import numpy as np
import math
from numpy.core.fromnumeric import shape

class MultipleLinearRegression:
    def __init__(self, X, y):
        if X.shape[0] > X.shape[1]: 
            self.X = X.T
            self.y = y.T
        else:
            self.X = X
            self.y = y
        
        self.m = int(self.X.shape[1])
        '''
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, )) # can use this
        '''
        #self.W = np.random.randn(1, len(self.X))

        n_features = int(self.X.shape[0])
        limit = 1/math.sqrt(n_features)
        self.W = np.random.uniform(-limit, limit, (1, n_features))

        self.b = np.zeros(shape=(1,1))
        self.history = {'loss': []}
        self.count = 0

    def predict(self, X):
        if X.shape[0] > X.shape[1]:
            X = X.T
        predictions = np.dot(self.W, X) + self.b
        return predictions
    
    def _MSE(self):
        J = (self.y - (np.dot(self.W, self.X)+self.b)) ** 2
        mse = 1/(2*self.m) * (np.sum(J,axis=1, keepdims=True))
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
        w_grads = -2/self.m * np.dot((self.y - (np.dot(self.W, self.X)+self.b)), self.X.T) 
        b_grads = -2/self.m * np.sum((self.y - (np.dot(self.W, self.X)+self.b)), axis=1, keepdims=True)
        return (w_grads, b_grads)
    
    def train(self, epochs=100, learning_rate=0.001, show_history=False):
        for epoch in range(epochs):
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
                if epoch % 50 == 0: print(f"After epoch {epoch} loss : {self._MSE()}, weights: {self.W}, bias: {self.b}")
            self.history['loss'].append(int(self._MSE()))


    def train_recursive(self, learning_rate = 0.003):
        loss_before = self._MSE()

        w_grad, b_grad = self._compute_grads()
        self.W -= learning_rate * w_grad 
        self.b -= learning_rate * b_grad

        loss_after = self._MSE()

        print(loss_before, loss_after, "\n")
        self.count += 1
        if self.count % 2 == 0: print(f"after {self.count} loss: {self._MSE()}")
        
        if abs(loss_before - loss_after) == 0 : self.train_recursive()
        else: return



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

    #print(X_train.shape, y_train.shape)
    #print(X_test.shape, y_test.shape)

    mlr = MultipleLinearRegression(X_train, y_train)
    mlr.train(200, 0.03)

    model = LinearRegression().fit(X_train, y_train)
    pred_model = model.predict(X_test)

    pred_mlr = mlr.predict(X_test)
    #print(pred_model.shape, pred_mlr.shape)

    l_model = mlr.loss(pred_model, y_test)
    l_mlr = mlr.loss(pred_mlr, y_test)

    print(l_model, "\n", l_mlr)