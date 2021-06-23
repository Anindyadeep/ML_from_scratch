import math 
import numpy as np
from ml_utils import *


                            #######################################
                            #          Linear regression          #
                            #######################################



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
        # neeed to add the r2 score
        self.history = {'loss': []}
        self.count = 0

    def predict(self, X):
        '''
        Input: X (feature), where the class can automatically 
        can control the dimensionality, and it will return the predictions
        after performing the regression from the trained weights and bias
        '''
        X, _ = universal_reshape(X, X)
        predictions = np.dot(self.W, X) + self.b
        return predictions
    
    def _MSE(self):
        '''
        This function will calculate the Mean Squared Error loss
        of the model, where \n
        y_hat (i.e. predicted) = W . X + b, \n
        y -->  (ground truth value) \n

        loss (L) = (y-y_hat) ^ 2
        cost (J) = ∑L (from i = 1, m) where m = total no: of examples.
        '''
        J = (self.y - (np.dot(self.W, self.X)+self.b)) ** 2
        mse = 1/(2*self.m) * (np.sum(J,axis=1, keepdims=True))
        return mse 
    
    def loss(self, predictions, ground_truth):
        '''
        Input: (predictions), (ground_truth) \n
        Output: Total Loss \n 

        Where, Loss (L) = (predictions - ground_truth) ^ 2 \n
        and Cost (J) = ∑L (from i = 1, m) where m = total no: of examples.
        '''
        predictions, _ = universal_reshape(predictions, predictions)
        ground_truth, _ = universal_reshape(ground_truth, ground_truth)

        if predictions.shape[0] > predictions.shape[1]:
            predictions = predictions.T
        if ground_truth.shape[0] > ground_truth.shape[1]:
            ground_truth = ground_truth.T

        loss = 1/(2*self.m) * (np.sum((ground_truth - predictions), axis=1, keepdims=True)**2)
        return loss
    
    def _compute_grads(self):
        '''
        This function will compute the gradients of the weights and the bias w.r.t. the loss \n
        where: \n
        
        dJ/dW = -2/m * (Y-(W.X + b) . transpose(X)) 
        dJ/db = -2/m * ∑ (Y-*W.X + b)

        W := W - (learning rate) * dJ/dW
        b := b - (learning rate) * dJ/db
        '''
        w_grads = -2/self.m * np.dot((self.y - (np.dot(self.W, self.X)+self.b)), self.X.T) 
        b_grads = -2/self.m * np.sum((self.y - (np.dot(self.W, self.X)+self.b)), axis=1, keepdims=True)
        return (w_grads, b_grads)

    def train(self, epochs=100, learning_rate=0.001, show_history=False):
        '''
        Required parameters : None \n
        Tweakable parameters: \n
        (epochs) : for how many iterations the train will happen \n
        (learning_rate) : a custom learning rate can be sometime useful \n
        (show_history) : will return the history of losses and any other metrics \n

        NOTE: \n
        It can retrive the model history by calling : model.history['loss'] or any other metric
        as it is in the form of the dictionary.

        '''
        for epoch in range(1, epochs+1):
            '''
            TODO: 
                    1. IMP DO THE LOSS AND ALL w.r.t THE VALIDATION LOSS
                    2. More optimized learning rate using grid search and other techniques
            '''
            
            w_grad, b_grad = self._compute_grads()
            self.W -= learning_rate * w_grad 
            self.b -= learning_rate * b_grad

            if epoch % 50 == 0 and show_history:
                if epoch % 50 == 0: print(f"After epoch {epoch} loss : {self._MSE()}")
            self.history['loss'].append(int(self._MSE()))
    






                            #######################################
                            #         Polynomial regression       #
                            #######################################



class PolynomialRegression(LinearRegression):
    def __init__(self, X, y, degree, weight_initializer = 'uniform'):
        super(PolynomialRegression, self).__init__(X, y)
        X_temp, self.y = universal_reshape(X, y)
        self.degree = degree
        self.X = polynomial_features(X_temp.T, self.degree).T

        n_features = int(self.X.shape[0])
        limit = 1/math.sqrt(n_features)

        if weight_initializer == 'uniform':
            self.W = np.random.uniform(-limit, limit, (1, n_features))
        elif weight_initializer == 'random':
            self.W = np.random.randn(1, n_features)
        else:
            print('error')

        self.b = np.zeros(shape=(1,1))
        # neeed to add the r2 score
        self.history = {'loss': []}
        self.count = 0
        self.m = int(self.X.shape[1])     

    def predict(self, X):
        '''
        Input: X (feature), where the class can automatically 
        can control the dimensionality, and it will return the predictions
        after performing the regression from the trained weights and bias
        '''
        X, _ = universal_reshape(X, X)
        if X.shape[0] != self.X.shape[0]:
            X = polynomial_features(X.T, self.degree).T 

        predictions = np.dot(self.W, X) + self.b
        return predictions    