import math 
import numpy as np
from ml_regression_utils import *


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
            raise Exception("Weight intialization can be either 'uniform' or 'normal' ")

        self.b = np.zeros(shape=(1,1))
        # neeed to add the r2 score
        self.history = {'loss': [], 'weights': [], 'bias': []}
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
            self.history['loss'].append(float(self._MSE()))

            if self.W.shape[1] == 1:
                self.history['weights'].append(float(self.W))
                self.history['bias'].append(float(self.b))
    






                            #######################################
                            #         Polynomial regression       #
                            #######################################



class PolynomialRegression(LinearRegression):
    def __init__(self, X, y, degree, weight_initializer = 'uniform'):
        super(PolynomialRegression, self).__init__(X, y, weight_initializer)
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
            raise Exception("Weight intialization can be either 'uniform' or 'normal' ") 

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




                            #######################################
                            #            Ridge regression         #
                            #######################################


class RidgeRegression(LinearRegression):
    def __init__(self, X, y, regularization_factor, weight_initializer='uniform'):
        super(RidgeRegression, self).__init__(X, y, weight_initializer)
        self.regularization_factor = regularization_factor

    def _ridge_loss(self):
        return self._MSE() + 1/(2*self.m) * L2_regularization(self.W, self.regularization_factor)


    def train(self, epochs=100, learning_rate=0.001, show_history=False):

        for epoch in range(1, epochs+1):
            w_grad, b_grad = self._compute_grads()
            w_grad +=  (self.regularization_factor * self.W)  

            self.W -= learning_rate * w_grad 
            self.b -= learning_rate * b_grad

            if epoch % 50 == 0 and show_history:
                if epoch % 50 == 0: print(f"After epoch {epoch} loss : {self._ridge_loss()}")
            self.history['loss'].append(int(self._ridge_loss()))






                            #######################################
                            #            Lasso regression         #
                            #######################################


class LassoRegression(LinearRegression):
    def __init__(self, X, y, regularization_factor, weight_initializer='uniform'):
        super(LassoRegression, self).__init__(X, y, weight_initializer)
        self.regularization_factor = regularization_factor

    def _lasso_loss(self):
        return self._MSE() + 1/(self.m) * L1_regularization(self.W, self.regularization_factor)

    def train(self, epochs=100, learning_rate=0.001, show_history=False):

        for epoch in range(1, epochs+1):
            w_grad, b_grad = self._compute_grads()
            w_grad += (self.regularization_factor * np.sign(self.W))  

            self.W -= learning_rate * w_grad 
            self.b -= learning_rate * b_grad

            if epoch % 50 == 0 and show_history:
                if epoch % 50 == 0: print(f"After epoch {epoch} loss : {self._lasso_loss()}")
            self.history['loss'].append(int(self._lasso_loss()))




                            #######################################
                            #         ElasticNet regression       #
                            #######################################



class ElasticNetRegression(LinearRegression):
    def __init__(self, X, y, alpha, beta, weight_initializef='uniform'):
        super(ElasticNetRegression, self).__init__(X, y)
        self.alpha = alpha 
        self.beta = beta 
        
    def _elastic_loss(self):
        return self._MSE() + 1/(self.m) * L1_regularization(self.W, self.alpha) + 1/(2*self.m) * L2_regularization(self.W, self.beta)
    
    def train(self, epochs=100, learning_rate=0.001, show_history=False):
        for epoch in range(1, epochs+1):
            w_grad, b_grad = self._compute_grads()
            w_grad += (self.alpha * np.sign(self.W)) + (self.beta * self.W)

            self.W -= learning_rate * w_grad 
            self.b -= learning_rate * b_grad

            if epoch % 50 == 0 and show_history:
                if epoch % 50 == 0: print(f"After epoch {epoch} loss : {self._elastic_loss()}")
            self.history['loss'].append(int(self._elastic_loss()))
        

                            #######################################
                            #      Polynomial Ridge regression    #
                            #######################################



class PolynomialRidgeRegression(LinearRegression):
    def __init__(self, X, y, degree, regularization_factor, weight_initializer='uniform'):
        super(PolynomialRidgeRegression, self).__init__(X, y, weight_initializer)

        self.regularization_factor = regularization_factor
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
            raise Exception("Weight intialization can be either 'uniform' or 'normal' ")

        
    def predict(self, X):
        X, _ = universal_reshape(X, X)
        if X.shape[0] != self.X.shape[0]:
            X = polynomial_features(X.T, self.degree).T 

        predictions = np.dot(self.W, X) + self.b
        return predictions  

    def _ridge_loss(self):
        return self._MSE() + 1/(2*self.m) * L2_regularization(self.W, self.regularization_factor)


    def train(self, epochs=100, learning_rate=0.001, show_history=False):

        for epoch in range(1, epochs+1):
            w_grad, b_grad = self._compute_grads()
            w_grad +=  (self.regularization_factor * self.W)  

            self.W -= learning_rate * w_grad 
            self.b -= learning_rate * b_grad

            if epoch % 50 == 0 and show_history:
                if epoch % 50 == 0: print(f"After epoch {epoch} loss : {self._ridge_loss()}")
            self.history['loss'].append(int(self._ridge_loss()))




                            #######################################
                            #           KNN regression            #
                            #######################################




class KNN_Regression:
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
            x_test = x_test.reshape(1, len(x_test))
            for (x_train, y_train) in zip(self.X_train, self.Y_train):
                x_train = x_train.reshape(1, len(x_train))
                dist = self._ecludian_distances(x_test, x_train)
                distances[float(dist)] = y_train
            
            k = 0
            k_dist = []
            for i in sorted(distances):
                k += 1
                k_dist.append(distances[i])
                if k == self.K:
                    break
            predictions.append(np.sum(np.array(k_dist))/self.K)
        
        predictions = np.array(predictions).reshape(1, len(predictions))
        return predictions


    
    def loss(self, predictions, ground_truth):
        _, ground_truth = universal_reshape(ground_truth, ground_truth)
        m = len(predictions)
        loss = 1/(2*m) * (np.sum((ground_truth - predictions), axis=1, keepdims=True)**2)
        return loss