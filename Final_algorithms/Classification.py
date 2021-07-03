import math
import numpy as np 
from ml_regression_utils import universal_reshape
from ml_classification_utils import OneHotEncode



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






                            #######################################
                            #      Binary Logistic Regression     #
                            #######################################



class LogisticRegression:

    """
    Binary Logistic Regression Classifier:

    parameters:
    ----------

    epochs : int
                The number iterations the model should be trained to get the optimal results
    
    learning_rate : float
                        This will determine by how much the gradient descent would be 
                        maintained by the model for better training
    """

    def __init__(self, X, y, weight_initializer='uniform'):
        self.X, self.y = universal_reshape(X,y)
        if weight_initializer == 'uniform':
            n_features = self.X.shape[0]
            limit = 1/(math.sqrt(n_features))
            self.W = np.random.uniform(-limit, limit, (1, n_features))
        
        elif weight_initializer == 'random':
            self.W = np.random.randn(1, n_features)
        
        else:
            raise Exception('No such Initialization')
        
        self.b = np.zeros(shape=(1,1))

    def _feed_forward(self):
        Z = np.dot(self.W, self.X) + self.b
        return Z
    
    def _sigmoid(self, Z):
        A = 1 / (1 + np.exp(-Z))
        return A
    
    def predict(self, X, round=True):
        X, _ = universal_reshape(X, X)
        Z = np.dot(self.W, X) + self.b
        predictions = self._sigmoid(Z)
        
        if round:
            return np.round(predictions)
        else:
            return predictions
    
    def compute_accuracy(self, predictions, ground_truth):

        if predictions.shape[0] != 1:
            predictions = predictions.reshape(1, len(predictions))
        
        if ground_truth.shape[0] != 1:
            ground_truth = ground_truth.reshape(1, len(ground_truth))

        number_example = prediction.shape[1]
        accuracy = (np.sum(np.round(prediction) == ground_truth)) / number_example
        return accuracy
    

    def _binary_cross_entropy(self, Y_pred):
        m = Y_pred.shape[1]
        cost = -1 / m * np.sum(self.y * np.log(Y_pred) + (1 - self.y) * (np.log(1 - Y_pred)))
        return cost
    
    
    def _compute_grads(self, Y_pred):
        W_grad = np.dot((Y_pred-self.y), self.X.T)
        b_grad = np.sum(Y_pred-self.y, axis=1, keepdims=True)
        
        return (W_grad, b_grad)
    
    def train(self, learning_rate = 0.01, epochs = 100):
        for epoch in range(epochs):
            Z_pred = self._feed_forward()
            Y_pred = self._sigmoid(Z_pred)
            
            cost = self._binary_cross_entropy(Y_pred)
            W_grad, b_grad = self._compute_grads(Y_pred)
            
            self.W -= learning_rate * W_grad
            self.b -= learning_rate * b_grad
            
            if epoch % 50 == 0:
                print(f"After epoch {epoch} cost: {cost}, acc: {self.compute_accuracy(Y_pred, self.y)}")





                            #######################################
                            #    Multi class Logistic Regression  #
                            #######################################


"""

NOTE: This part is under developement, one fixed finally 
    a final class called 'LogisticRegression' will be created as
    more general in its nature

"""

class MultClassLogisticRegression:
    def __init__(self, X, y):
        self.X_train, self.y_train  = universal_reshape(X, y)
        self.y_train = OneHotEncode(self.y_train)
        
        self.W = np.random.randn(self.y_train.shape[1], self.X_train.shape[0])
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
        # make an another universal reshape fucntion for one hot encodes speacially

        prediction = np.argmax(prediction, axis=1)
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
            Z_train, A_train = self._feed_forward()
            loss = self.log_loss(A_train, self.y_train)

            W_grad, b_grad = self._compute_grads(A_train)
            self.W -= learning_rate * W_grad 
            self.b -= learning_rate * b_grad 

            if epoch % 50  == 0:
                print(f"After epoch {epoch} loss: {loss} acc: {self.accuracy(A_train, self.y_train)}")



if __name__ == '__main__':
  import numpy as np
  import pandas as pd
  from sklearn.model_selection import train_test_split

  data = pd.read_csv(r'/content/sample_data/mnist_train_small.csv')
  features, labels = np.array(data.iloc[:,1:]), np.array(data.iloc[:, :1])
  X_train, X_test, y_train, y_test = train_test_split(features, labels)
  X_train = (X_train - np.mean(X_train)) / np.std(X_train)
  X_test = (X_test - np.mean(X_test)) / np.std(X_test)
  
  regressor = MultClassLogisticRegression(X_train, y_train)
  regressor.fit(epochs=50, learning_rate=0.03)

  predictions = regressor.predict(X_test)
  predictions = predictions.reshape(len(predictions), 1)

  print(predictions.shape, y_test.shape)
  print(regressor.accuracy(predictions, y_test))


