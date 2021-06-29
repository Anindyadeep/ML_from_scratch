import numpy as np 


class KNN:
    def __init__(self, K):
        self.K = K
    
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
    
    def accuray(self, predictions, y_test):
        return (np.sum(predictions == y_test))/len(y_test)
    
    
if __name__ == '__main__':
    from sklearn.datasets import make_classification, load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier

    print("TEST 1")

    X, y = make_classification(n_samples=200, n_features=6)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1234)

    model = KNN(K = 3)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("MY MODEL: ", model.accuray(preds, y_test))

    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)
    preds = classifier.predict(X_test)

    classifier.fit(X_train, y_train)
    print("SKLEARN: ", model.accuray(preds, y_test))

    print("\nTEST 2")

    data = load_iris()
    X, y = data['data'], data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1234)

    iris_model = KNN(K = 3)
    iris_model.fit(X_train, y_train)
    model_preds = iris_model.predict(X_test)
    print("MY MODEL: ", model.accuray(model_preds, y_test))

    iris_sk = KNeighborsClassifier(n_neighbors=3)
    iris_sk.fit(X_train, y_train)
    sk_preds = iris_sk.predict(X_test)
    print("SKLEARN: ", model.accuray(sk_preds, y_test))
