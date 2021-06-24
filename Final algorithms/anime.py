from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import Regression as r 
import sklearn.linear_model as lm 
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np 

n_features = 1
X, y = make_regression(n_samples=600, n_features=n_features, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# our method
regressor_r = r.LinearRegression(X_train, y_train)
regressor_r.train(epochs=200, learning_rate=0.03)

# sklearn's method
regressor_lm = lm.LinearRegression().fit(X_train, y_train)

print("TEST DATA LOSS OF OUR MODEL: ", regressor_r.loss(regressor_r.predict(X_test), y_test))
print("TEST DATA LOSS OF OUR SKLEARN's: ", regressor_r.loss(regressor_lm.predict(X_test), y_test))


weights = regressor_r.history['weights']
bias = regressor_r.history['bias']

def get_line_history(weights, bias, x):
    lines = []
    for i in range(len(weights)):
        line_pt = x * weights[i] + bias[i]
        lines.append(line_pt)
    return lines

lines = get_line_history(weights, bias, X_test)

fig, ax = plt.subplots()
x = X_test.reshape(len(X_test),)
y = y_test.reshape(len(y_test),)

line, = ax.plot(x, y)

plt.plot(X_test.reshape(len(X_test),), 'o', label='Training data')

def animate(i):
    print(i)
    x = X_test.reshape(len(X_test),)
    y = lines[i].reshape(len(lines[i]),)

    #line.set_xdata(x)
    line.set_ydata(y)
    return line,

def init():
    line.set_ydata(y)
    return line,

ani = animation.FuncAnimation(fig, animate, frames=np.arange(1, 20), interval=100)
plt.show()

