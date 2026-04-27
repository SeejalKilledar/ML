import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as py

from sklearn.datasets import make_regression
X,y = make_regression(n_samples=100, n_targets=1, n_features=1, n_informative=1, noise=20)
plt.scatter(X,y)
plt.show()

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)
print(lr.coef_)
print(lr.intercept_)

m = 31.47

class GDRegressor:

    def __init__(self, learning_rate, epochs):
        self.m = 31.47
        self.b = -120
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self,X,y):
        # calculate b using GD
        for i in range(self.epochs):
            loss_slope = -2 * np.sum(y - self.m * X.ravel() - self.b)
            self.b = self.b - (self.lr * loss_slope)
        print(self.b)

gd = GDRegressor(0.001, 1000)
print(gd.fit(X,y))
