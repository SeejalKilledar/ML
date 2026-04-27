import pandas as pd
from sklearn.datasets import make_regression, load_diabetes
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



X,y = load_diabetes(return_X_y=True)
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)
lr = LinearRegression()
lr.fit(X_train,y_train)
print(lr.coef_)
print(lr.intercept_)

y_pred = lr.predict(X_test)
print(r2_score(y_test, y_pred))

class GDRegressor:

    def __init__(self, learning_rate = 0.01, epochs = 100):
        self.coef = None
        self.intercept = None
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self,X_train, y_train):
        # initialize the coeff
        self.intercept = 0 # Beta 0
        self.coef = np.ones(X_train.shape[1])
        print(X_train.shape[1])


        for i in range(self.epochs):
            #update all the coefs and the intercepts
            y_hat =  np.dot(X_train, self.coef) + self.intercept
            #print(y_hat.shape)
            intercept_der = -2 * np.mean(y_train-y_hat)
            self.intercept = self.intercept - (self.lr * intercept_der)


            coef_der = -2 * np.dot((y_train-y_hat),X_train)/X_train.shape[0]
            self.coef = self.coef - (self.lr + coef_der)


        print(self.intercept, self.coef)
    def predict(self,X_test):
        return np.dot(X_test, self.coef) + self.intercept


gdr = GDRegressor(learning_rate=0.1,epochs=800)
gdr.fit(X_train,y_train)

y_pred = gdr.predict(X_test)
print(r2_score(y_test, y_pred))