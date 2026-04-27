import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X,y = load_diabetes(return_X_y=True)
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)
reg = LinearRegression()
reg.fit(X_train, y_train)

print(reg.coef_)
print(reg.intercept_)

y_pred = reg.predict(X_test)
print(r2_score(y_test,y_pred))


import random
class MBGDRegressor:

    def __init__(self, batch_size, learning_rate = 0.01, epochs = 100):
        self.coef = None
        self.intercept = None
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self,X_train, y_train):
        # initialize the coeff
        self.intercept = 0 # Beta 0
        self.coef = np.ones(X_train.shape[1])
        print(X_train.shape[1])


        for i in range(self.epochs):
            #update all the coefs and the intercepts

            """
            No of batches = (Total no of rows)/batch_size
            """

            for j in range(int(X_train.shape[0]/self.batch_size)):

                idx = random.sample(range(X_train.shape[0]), self.batch_size)

                y_hat =  np.dot(X_train[idx], self.coef) + self.intercept
                #print(y_hat.shape)
                intercept_der = -2 * np.mean(y_train[idx]-y_hat)
                self.intercept = self.intercept - (self.lr * intercept_der)


                coef_der = -2 * np.dot((y_train[idx]-y_hat),X_train[idx])
                self.coef = self.coef - (self.lr + coef_der)


        print(self.intercept, self.coef)
    def predict(self,X_test):
        return np.dot(X_test, self.coef) + self.intercept


mbr = MBGDRegressor(batch_size=int(X_train.shape[0]/10), learning_rate=0.01, epochs=50)
mbr.fit(X_train,y_train)

y_pred = mbr.predict(X_test)
print(r2_score(y_test, y_pred))