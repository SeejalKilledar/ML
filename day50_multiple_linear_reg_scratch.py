import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


(X,y) = load_diabetes(return_X_y=True) # X is input and y is output
print(X.shape)

print(y)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,
                                                    random_state=2)
print(X_train.shape)
print(X_test.shape)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_predict = lr.predict(X_test)
from sklearn.metrics import r2_score
print(r2_score(y_test, y_predict))

print(lr.coef_)
print(lr.intercept_)

print("Making our own Linear Regression class")
class MyLR:
    def __init__(self):
        self.coef_ =None
        self.intecept_ = None

    def fit(self, X_train, y_train):
        """
        : Calculate beta = (inverse(transpose(X_train)*X_train)) * (transpose(X_train)*y_train)
        :param X_train: adding 1 to the matrix, hence the shape is (353,11)
        :param y_train:
        :return:
        """
        X_train = np.insert(X_train, 0,1, axis=1)

        # calculate coeff
        betas = np.linalg.inv(np.dot(X_train.T, X_train)).dot(X_train.T).dot(y_train)
        print(betas)
        self.intercept = betas[0]
        self.coef_ = betas[1:]

    def predict(self, X_test):
        y_pred = np.dot(X_test, self.coef_) + self.intercept
        return y_pred


my_lr = MyLR()

my_lr.fit(X_train, y_train)

print(X_train.shape)

print(np.insert(X_train, 0,1, axis=1).shape)

y_pred = my_lr.predict(X_test)

print(r2_score(y_test, y_pred))

print(my_lr.coef_)

