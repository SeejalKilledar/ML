import pandas as pd
import numpy as np


class linear_regression_from_scratch:
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, X_train, y_train):
        numerator = 0
        denometer = 0
        X_train = X_train.values.flatten()
        y_train = y_train.values.flatten()
        for i in range(X_train.shape[0]):
            numerator = numerator + ((X_train[i] - X_train.mean()) * (y_train[i] - y_train.mean()))
            denometer = denometer + ((X_train[i] - X_train.mean()) * (X_train[i] - X_train.mean()))

        self.m = numerator / denometer
        self.b = y_train.mean() - (self.m * X_train.mean())
        print(self.m)
        print(self.b)

    def predict(self, X_test):
        X_test = X_test.values.flatten()
        return self.m * X_test + self.b



df = pd.read_csv('placement_simple_linear_reg.csv')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('package', axis = 1), df['package'], test_size = 0.2, random_state = 42)
print(X_train.shape)

lr = linear_regression_from_scratch()
lr.fit(X_train, y_train)
lr.predict(X_test.iloc[[0]])
print(X_test.iloc[[0]])