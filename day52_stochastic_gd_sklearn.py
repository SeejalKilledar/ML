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


from sklearn.linear_model import SGDRegressor
reg = SGDRegressor(max_iter=100, learning_rate='constant', eta0=0.01)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(r2_score(y_test, y_pred))