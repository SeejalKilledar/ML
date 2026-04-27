import pandas as pd
from PIL.ImagePalette import random
from sklearn.datasets import make_regression, load_diabetes
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import random



X,y = load_diabetes(return_X_y=True)
print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)


from sklearn.linear_model import SGDRegressor
reg = SGDRegressor(learning_rate='constant', eta0=0.1)
batch_size = 35

for i in range(100):
    idx = random.sample(range(X_train.shape[0]),batch_size)
    reg.partial_fit(X_train[idx], y_train[idx])

print(reg.coef_)
print(reg.intercept_)
y_pred = reg.predict(X_test)
print(r2_score(y_test, y_pred))