import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X,y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=45)

from sklearn.linear_model import LinearRegression
L =LinearRegression()
L.fit(X_train,y_train)
y_pred = L.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error
print(r2_score(y_test, y_pred))
print(np.sqrt(mean_squared_error(y_test, y_pred)))

from sklearn.linear_model import Ridge
R = Ridge(alpha=0.0001)

R.fit(X_train,y_train)
y_pred1 = R.predict(X_test)
print(r2_score(y_test, y_pred1))
print(np.sqrt(mean_squared_error(y_test, y_pred1)))







