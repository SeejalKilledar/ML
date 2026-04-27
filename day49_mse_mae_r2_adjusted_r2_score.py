

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("placement_simple_linear_reg.csv")
print(df.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(df.drop('package', axis=1),
                                                    df['package'],
                                                    test_size = 0.2,
                                                    random_state=42)
plt.scatter(x = 'cgpa', y = 'package', data = df)
plt.show()

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(f'R2 score : {r2_score(y_test, y_pred)}')
print(f'MAE score : {mean_absolute_error(y_test, y_pred)}')
print(f'MSE score: {mean_squared_error(y_test, y_pred)}')
print(f'RMSE score: {np.sqrt(mean_squared_error(y_test, y_pred))}')
r2 = r2_score(y_test, y_pred)
print(f'Adjusted score: {1-((1-r2)*(X_test.shape[0]-1)/(X_test.shape[0]-1-1))}')

plt.scatter(x = 'cgpa', y = 'package', data = df)
plt.plot(X_train, lr.predict(X_train), color = 'red')
plt.show()


