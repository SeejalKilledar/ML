import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


df = pd.read_csv('placement_simple_linear_reg.csv')
print(df.head())
plt.scatter(df['cgpa'], df['package'])
plt.xlabel('CGPA')
plt.ylabel('Package')
plt.show()

X = df.drop(columns='package')
y = df['package']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=2)

print(X_train)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_predict = lr.predict(X_test)

print(r2_score(y_test, y_predict))

plt.scatter(df['cgpa'], df['package'])
plt.plot(X_test, lr.predict(X_test), color = 'red') # best fit line
#plt.plot(X_train, lr.predict(X_train), color = 'red')
plt.xlabel('CGPA')
plt.ylabel('Package')
plt.show()

"""
X_test
y_test
lr.predict(X_test.iloc[1].values.reshape(1,1))

"""

# slope value
m = lr.coef_

# intercept
b = lr.intercept_

# y = mx+b
"""
m * 8.58 + b
"""