import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


print("Decision Tree")
np.random.seed(42)
X,y = make_circles(n_samples=500, factor=0.1, noise=0.35, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
plt.scatter(X[:,0], X[:,1], c = y)
plt.show()

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

plt.figure(figsize=(12,8))
x_range= np.linspace(X.min(), X.max(), 100)
xx1, xx2 = np.meshgrid(x_range, x_range)
y_hat = dt.predict(np.c_[xx1.ravel(), xx2.ravel()])
y_hat = y_hat.reshape(xx1.shape)
plt.contourf(xx1, xx2, y_hat, alpha=0.2)
plt.scatter(X[:,0], X[:,1], c= y, cmap='viridis', alpha=0.7)
plt.title("Decision Tree")
plt.show()

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)
plt.figure(figsize=(12,8))
x_range= np.linspace(X.min(), X.max(), 100)
xx1, xx2 = np.meshgrid(x_range, x_range)
y_hat = rf.predict(np.c_[xx1.ravel(), xx2.ravel()])
y_hat = y_hat.reshape(xx1.shape)
plt.contourf(xx1, xx2, y_hat, alpha=0.2)
plt.scatter(X[:,0], X[:,1], c= y, cmap='viridis', alpha=0.7)
plt.title("Random Forest")
plt.show()

print("Regression")
n_train = 150
n_test = 1000
noise = 0.1
def f(x):
    x = x.ravel()
    return np.exp(-x**2) + 1.5 * np.exp(-(x-2)**2)

def generate(n_samples, noise):
    X = np.random.rand(n_samples) * 10 -5
    X = np.sort(X).ravel()
    y = np.exp(-X**2) + 1.5 * np.exp(-(X-2)**2)\
        + np.random.normal(0.0, noise, n_samples)
    X = X.reshape((n_samples, 1))

    return X,y

X_train, y_train = generate(n_samples=n_train, noise=noise)
X_test, y_test = generate(n_samples=n_test, noise=noise)

plt.figure(figsize=(18,8))
plt.plot(X_test, f(X_test), "r")
plt.scatter(X_train, y_train, c='b', s= 20)
plt.xlim([-5,5])
plt.show()

print("Random Forest")
from sklearn.ensemble import RandomForestRegressor
rfe = RandomForestRegressor(n_estimators=1000)
rfe.fit(X_train, y_train)
rfe_predict =rfe.predict(X_test)
plt.figure(figsize=(18,8))
plt.plot(X_test, f(X_test), "r")
plt.scatter(X_train, y_train, c='b', s= 20)
plt.plot(X_test, rfe_predict, "g", lw = 2 )
plt.xlim([-5,5])
plt.title("Random Forest regressor")
plt.show()