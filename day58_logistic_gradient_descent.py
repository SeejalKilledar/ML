from networkx.algorithms.bipartite import color
from sklearn.datasets import make_classification
import numpy as np



X, y = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0, n_classes=2,
                           n_clusters_per_class=1, random_state=41, hypercube=False, class_sep=20)

import matplotlib.pyplot as plt
plt.figure(figsize=(10,6))
plt.scatter(X[:,0], X[:,1], c= y,cmap='winter', s=100)
plt.show()

from sklearn.linear_model import LogisticRegression
lor = LogisticRegression(penalty=None, solver='sag')
lor.fit(X,y)
"""
Penalty : bydefault, Logistic reg uses penanlty, we are making this as nine, coz the method that we are goin
to write is without penalty
solver : this is used for optimization
"""
print(lor.coef_)
print(lor.intercept_)

m1 = -(lor.coef_[0][0]/lor.coef_[0][1])
b1 = -(lor.intercept_/lor.coef_[0][1])

x_input = np.linspace(-3,3,100)
y_input = m1*x_input + b1

def gd(X,y):
    X = np.insert(X,0,1, axis=1)
    weight = np.ones(X.shape[1])
    lr = 0.5

    for i in range(2500):
        y_hat = sigmoid(np.dot(X,weight))
        weight = weight + lr*(np.dot((y-y_hat),X)/X.shape[0])

    return weight[1:],weight[0]

def sigmoid(z):
    return 1/(1+np.exp(-z))

coef_, intercept = gd(X,y)
m = -(coef_[0]/coef_[1])
b = -(intercept/coef_[1])


x_input1 = np.linspace(-3,3,100)
y_input1 = m*x_input1 + b

plt.figure(figsize=(10,6))
plt.plot(x_input, y_input, color= 'red', linewidth = 3)
plt.plot(x_input1, y_input1, color= 'black', linewidth = 3)
plt.scatter(X[:,0], X[:,1], c= y,cmap='winter', s=100)
plt.ylim(-3,2) # focuses on the points -3 to 2 
plt.show()