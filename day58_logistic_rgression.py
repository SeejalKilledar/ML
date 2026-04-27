import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
import numpy as np
from sklearn.linear_model import LogisticRegression
X,y = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0,
                          n_classes=2, n_clusters_per_class=1, random_state=41, hypercube=False, class_sep=10)
lr = LogisticRegression()
lr.fit(X,y)

"""
m = -(a/b)
b = -(c/b)

"""
m = -(lr.coef_[0][0]/lr.coef_[0][1])
b = -(lr.intercept_/lr.coef_[0][1])

x_input1 = np.linspace(-3,3,100)
y_input1 = m*x_input1 +b

plt.figure(figsize=(10,6))
plt.plot(x)