from pyexpat import features
from sklearn.datasets import make_classification
import numpy as np
X,y = make_classification(n_samples=100, n_features=2, n_informative=1, n_redundant=0,
                          n_classes=2, n_clusters_per_class=1, random_state=41, hypercube=False, class_sep=10)

#print(y)

def perceptron(X,y):
    X = np.insert(X, 0, 1, axis=1) # creating bias value
    weights = np.ones(X.shape[1])
    lr = 0.1

    for i in range(1000):
        j = np.random.randint(0,100) # 100 coz the shape is 100
        y_hat =step(np.dot(X[j],weights))
        weights = weights + lr*(y[j]-y_hat)*X[j]

    return weights[0], weights[1:]

def step(z):
    return 1 if z > 0 else 0


