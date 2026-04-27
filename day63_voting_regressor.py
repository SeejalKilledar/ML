# Voting classification in kaggle notebook
import numpy as np
# voting regression
from sklearn.datasets import load_diabetes
from sklearn.ensemble import VotingRegressor

X,y = load_diabetes(return_X_y=True)
print(X.shape)
print(y.shape)
print(X)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

lr = LinearRegression()
dt = DecisionTreeRegressor()
svr = SVR()

estimators = [('lr', lr), ('dt', dt), ('svr', svr)]
for estimator in estimators:
    scores = cross_val_score(estimator[1], X,y, cv=10, scoring='r2')
    print(estimator[0], np.round(np.mean(scores),2))

vr = VotingRegressor(estimators= estimators)
scores = cross_val_score(vr, X, y, cv = 10, scoring='r2')
print(f'VR {np.round(np.mean(scores),2)}')

# weights
for i in range(1,4):
    for j in range(1,4):
        for k in range(1,4):
            vr = VotingRegressor(estimators= estimators, weights = [i, j, k])
            scores = cross_val_score(vr, X, y, cv=10, scoring='r2')
            print(f'i={i}, j={j}, k={k}, {np.round(np.mean(scores),2)}')



# using same algorithm

dt1 = DecisionTreeRegressor(max_depth=1)
dt2 = DecisionTreeRegressor(max_depth=2)
dt3 = DecisionTreeRegressor(max_depth=3)
dt4 = DecisionTreeRegressor(max_depth=4)
dt5 = DecisionTreeRegressor(max_depth=5)

estimators = [('dt1', dt1),('dt2', dt2),('dt3', dt3),('dt4', dt4),('dt5', dt5)]
for estimator in estimators:
    scores = cross_val_score(estimator[1], X,y, cv= 10, scoring='r2')
    print(estimator[0], np.round(np.mean(scores),2))

vr = VotingRegressor(estimators)
scores = cross_val_score(vr, X,y, cv= 10, scoring='r2')
print(f"VR regressor,  {np.round(np.mean(scores),2)}")
