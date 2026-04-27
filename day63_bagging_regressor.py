from sklearn.datasets import load_diabetes
from statsmodels.sandbox.distributions.gof_new import bootstrap

from day63_voting_regressor import estimator

diabetes = load_diabetes()
X_dia, y_dia = diabetes.data, diabetes.target
print(f'Feature name : {diabetes.feature_names}')
#Feature name : ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import BaggingRegressor

X_train, x_test, y_train,y_test = train_test_split(X_dia, y_dia, test_size=0.20, random_state=42)
print(f'Train/test split - {X_train.shape} {x_test.shape} {y_train.shape} {y_test.shape}')

lr = LinearRegression()
knn = KNeighborsRegressor()
dt = DecisionTreeRegressor()

lr.fit(X_train,y_train)
knn.fit(X_train,y_train)
dt.fit(X_train,y_train)

y_pred1 = lr.predict(x_test)
y_pred2 = knn.predict(x_test)
y_pred3 = dt.predict(x_test)

print(f' Linear Regression : {r2_score(y_test,y_pred1)}')
print(f'K-neighbor : {r2_score(y_test,y_pred2)}')
print(f'Decision Tree {r2_score(y_test,y_pred3)}')


br = BaggingRegressor(random_state=1)
br.fit(X_train,y_train)
y_predit1 = br.predict(x_test)
print(f"training coeffs: {br.score(X_train,y_train)}")
print(f"test coeffs: {br.score(x_test,y_test)}")

print("GridCV")
params = {
    'estimator' : [None, LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor()],
    'n_estimators' : [20,50,100],
    'max_samples': [0.5,1.0],
    'max_features': [0.5,1.0],
    'bootstrap' : [True, False],
    'bootstrap_features': [True, False]
}

bagging_grid = GridSearchCV(BaggingRegressor(random_state=1, n_jobs=1), param_grid=params, cv= 10, n_jobs=1, verbose=1)
bagging_grid.fit(X_train,y_train)

print(f"Training: {bagging_grid.best_estimator_.score(X_train, y_train)}")
print(f"Testing : {bagging_grid.best_estimator_.score(x_test, y_test)}")
print(f"Best score : {bagging_grid.best_score_}")
print(f"Best params : {bagging_grid.best_params_}")
