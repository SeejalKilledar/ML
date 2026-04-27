import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score



df = pd.read_csv('heart.csv')
print(df.head())

X = df.drop(columns='target')
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)

rf = RandomForestClassifier()
gb = GradientBoostingClassifier()
svm = SVC()
lr = LogisticRegression()

rf.fit(X_train, y_train)
rf_predict = rf.predict(X_test)
print("Random Forest")
print(accuracy_score(y_test, rf_predict))
#print(np.mean(cross_val_score(RandomForestClassifier(), X_train, y_train, cv = 5, scoring = 'accuracy')))

rft = RandomForestClassifier(max_samples=0.75, random_state=42)
rft.fit(X_train, y_train)
rft_predict = rft.predict(X_test)
print("Random Forest with Tuning")
print("Seejal")
print(accuracy_score(y_test, rft_predict))


gb.fit(X_train, y_train)
gb_predict = gb.predict(X_test)
print("Gradient boosting")
print(accuracy_score(y_test,gb_predict))

svm.fit(X_train, y_train)
svm_predict = svm.predict(X_test)
print("SVM")
print(accuracy_score(y_test,svm_predict))

lr.fit(X_train, y_train)
lr_predict = lr.predict(X_test)
print("Logistic regression")
print(accuracy_score(y_test,lr_predict))

print("GridSearch CV")
param_grid = {
    'n_estimators' : [20, 50, 100],
    'max_features' : [0.2, 0.6, 1.0],
    'max_depth' : [2, 8, None],
    'max_samples' : [0.5, 0.75, 1.0]
}


rf_grid = GridSearchCV(estimator= rf, param_grid=param_grid, cv = 5, verbose=2, n_jobs=-1)
rf_grid.fit(X_train, y_train)
print(rf_grid.best_params_)
print(rf_grid.best_score_)

print("Random Search Cv")
param_grid_1 = {
    'n_estimators' : [20, 50, 100],
    'max_features' : [0.2, 0.6, 1.0],
    'max_depth' : [2, 8, None],
    'max_samples' : [0.5, 0.75, 1.0],
    'bootstrap' : [True, False],
    'min_samples_split': [2,5],
    'min_samples_leaf':[1,2]
}
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = param_grid_1, cv = 5, verbose=2, n_jobs=-1)

rf_random.fit(X_train, y_train)
print(rf_random.best_params_)
print(rf_random.best_score_)