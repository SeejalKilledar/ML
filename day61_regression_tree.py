import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_boston

boston = load_boston()
df = pd.DataFrame(boston)

df.columns = boston.feature_names
df['MEDV'] = boston.target

df.head()

X = df.iloc[:,0:13]
y = df.iloc[:,13]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
rt = DecisionTreeClassifier(criterion='mse', max_depth=5)
rt.fit(X_train, y_train)
y_pred = rt.predict(X_test)
r2_score(y_test,y_pred)

print("************ Hyperparameters Tuning *************")
# use gridsearch or random search
param_grid = {
    'max_depth' : [2,4,8,10,None],
    'criterion' : ['mse', 'mae'],
    'max_features' : [0.25,0.5,1.0],
    'min_samples_split' : [0.25,0.5,1.0]
}

reg = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid)
reg.fit(X_train, y_train)
print(reg.best_score_)
print(reg.best_params_)

print("Feature Importance")

"""
feature_importance = While making the decision tree, it will calculate how many times the feature is used, it 
will calculate importance for every col

Uses : Can be used in Feature selection, (Dimensionality reduction)
"""

for importance, name in sorted(zip(rt.feature_importances_,X_train.columns),reverse=True):
    print(name, importance)
