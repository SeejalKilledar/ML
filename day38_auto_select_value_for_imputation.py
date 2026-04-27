import numpy as np
import pandas as pd
from Tools.demo.sortvisu import steps
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('titanic.csv', )
df.drop(columns=['Name', 'Ticket', 'PassengerId', 'Cabin'],inplace=True)
print(df.head())

print(df.isnull().mean()*100)

X = df.drop(columns='Survived')
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)

print(X_train.head())

numerical_features = ['Age', 'Fare']
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_features = ['Sex', 'Embarked']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('one', OneHotEncoder(handle_unknown='ignore'))
])


preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())

])

param_grid = {
    'preprocessor__num__imputer__strategy' : ['mean', 'median'],
    'preprocessor__cat__imputer__strategy' : ['most_frequent', 'constant'],
    'classifier__C' : [0.1,1.0,10,100] # these are decided for logistic regression

}

grid_search = GridSearchCV(clf, param_grid, cv = 10)


grid_search.fit(X_train, y_train)
print("Best parameter")
print(grid_search.best_params_)

print(f'Internal CV score {grid_search.best_score_:.3f}')

cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results = cv_results.sort_values('mean_test_score', ascending=False)
print(cv_results)
#print(cv_results[['param_classifier_c','preprocessor__cat__imputer__strategy','preprocessor__num__imputer__strategy','mean_test_score']])