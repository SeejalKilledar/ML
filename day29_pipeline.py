import pandas as pd
import numpy as np
from pandas import isnull

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import eImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('titanic.csv')

print(df.isnull().sum())

df = df.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin'])

X_train, X_test, y_train, y_test = train_test_split(
                                    df.drop('Survived', axis = 1),
                                    df['Survived'], test_size= 0.2,random_state= 0)

print(X_train.head())

# Simple Impute : Age and Embarked

# filling missing values
trf1 = ColumnTransformer(
    [

        ('impute_age', SimpleImputer(),[2]),
        ('impute_embarked', SimpleImputer(strategy='most_frequent'),[6])
    ], remainder='passthrough'
)

# One Hot Encoding\

trf2 = ColumnTransformer(
    [
        ('ohe_sex_embarked', OneHotEncoder(sparse_output=False, handle_unknown= "ignore"),[1,6])
    ],remainder='passthrough'

)

# scaling
trf3 = ColumnTransformer(
    [
        ('scale', MinMaxScaler(), slice(0,10))
    ]

)

# Feature Selection
trf4 = SelectKBest(score_func=chi2, k=8) #  k is top 8 columns

# Train the model
trf5 = DecisionTreeClassifier()

# create pipeline

# Pipeline : Pipeline requires naming of the steps
# make_pipeline : does not require naming of the steps
# pipe = make_pipeline(trf1,trf2,trf3,trf4,trf5,)
pipe = Pipeline([
    ('trf1', trf1),
    ('trf2', trf2),
    ('trf3', trf3),
    ('trf4', trf4),
    ('trf5', trf5)
]
)

#display pipeline
from sklearn import set_config
set_config(display='diagram')


# train
pipe.fit(X_train,y_train)

print(pipe.named_steps['trf1'].transformers_[0][1].statistics_)

# predict
y_pred = pipe.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(X_train)
# Cross Validation using pipeline
from sklearn.model_selection import cross_val_score
print(cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean())

# GridSearch using Pipeline
params = {
    'trf5__max_depth' : [1,2,3,4,5,None]
}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(grid.best_score_)
print(grid.best_params_)


# Exporting the pipeline
# How to use the file in production
import pickle
pickle.dump(pipe, open('pipe.pkl', 'wb'))

# check production file - predict_using_pipeline.py