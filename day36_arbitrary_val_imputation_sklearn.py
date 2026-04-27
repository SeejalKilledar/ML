import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


df = pd.read_csv('titanic_toy.csv')
print(df.head())

print(df.isnull().mean())

X = df.drop(columns='Survived')
y = df['Survived']

X_train, X_test, y_train, y_test =train_test_split(X,y,test_size = 0.2,
                                                   random_state=2)


imputer1 = SimpleImputer(strategy='constant', fill_value=99)
imputer2 = SimpleImputer(strategy='constant', fill_value=999)

trf = ColumnTransformer(
    [

        ('imputer1', imputer1, ['Age']),
        ('imputer2', imputer2, ['Fare'])

    ], remainder='passthrough'

)
trf.fit(X_train)

print(trf.named_transformers_['imputer1'].statistics_)
print(trf.named_transformers_['imputer2'].statistics_)

X_train = trf.transform(X_train)
X_test = trf.transform(X_test)

print(X_train)