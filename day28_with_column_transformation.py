import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split


df = pd.read_csv('covid_toy.csv')

X_train, X_test, y_train, y_test = train_test_split(
                                         df.drop('has_covid', axis = 1),
                                        df['has_covid'],
                                        test_size= 0.2, random_state=0)

transform = ColumnTransformer(
    transformers= [
        ('tnf1', SimpleImputer(), ['fever']),
        ('tnf2', OrdinalEncoder(categories=[['Mild','Strong']]),['cough']),
        ('tnf3', OneHotEncoder(sparse_output=False, drop='first'),['gender', 'city'])
    ], remainder='passthrough'
)
print(transform.fit_transform(X_train))
print(transform.fit_transform(X_train).shape)