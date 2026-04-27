import pandas as pd
import numpy as np
from numba import int32
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

df = pd.read_csv('covid_toy.csv')

print(df.head())


"""
Gender, City     : Nominal, OneHotEncoder
Cough            : Ordinal encoding
Fever            : Simple Imputer - has null values
has_covid        : Label encoding

"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
                                         df.drop('has_covid', axis = 1),
                                        df['has_covid'],
                                        test_size= 0.2, random_state=0)

# print(X_train)
# print(X_train.shape)
# print(X_test.shape)

# Fever            : Simple Imputer - has null values
si = SimpleImputer()
X_train_fever = si.fit_transform(X_train[['fever']])
X_test_fever = si.fit_transform(X_test[['fever']])
print(X_train_fever.shape)


# Cough            : Ordinal encoding
oe = OrdinalEncoder(categories=[['Mild','Strong']])
X_train_cough = oe.fit_transform(X_train[['cough']])
X_test_cough = oe.fit_transform(X_test[['cough']])
print(X_train_cough.shape)

# Gender, City     : Nominal, OneHotEncoder
ohe = OneHotEncoder(drop='first')
X_train_gender_city = ohe.fit_transform(X_train[['gender','city']]).toarray()
X_test_gender_city = ohe.fit_transform(X_test[['gender','city']]).toarray()
print("This is ...........")
print(X_train_gender_city)



"""
Now we have to contatenate fever, cough, gender and city to age
1st : Extract Age from the original df 
2nd : use np.concatenate  
"""

X_train_age = X_train.drop(columns = ['gender', 'fever', 'cough', 'city']).values
X_test_age = X_test.drop(columns = ['gender', 'fever', 'cough', 'city']).values

X_train_transformed = np.concatenate((X_train_age, X_train_fever, X_train_gender_city, X_train_cough),axis =1)
X_test_transformed = np.concatenate((X_test_age, X_test_fever, X_test_gender_city, X_test_cough),axis =1)
