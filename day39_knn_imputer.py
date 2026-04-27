import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('titanic.csv', usecols=['Age', 'Pclass','Fare', 'Survived'])
print(df.head())

X = df.drop(columns='Survived')
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=2)

print(X_train)

knn = KNNImputer(n_neighbors=3, weights='distance')
X_train_trf = knn.fit_transform(X_train)
X_test_trf = knn.transform(X_test)

lr = LogisticRegression()

lr.fit(X_train_trf, y_train)
y_predit = lr.predict(X_test_trf)
print(accuracy_score(y_test, y_predit))


si = SimpleImputer()
X_train_trf2 = si.fit_transform(X_train)
X_test_trf2 = si.transform(X_test)

lr = LogisticRegression()

lr.fit(X_train_trf2, y_train)
y_predit2 = lr.predict(X_test_trf2)
print(accuracy_score(y_test, y_predit2))