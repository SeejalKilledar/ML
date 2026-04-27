import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from day24_standardization import y_pred

df = pd.read_csv('titanic.csv', usecols=['Age', 'Fare', 'Survived'])
print(df.head)

X = df.drop(columns = 'Survived')
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)

si = SimpleImputer(add_indicator=True) # add_indicator will automatically create missing indicator class
X_train_trf = si.fit_transform(X_train)
X_test_trf = si.transform(X_test)

clf = LogisticRegression()
clf.fit(X_train_trf, y_train)
y_pred = clf.predict(X_test_trf)
print(accuracy_score(y_test, y_pred))