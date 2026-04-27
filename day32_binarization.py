import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer



df = pd.read_csv('titanic.csv', usecols=['Age', 'Fare', 'SibSp', 'Parch', 'Survived'])

df.dropna(inplace=True)

print(df.head())

print(df.isnull().sum())

df['family'] = df['SibSp'] + df['Parch']
df = df.drop(columns=['SibSp', 'Parch'])
print(df.head())

X = df.drop(columns='Survived')
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
print(accuracy_score(y_test, y_predict))

print(np.mean(cross_val_score(clf, X,y, cv=10, scoring = 'accuracy')))


from sklearn.preprocessing import Binarizer

trf = ColumnTransformer(

    [
        ('bin', Binarizer(copy=False),['family'])
    ], remainder='passthrough'
)

X_train_trf = trf.fit_transform(X_train)
X_test_trf = trf.fit_transform(X_test)

print(pd.DataFrame(X_train_trf, columns = ['family', 'Age', 'Fare']))


clf = DecisionTreeClassifier()
clf.fit(X_train_trf, y_train)
y_predict_1 = clf.predict(X_test_trf)
print(accuracy_score(y_test, y_predict_1))

print(np.mean(cross_val_score(clf, X,y, cv=10, scoring = 'accuracy')))