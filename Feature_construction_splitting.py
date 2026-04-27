from itertools import groupby

import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns


df = pd.read_csv('titanic.csv', usecols=['Age', 'Pclass', 'SibSp', 'Parch', 'Survived'])
print(df)

print(df.isnull().mean()*100)
df.dropna(inplace = True)
print(df.isnull().mean()*100)

X = df.drop(columns='Survived')
y = df['Survived']

print(X.head())
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=2)
print(np.mean(cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=20)))

# Applying feature construction
print("*************** Feature Construction ***************************")
X['Family_size'] = X['SibSp'] + X['Parch'] +1
print(X.head())

def myfunc(num):
    if num == 1:
        return 0 # alome
    elif num > 1 and num <=4:
        return 1 # small family
    else:
        return 2

print(myfunc(5))
print(myfunc(1))
print(myfunc(2))

X['Family_type'] = X['Family_size'].apply(myfunc)

X.drop(columns=['SibSp', 'Parch', 'Family_size'], inplace = True)
print(X.head())
print(np.mean(cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=20)))

print("******************** Feature Splitting ******************************")
df = pd.read_csv('titanic.csv')
#print(df['Name'])
# Kelly, Mr. James
df['Title'] = df['Name'].str.split(", ", expand = True)[1].str.split(". ", expand = True)[0]
print(df['Title'], df['Name'])

# (df(groupby('Title').mean['Survived']).sort_values(ascending=True))
df['Is_Married'] = 0
df['Is_Married'].loc[df['Title'] == 'Mrs'] = 1

print(df['Is_Married'])

